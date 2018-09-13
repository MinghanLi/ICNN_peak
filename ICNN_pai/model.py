from torchvision import models
import torch.nn as nn
import torch
from vgg import vgg16
from densenet import densenet169
from torch.autograd import Variable
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.args = args
        self.base_model = nn.Sequential(*list(vgg16(pretrained=True).features.children())[:-1])
        M = self.base_model[-2].out_channels
        self.fc6 = nn.Conv2d(M, 1024, kernel_size=3, padding=1, dilation=1)  # fc6
        self.relu6 = nn.ReLU(inplace=True)
        # nn.Dropout(0.5),
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1)  # fc7
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Conv2d(1024, args.num_classes, kernel_size=1, padding=0)  # fc8
        self.out_size = 14

        # create templates for all filters
        mus = torch.FloatTensor([[i, j] for i in range(self.out_size) for j in range(self.out_size)])
        templates = torch.zeros(mus.size(0), self.out_size, self.out_size)

        n_square = self.out_size * self.out_size

        tau = 0.5 / n_square
        alpha = n_square / (1 + n_square)
        beta = 4

        for k in range(templates.size(0)):
            for i in range(self.out_size):
                for j in range(self.out_size):
                    if k < templates.size(0) - 1:  # positive templates
                        norm = (torch.FloatTensor([i, j]) - mus[k]).norm(self.args.norm_template, -1)
                        out = tau * torch.clamp(1 - beta * norm / self.out_size, min=-1)
                        templates[k, i, j] = float(out)
        if args.pai:
            self.templates_f = Variable(templates, requires_grad=False).cuda()
            neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
            templates = torch.cat([templates, neg_template], 0)
            self.templates_b = Variable(templates, requires_grad=False).cuda()
            p_T = [alpha / n_square for _ in range(n_square)]
            p_T.append(1 - alpha)
            self.p_T = Variable(torch.FloatTensor(p_T), requires_grad=False).cuda()
        else:
            self.templates_f = Variable(templates, requires_grad=False)
            neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
            templates = torch.cat([templates, neg_template], 0)
            self.templates_b = Variable(templates, requires_grad=False)
            p_T = [alpha / n_square for _ in range(n_square)]
            p_T.append(1 - alpha)
            self.p_T = Variable(torch.FloatTensor(p_T), requires_grad=False)

    def get_masked_output(self, x):
        # choose template that maximize activation and return x_masked
        indices = F.max_pool2d(x, self.out_size, return_indices=True)[1].squeeze()
        selected_templates = torch.stack([self.templates_f[i] for i in indices], 0)
        x_masked = F.relu(x * selected_templates)
        return x_masked

    def compute_local_loss(self, x):
        x = x.permute(1, 0, 2, 3)
        exp_tr_x_T = (x[:, :, None, :, :] * self.templates_b[None, None, :, :, :]).sum(-1).sum(-1).exp()
        Z_T = exp_tr_x_T.sum(1, keepdim=True)
        p_x_T = exp_tr_x_T / Z_T

        p_x = (self.p_T[None, None, :] * p_x_T).sum(-1)
        p_x_T_log = (p_x_T * torch.log(p_x_T/p_x[:, :, None])).sum(1)
        loss = - (self.p_T[None, :] * p_x_T_log).sum(-1)
        return loss

    def forward(self, x, train=True):
        x = self.base_model(x)
        x0 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x1 = self.get_masked_output(x0)
        x = self.relu6(self.fc6(x1))
        x2 = self.get_masked_output(x)
        x = self.relu7(self.fc7(x2))
        x3 = self.fc8(x)
        x = F.avg_pool2d(x3, kernel_size=14, stride=1).view(x.size(0), -1)

        # compute local loss:
        loss_1 = self.compute_local_loss(x1)
        loss_2 = self.compute_local_loss(x2)

        return x, x3, loss_1, loss_2


if __name__ == '__main__':
    model = Model()
