from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import math
from vgg import vgg16
from densenet import densenet169
from torch.autograd import Variable
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.args = args
        self.layer0 = nn.Sequential(*list(vgg16(pretrained=True).features.children())[0:4])    # 256*256, 64
        self.layer1 = nn.Sequential(*list(vgg16(pretrained=True).features.children())[4:9])    # 128*128, 128
        self.layer2 = nn.Sequential(*list(vgg16(pretrained=True).features.children())[9:16])    # 64*64, 256
        self.layer3 = nn.Sequential(*list(vgg16(pretrained=True).features.children())[16:23])   # 32*32, 512
        self.layer4 = nn.Sequential(*list(vgg16(pretrained=True).features.children())[23:30])  # 16*16, 512

        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        M = 256
        self.fc6 = nn.Conv2d(M, 512, kernel_size=3, padding=1, dilation=1)  # fc6
        self.relu6 = nn.ReLU(inplace=True)
        # nn.Dropout(0.5),
        self.fc7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)  # fc7
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Conv2d(512, args.num_classes, kernel_size=1, padding=0)  # fc8
        # self.out_size = 14

    def create_templates(self, ind, h):
        # create templates for all filters
        center = torch.stack([torch.div(ind, h), torch.remainder(ind, h)], 0).t()
        templates = torch.zeros(ind.size(0), h, h)
        tau = 0.5 / h ** 2

        for i in range(h):
            for j in range(h):
                temp = torch.FloatTensor([i, j]).view(1, 2).cuda()
                norm = (temp - center.type(torch.float)).norm(self.args.norm_template, -1)
                out = tau * torch.clamp(1 - 4 * norm / h, min=-1)
                templates[:, i, j] = out.view(1, -1).type(torch.float)
        return templates.cuda()

        # self.templates_f = Variable(templates.cuda(), requires_grad=False)
        # neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
        # templates = torch.cat([templates, neg_template], 0)
        # self.templates_b = Variable(templates.cuda(), requires_grad=False)

    def get_masked_output(self, x):
        # choose template that maximize activation and return x_masked
        h = x.size(3)
        indices = F.max_pool2d(x, h, return_indices=True)[1].squeeze()
        selected_templates = torch.stack([self.create_templates(ind, h) for ind in indices], 0)
        selected_templates = Variable(selected_templates, requires_grad=False)
        x_masked = F.relu(x * selected_templates)
        return x_masked

    # def compute_local_loss(self, x):
    #     n_square = x.size(3)**2
    #     weight = 1 / (1 + n_square)
    #     p_T = [weight for _ in range(n_square)]
    #     p_T = Variable(torch.FloatTensor(p_T.append(weight)).cuda(), requires_grad=False)
    #     x = x.permute(1, 0, 2, 3)
    #     exp_tr_x_T = (x[:, :, None, :, :] * self.templates_b[None, None, :, :, :]).sum(-1).sum(-1).exp()
    #     Z_T = exp_tr_x_T.sum(1, keepdim=True)
    #     p_x_T = exp_tr_x_T / Z_T
    #
    #     p_x = (p_T[None, None, :] * p_x_T).sum(-1)
    #     p_x_T_log = (p_x_T * torch.log(p_x_T/p_x[:, :, None])).sum(1)
    #     loss = - (p_T[None, :] * p_x_T_log).sum(-1)
    #     return loss

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x, train=True):
        # bottom-up
        c0 = self.layer0(x)        # 256
        c1 = self.layer1(c0)       # 128
        c2 = self.layer2(c1)       # 64
        c3 = self.layer3(c2)       # 32
        c4 = self.layer4(c3)       # 16

        # top-down
        p4 = self.toplayer(c4)     # 16*16, 256
        p3 = self._upsample_add(p4, self.latlayer1(c3))   # 32
        p2 = self._upsample_add(p3, self.latlayer2(c2))   # 64
        p1 = self._upsample_add(p2, self.latlayer3(c1))   # 128
        # smooth
        p3 = self.smooth1(p3)  # 32*32, 256
        p2 = self.smooth2(p2)  # 64*64, 256
        p1 = self.smooth3(p1)  # 128*128, 256

        # cls branch4
        x = self.relu6(self.fc6(p4))
        x = self.relu7(self.fc7(x))
        x4 = self.fc8(x)
        out4 = F.avg_pool2d(x4, kernel_size=14, stride=1).view(x.size(0), -1)

        # cls branch3
        x = self.relu6(self.fc6(p3))
        x = self.relu7(self.fc7(x))
        # mask32 = self.get_masked_output(x)
        x3 = self.fc8(x)
        out3 = F.avg_pool2d(x3, kernel_size=28, stride=1).view(x.size(0), -1)

        # cls branch2
        # p1 = F.avg_pool2d(p2, kernel_size=25, stride=1, padding=12)
        x = self.relu6(self.fc6(p2))
        # mask21 = self.get_masked_output(x)
        x = self.relu7(self.fc7(x))
        # mask22 = self.get_masked_output(x)
        x2 = self.fc8(x)
        out2 = F.avg_pool2d(x2, kernel_size=56, stride=1).view(x.size(0), -1)

        # cls branch1
        # p1 = F.avg_pool2d(p1, kernel_size=25, stride=1, padding=12)
        x = self.relu6(self.fc6(p1))
        # mask11 = self.get_masked_output(x)
        x = self.relu7(self.fc7(x))
        # mask12 = self.get_masked_output(x)
        x1 = self.fc8(x)
        out1 = F.avg_pool2d(x1, kernel_size=112, stride=1).view(x.size(0), -1)
        # out = 0.25*(out1+out2+out3+out4)

        # compute local loss:
        # loss_1 = self.compute_local_loss(mask11)+self.compute_local_loss(mask21) \
        #          + self.compute_local_loss(mask31)+self.compute_local_loss(mask41)
        # loss_2 = self.compute_local_loss(mask12) + self.compute_local_loss(mask22) \
        #          + self.compute_local_loss(mask32) + self.compute_local_loss(mask42)
        loss_1, loss_2 = 0, 0
        return out1, out2, out3, out4, loss_1, loss_2


if __name__ == '__main__':
    model = Model()
