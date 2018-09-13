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

        self.conv1 = nn.Conv2d(1408, 512, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1)  # fc6
        self.relu6 = nn.ReLU(inplace=True)
        # nn.Dropout(0.5),
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1)  # fc7
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Conv2d(1024, args.num_classes, kernel_size=1, padding=0)  # fc8
        # self.deconv = nn.ConvTranspose2d(args.num_classes, args.num_classes, 32, stride=32, bias=False)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(args.num_classes, affine=False)
        self.out_size = 16

        # self.index = torch.FloatTensor([[i, j] for i in range(512) for j in range(512)])
        self.index = torch.Tensor([[i, j] for i in range(-9, 10) for j in range(-9, 10)])

        # create templates for all filters
        self.mus = torch.FloatTensor([[i, j] for i in range(self.out_size) for j in range(self.out_size)])
        templates = torch.zeros(self.mus.size(0), self.out_size, self.out_size)

        n_square = self.out_size * self.out_size

        tau = 0.5 / n_square
        alpha = n_square / (1 + n_square)
        beta = 4

        for k in range(templates.size(0)):
            for i in range(self.out_size):
                for j in range(self.out_size):
                    if k < templates.size(0) - 1:  # positive templates
                        norm = (torch.FloatTensor([i, j]) - self.mus[k]).norm(self.args.norm_template, -1)
                        out = tau * torch.clamp(1 - beta * norm / self.out_size, min=-1)
                        templates[k, i, j] = float(out)
        if args.pai:
            self.templates_f = Variable(templates.cuda(), requires_grad=False)
            neg_template = -tau * torch.ones(1, self.out_size, self.out_size)
            templates = torch.cat([templates, neg_template], 0)
            self.templates_b = Variable(templates.cuda(), requires_grad=False)
            p_T = [alpha / n_square for _ in range(n_square)]
            p_T.append(1 - alpha)
            self.p_T = Variable(torch.FloatTensor(p_T).cuda(), requires_grad=False)
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

    def get_heatmap_GT(self):
        peak_point = self.args.peak_point
        heatmap_GT = []
        for peak in peak_point:
            if list(peak) == [1000, 1000]:
                temp = torch.ones(512, 512)
            else:
                temp = torch.zeros(512, 512)
                idx = self.index + peak  # peak: floattensor
                for idxk in idx:
                    idxk = idxk.type(torch.long)
                    temp[idxk] = 1
                # temp = ((-((self.index - peak)**2).sum(1) / 50).exp() / 5).reshape(512, 512)  # sigma = 5
            heatmap_GT.append(temp)
        heatmap_GT = torch.stack(heatmap_GT, 0).reshape(peak_point.size(0), 1, 512, 512)  # (8,1,512,512)
        heatmap_GT = Variable(heatmap_GT.type(torch.long).cuda())
        return heatmap_GT

    def get_peak_loss(self, heatmap):
        heatmap_pred = heatmap.view(8, 2, -1)
        heatmap_GT = self.get_heatmap_GT().view(8, -1)  # torch.cuda.floattensor
        loss = F.cross_entropy(heatmap_pred, heatmap_GT)  # GT need longtensor
        return loss

    def forward(self, x, train=True):
        # bottom-up
        c0 = self.layer0(x)        # 256
        c1 = self.layer1(c0)       # 128
        c2 = self.layer2(c1)       # 64
        c3 = self.layer3(c2)       # 32
        c4 = self.layer4(c3)       # 16

        # skip connection
        p3 = F.avg_pool2d(c3, kernel_size=2, stride=2)
        p2 = F.avg_pool2d(c2, kernel_size=4, stride=4)
        p1 = F.avg_pool2d(c1, kernel_size=8, stride=8)
        x0 = self.conv1(torch.cat((c4, p3, p2, p1), 1))
        x1 = self.get_masked_output(x0)
        x = self.relu6(self.fc6(x1))
        x2 = self.get_masked_output(x)
        x = self.relu7(self.fc7(x2))
        x3 = self.fc8(x)
        out = F.avg_pool2d(x3, kernel_size=16, stride=1).view(x.size(0), -1)

        # seg branch
        x4 = self.bn(self.upsample(x3))  # 512

        # compute local loss:
        loss_1 = self.compute_local_loss(x1)
        loss_2 = self.compute_local_loss(x2)
        loss_peak = self.get_peak_loss(x4)

        return out, loss_1, loss_2, loss_peak


if __name__ == '__main__':
    model = Model()
