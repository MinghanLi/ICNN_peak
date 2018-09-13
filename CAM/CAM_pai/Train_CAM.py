from __future__ import absolute_import, division, print_function

import argparse
import shutil
import os
from os.path import join

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from densenet import densenet169
import torch.distributed as dist
import torch.utils.data.distributed

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from DataRead import MuraDataset

print("torch : {}".format(torch.__version__))
print("torch vision : {}".format(torchvision.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data_dir', default='/data/Volume1', metavar='DIR', help='path to dataset')
parser.add_argument('--data_csv_dir', default='/data/volume2', metavar='DIR', help='path to dataset csv paths')
parser.add_argument('--arch', default='densenet169', choices=model_names, help='nn architecture')
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=.1, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--pretrained', default='Ture', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--fullretrain', dest='fullretrain', action='store_true', help='retrain all layers of the model')
parser.add_argument('--seed', default=1337, type=int, help='random seed')
parser.add_argument('--dist_rank', default=0, type=int, help='rank of distributed processes')
parser.add_argument('--dist_backend', default='mpi', type=str, help='distributed backend')

best_val_loss = 0

tb_writer = SummaryWriter()


def main():
    global args, best_val_loss
    args = parser.parse_args()
    dist.init_process_group(backend=args.dist_backend)
    print("=> setting random seed to '{}'".format(args.seed))
    np.random.seed(args.seed)     # 1337
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = densenet169(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        if 'resnet' in args.arch:
            # for param in model.layer4.parameters():
            model.fc = nn.Linear(2048, args.classes)

        if 'dense' in args.arch:
            if '121' in args.arch:
                # (classifier): Linear(in_features=1024)
                model.classifier = nn.Linear(1024, args.classes)
            elif '169' in args.arch:
                # (classifier): Linear(in_features=1664)
                model.classifier = nn.Linear(1664, args.classes)
            else:
                return

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # GPU accurate
    model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    net = model.module
    # optionally resume from a checkpoint, save model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> found checkpoint")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])

            args.epochs = args.epochs + args.start_epoch
            print("=> loading checkpoint '{}' with acc of '{}'".format(
                args.resume,
                checkpoint['best_val_loss'], ))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_csv_dir = args.data_csv_dir
    train_csv = join(data_csv_dir, 'train.csv')
    val_csv = join(data_csv_dir, 'valid.csv')

    # Before feeding images into the network, we normalize each image to have
    # the same mean and standard deviation of images in the ImageNet training set.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # We augment by applying random lateral inversions and rotations.
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    print('=>load traindata')
    train_data = MuraDataset(train_csv, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   # shuffle=True,
                                   num_workers=args.workers,
                                   sampler=train_sampler,
                                   pin_memory=True)
    print('=>load testdata')
    val_loader = data.DataLoader(
        MuraDataset(val_csv,
                    transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.fullretrain:
        print("=> optimizing all layers")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(net.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        print("=> optimizing fc/classifier layers")
        optimizer = optim.Adam(net.classifier.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True)
    acc_train, acc_val = [], []
    loss_train, loss_val = [], []
    best_val_loss = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        tr_acc, tr_loss = train(train_loader, net, criterion, optimizer, epoch)
        acc_train.append(tr_acc)
        loss_train.append(tr_loss)
        # evaluate on validation set
        val_acc, val_loss = validate(val_loader, net, criterion, epoch)
        acc_val.append(val_acc)
        loss_val.append(val_loss)
        scheduler.step(val_loss)
        # remember best Accuracy and save checkpoint
        # if epoch == 0:
        #     best_val_loss = val_loss
        # is_best = val_loss < best_val_loss
        # best_val_loss = min(val_loss, best_val_loss)
        # if is_best:
        torch.save(model.state_dict(), '/data/oss_bucket/' + str(epoch) + 'CAM_MURA.pth.tar')

    print(acc_train, loss_train, acc_val, loss_val)


def train(train_loader, model, criterion, optimizer, epoch):
    losses, acc = 0, 0

    # ensure model is in train mode
    model.train()
    pbar = tqdm(train_loader)
    for i, (images, target, meta) in enumerate(pbar):
        target = target.cuda(async=True)
        image_var = Variable(images.cuda())
        label_var = Variable(target.cuda())

        # pass this batch through our model and get y_pred
        y_pred = model(image_var)

        # update loss metric
        loss = criterion(y_pred, label_var)
        losses += loss
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        acc += prec1[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = losses/(i+1)
    acc = acc/(i+1)
    pbar.close()
    print('\nEpoch %d, Train acc:%.4f ,loss:%.4f' %(epoch, acc, losses))
    # print('Finished training')
    return acc, losses


def validate(val_loader, model, criterion, epoch):
    model.eval()
    losses, acc = 0, 0
    pbar = tqdm(val_loader)
    for i, (images, target, meta) in enumerate(pbar):
        with torch.no_grad():
            target = target.cuda(async=True)
            image_var = Variable(images.cuda())
            label_var = Variable(target.cuda())

            y_pred = model(image_var)
            # udpate loss metric
            loss = criterion(y_pred, label_var)
            losses += loss

        # update accuracy metric on the GPU
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        acc +=prec1[0]

    losses = losses/(i+1)
    acc = acc/(i+1)
    pbar.close()
    print('\nEpoch%d, Valid acc:%.4f ,loss:%.4f' %(epoch, acc, losses))
    # print('Finished valing')
    # return the metric we want to evaluate this model's performance by
    return acc, losses


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
