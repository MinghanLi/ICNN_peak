import os
from os.path import join
from os import getcwd
import argparse
from torchvision import models, transforms
from DataRead import MuraDataset
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import torch
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data_dir', default='MURA-v1.1', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', default='densenet121', choices=model_names, help='nn architecture')
parser.add_argument('--resume', default='checkpoint1.pth.tar', type=str)
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')
parser.add_argument('--workers', default=4, type=int)
global args
args = parser.parse_args()
tb_writer = SummaryWriter()


def main():
    if 'squeezenet' in args.arch:
        net = models.squeezenet1_1(pretrained=True)
    elif 'resnet' in args.arch:
        net = models.resnet18(pretrained=True)
    elif 'dense' in args.arch:
        net = models.densenet121(pretrained=True)
        net.classifier = nn.Linear(1024, 2)
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                net.load_state_dict(checkpoint['state_dict'])

    net.eval()

    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize ])

    data_dir = join(getcwd(), args.data_dir)
    test_dir = join(data_dir, 'valid/XR_SHOULDER')
    test_csv = join(data_dir, 'valid_shoulder.csv')
    test_data = MuraDataset(test_csv, transform=train_transforms)
    weights = test_data.balanced_weights
    weights = torch.DoubleTensor(weights)
    sampler = data.sampler.WeightedRandomSampler(weights, len(weights))

    test_loader = data.DataLoader(test_data,
                                  batch_size=args.batch_size,
                                  # shuffle=True,
                                  num_workers=args.workers,
                                  sampler=sampler,
                                  pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    val_loss = validate(test_loader, net, criterion)


def validate(test_loader, model, criterion):
    model.eval()
    acc = AverageMeter()
    losses = AverageMeter()
    meta_data = []
    pbar = tqdm(test_loader)
    for i, (images, target, meta) in enumerate(pbar):
        # target = target.cuda(async=True)
        image_var = Variable(images, volatile=True)
        label_var = Variable(target, volatile=True)

        y_pred = model(image_var)
        # udpate loss metric
        loss = criterion(y_pred, label_var)
        losses.update(loss.data[0], images.size(0))

        # update accuracy metric on the GPU
        prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        acc.update(prec1[0], images.size(0))

        sm = nn.Softmax()
        sm_pred = sm(y_pred).data.cpu().numpy()
        # y_norm_probs = sm_pred[:, 0] # p(normal)
        y_pred_probs = sm_pred[:, 1]  # p(abnormal)

        meta_data.append(
            pd.DataFrame({
                'img_filename': meta['img_filename'],
                'y_true': meta['y_true'].numpy(),
                'y_pred_probs': y_pred_probs,
                'patient': meta['patient'].numpy(),
                'study': meta['study'].numpy(),
                'image_num': meta['image_num'].numpy(),
                'encounter': meta['encounter'],
            }))

        pbar.set_description("Test[{}/{}]".format(i, len(test_loader)))
        pbar.set_postfix(
            acc="{acc.test:.4f} ({acc.avg:.4f})".format(acc=acc),
            loss="{loss.test:.4f} ({loss.avg:.4f})".format(loss=losses))
    df = pd.concat(meta_data)
    ab = df.groupby(['encounter'])['y_pred_probs', 'y_true'].mean()
    ab['y_pred_round'] = ab.y_pred_probs.round()
    ab['y_pred_round'] = pd.to_numeric(ab.y_pred_round, downcast='integer')

    f1_s = f1_score(ab.y_true, ab.y_pred_round)
    prec_s = precision_score(ab.y_true, ab.y_pred_round)
    rec_s = recall_score(ab.y_true, ab.y_pred_round)
    acc_s = accuracy_score(ab.y_true, ab.y_pred_round)
    tb_writer.add_scalar('test/f1_score', f1_s)
    tb_writer.add_scalar('test/precision', prec_s)
    tb_writer.add_scalar('test/recall', rec_s)
    tb_writer.add_scalar('test/accuracy', acc_s)
    # return the metric we want to evaluate this model's performance by
    return f1_s


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.test = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, test, n=1):
        self.test = test
        self.sum += test * n
        self.count += n
        self.avg = self.sum / self.count


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
