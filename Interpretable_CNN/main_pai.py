import argparse
from model_pai import Model
from os.path import join
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
import torch.utils.data.distributed
from DataRead import MuraDataset
import torchvision.transforms as transforms


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


def loss_fn(x, labels):
    main_loss = F.cross_entropy(x, labels)
    return main_loss


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
    parser = argparse.ArgumentParser(description='CapsNet')

    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-num_epochs', type=int, default=10)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('--data_csv_dir', default='/data/volume2', metavar='DIR',
                        help='path to dataset csv paths')
    parser.add_argument('-pretrained', type=str, default="")
    parser.add_argument('--dataset', type=str, default='CUB', metavar='N',
                        help='name of dataset: SmallNORB or MNIST or NORB or CIFAR10')
    parser.add_argument('-gpu', type=int, default=0, help="which gpu to use")
    parser.add_argument('-num_classes', type=int, default=2, help="how types")
    parser.add_argument('--loss', type=str, default='margin_loss', metavar='N',
                        help='loss to use: cross_entropy_loss, margin_loss, spread_loss')
    parser.add_argument('--routing', type=str, default='angle_routing', metavar='N',
                        help='routing to use: angle_routing, EM_routing, quickshift_routing, '
                             'reduce_noise_angle_routing')
    parser.add_argument('--use-recon', type=bool, default=True, metavar='N',
                        help='use reconstruction loss or not')
    parser.add_argument('--use-additional-loss', type=int, default=0, metavar='B',
                        help='use additional loss: 0: none, 1: contrastive, 2: lifted loss')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    parser.add_argument('-clip', type=float, default=5)
    parser.add_argument('-pai', type=bool, default=False)
    parser.add_argument('--growthRate', type=int, default=12, metavar='N',
                        help='Growth rate for DenseNet.')
    parser.add_argument('--depth', type=int, default=110, help='Model depth.')
    parser.add_argument('--norm_template', type=int, default=1, help='Norm of the template')
    parser.add_argument('--multi-abstract', type=bool, default=False, metavar='N',
                        help='use multi level of abstraction or not')
    parser.add_argument('--dist_backend', default='mpi', type=str, help='distributed backend')

    args = parser.parse_args()
    dist.init_process_group(backend=args.dist_backend)

    data_csv_dir = args.data_csv_dir
    train_csv = join(data_csv_dir, 'train.csv')
    val_csv = join(data_csv_dir, 'valid.csv')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # We augment by applying random lateral inversions and rotations.
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    print('=>load traindata')
    train_dataset = MuraDataset(train_csv, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    test_dataset = MuraDataset(val_csv, transform=train_transforms)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               # shuffle=True,
                                               sampler=train_sampler,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              sampler=test_sampler,
                                              # shuffle=False,
                                              pin_memory=True)

    model = Model(args)
    model = torch.nn.parallel.DistributedDataParallel(model.cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print("# parameters:", sum(param.numel() for param in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    steps, lambda_ = len(train_dataset) // args.batch_size, 5e-6,

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        m = 0.8
        lambda_ = 0.9

    acc_train, acc_val = [], []
    loss_train, loss_val = [], []
    best_val_loss = 0
    for epoch in range(args.num_epochs):
        # Train
        # print("Epoch {}".format(epoch))
        losses, acc = 0, 0
        pbar = tqdm(train_loader)
        for i, (imgs, labels, meta) in enumerate(pbar):
            labels = labels.cuda(async=True)
            imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
            labels = labels.squeeze()
            out, x1, x2, loss_1, loss_2 = model(imgs)
            loss = loss_fn(out, labels)
            loss = loss + lambda_ * (loss_1.sum() + loss_2.sum())
            losses += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(out.data, labels.data, topk=(1, 1))
            acc += prec1[0]
        pbar.close()
        losses = losses/(i+1)
        acc = acc/(i+1)
        loss_train.append(losses)
        acc_train.append(acc)
        print("\nEpoch%d Train acc:%.4f, loss:%.4f" % (epoch, acc, losses))
        scheduler.step(losses)

        # Test
        # print('Testing...')
        losses, acc = 0, 0
        model.eval()
        pbar2 = tqdm(test_loader)
        for i, (imgs, labels, meta) in enumerate(pbar2):
            with torch.no_grad():
                labels = labels.cuda(async=True)
                imgs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
                labels = labels.squeeze()
                out, x1, x2, loss_1, loss_2 = model(imgs)
                loss = loss_fn(out, labels)
                losses += loss

            prec1 = accuracy(out.data, labels.data, topk=(1, 1))
            acc += prec1[0]
        pbar2.close()
        losses = losses/(i+1)
        acc = acc/(i+1)
        loss_val.append(losses)
        acc_val.append(acc)
        # if epoch == 0:
        #     best_val_loss = losses
        # is_best = losses < best_val_loss
        # best_val_loss = min(losses, best_val_loss)
        # if is_best:
        torch.save(model.state_dict(), '/data/oss_bucket/' + str(epoch) + 'ICNN_MURA.pth.tar')
        print("Epoch%d Test acc:%.4f, loss:%.4f" % (epoch, acc, losses))
        model.train()




