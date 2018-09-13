# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import os
import cv2
import argparse
from os import getcwd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
to_pil = ToPILImage()
to_tensor = ToTensor()

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data_dir', default='MURA-v1.1', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', default='densenet169', choices=model_names, help='nn architecture')
parser.add_argument('--resume', default='13CAM_MURA.pth.tar', type=str)
global args
args = parser.parse_args()
# networks such as googlenet, resnet, densenet already use global average pooling at the end,
# so CAM could be used directly.
if 'squeezenet' in args.arch:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features'     # this is the last conv layer of the network
elif 'resnet' in args.arch:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
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
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

data_dir ='/Users/suqi.lmh/Musculoskeletal-Radiographs-Abnormality-Classifier/pytorch/MURA-v1.1/test/Finger/patient11253/study1_positive/image2.png'
img_pil = Image.open(data_dir)
img_pil_rgb = img_pil.convert("RGB") # convert RGBA 2 RGB
img_tensor = train_transforms(img_pil_rgb)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# download the imagenet category list
# file = open('Labels_imageNet.txt')
# Labels = file.read()
classes = ['Negative', 'Positive']

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %.3f, %s' % (probs[0], classes[idx[0]]))
img = cv2.imread(data_dir)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.2 + img * 0.5
cv2.imwrite(data_dir.strip('.png')+'btCAM'+str(probs[0])+classes[idx[0]]+'.png', result)


