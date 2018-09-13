# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
import os
import cv2
import argparse
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from model_skip import Model
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn as nn
to_pil = ToPILImage()
to_tensor = ToTensor()

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--net', type=str, default='vgg16')
parser.add_argument('--data_dir', default='MURA-v1.1', metavar='DIR', help='path to dataset')
parser.add_argument('--resume', default='8vgg16ICNN_MURA_skip.pth.tar', type=str)
parser.add_argument('--num_classes', type=int, default=2, help="how types")
parser.add_argument('--norm_template', type=int, default=1, help='Norm of the template')
parser.add_argument('--pai', type=bool, default=False)
global args
args = parser.parse_args()
# networks such as googlenet, resnet, densenet already use global average pooling at the end,
# so CAM could be used directly.
net = Model(args)
if args.resume:
    file_path = os.path.join('/Users/suqi.lmh/weaksup/models_train/New/FCN/MURA', args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(file_path, map_location='cpu')
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_checkpoint[name] = v
    net.load_state_dict(new_checkpoint)

net.eval()

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())


def returnCAM(feature_conv, index):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    feature_conv = feature_conv.data.numpy()
    cam = feature_conv.squeeze()[index]
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])

file_name = 'image1.png'
data_dir = os.path.join('/Users/suqi.lmh/weaksup/test/MURA/Shoulder/patient00064/study1_positive', file_name)
img_pil = Image.open(data_dir).convert("RGB")
img_size = img_pil.size
img_tensor = train_transforms(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
out = net(img_variable)
logit = out[0]
features_blobs = out[1]

classes = ['Negative', 'Positive']

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()
print('output ICNN.jpg for the top1 prediction: %.3f, %s' % (probs[0], classes[idx[0]]))

# generate class activation mapping for the top1 prediction
CAMs_yes = returnCAM(features_blobs, idx[0])
CAMs_no = returnCAM(features_blobs, idx[1])

# render the CAM and output
img = cv2.imread(data_dir)
height, width, _ = img.shape
heatmap_yes = cv2.applyColorMap(cv2.resize(CAMs_yes, (width, height)), cv2.COLORMAP_JET)
heatmap_no = cv2.applyColorMap(cv2.resize(CAMs_no, (width, height)), cv2.COLORMAP_JET)

result_yes = heatmap_yes * 0.2 + img * 0.5
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[0])+classes[idx[0]]+'.jpg', result_yes)

# peak response maps
# grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
# class_response_maps.backward(grad_output, retain_graph=True)   # ???
# prm = input.grad.detach().sum(1).clone().clamp(min=0)
# peak_response_maps.append(prm / prm.sum())


# path = '/Users/suqi.lmh/weaksup/test/MURA/Shoulder/patient00064/study1_positive/image1_c'
# for j in range(1):
#     features = out[5+j].squeeze()
#     for i in range(features.size(0)):
#         CAMs = returnCAM(features[i], idx[0])
#         heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
#         result = heatmap * 0.2 + img * 0.5
#         cv2.imwrite(path + str(j) + '/' + str(i) + '.jpg', result)

# result_no = heatmap_no * 0.2 + img * 0.5
# cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[1])+classes[idx[1]]+'.jpg', result_no)


