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


def returnCAM(feature_conv, index):
    size_upsample = (224, 224)
    feature_conv = feature_conv.data.numpy()
    cam = feature_conv.squeeze()[index]
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam


# peak response maps
def peak_response_map(img_variable, class_response_maps, fea_size):
    peak_idx = F.max_pool2d(class_response_maps, kernel_size=fea_size, return_indices=True)[1].squeeze()
    grad_output = torch.zeros(1, 2, fea_size, fea_size)
    grad_output[0, 0, int(peak_idx[0]/fea_size), peak_idx[0] % fea_size] = 1
    grad_output[0, 0, int(peak_idx[1]/fea_size), peak_idx[1] % fea_size] = 1
    class_response_maps.backward(grad_output, retain_graph=True)   # ???
    prm_get = img_variable.grad.detach().sum(1).clone()  # .clamp(min=mean())
    prm = returnCAM(prm_get, 0)  # / prm.sum()
    # prm = cv2.applyColorMap(prm, cv2.COLORMAP_JET)
    return prm


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-net', type=str, default='vgg16')
parser.add_argument('--data_dir', default='MURA-v1.1', metavar='DIR', help='path to dataset')
parser.add_argument('--resume', default='13vgg16ICNN_MURA_fpn.pth.tar', type=str)
parser.add_argument('-num_classes', type=int, default=2, help="how types")
parser.add_argument('--norm_template', type=int, default=1, help='Norm of the template')
parser.add_argument('-pai', type=bool, default=False)
global args
args = parser.parse_args()
# networks such as googlenet, resnet, densenet already use global average pooling at the end,
# so CAM could be used directly.
net = Model(args)
if args.resume:
    file_path = os.path.join('/Users/suqi.lmh/weaksup/models_train/New/FPN/ICNN_MURA', args.resume)
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

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])
# file_name = 'neg-1-1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.jpg'
# file_name = 'pos-1000-1.3.6.1.4.1.14519.5.2.1.6279.6001.134519406153127654901640638633.jpg'
# file_name = 'neg-1000-1.3.6.1.4.1.14519.5.2.1.6279.6001.134519406153127654901640638633.jpg'
# file_name = 'pos-1-1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.jpg'
# file_name = 'pos-600-1.3.6.1.4.1.14519.5.2.1.6279.6001.121993590721161347818774929286.jpg'
# file_name = 'pos-200-1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306.jpg'
file_name = 'image1.png'
# file_name = 'pos-9300-1.3.6.1.4.1.14519.5.2.1.6279.6001.970428941353693253759289796610.jpg'
data_dir = os.path.join('/Users/suqi.lmh/weaksup/test/MURA/Shoulder/patient00064/study1_positive', file_name)
img_pil = Image.open(data_dir).convert("RGB")
img_size = img_pil.size
img_tensor = train_transforms(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0), requires_grad=True)
out = net(img_variable)
logit = out[0]

# download the imagenet category list
classes = ['Negative', 'Positive']

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()
print('output ICNN.jpg for the top1 prediction: %.3f, %s' % (probs[0], classes[idx[0]]))

# generate class activation mapping for the top1 prediction
CAMs_yes1 = returnCAM(out[1], idx[0])
# CAMs_no = returnCAM(features_blobs, idx[1])
CAMs_yes2 = returnCAM(out[2], idx[0])
CAMs_yes3 = returnCAM(out[3], idx[0])
CAMs_yes4 = returnCAM(out[4], idx[0])

# render the CAM and output
img = cv2.imread(data_dir)
height, width, _ = img.shape
heatmap_yes1 = cv2.applyColorMap(cv2.resize(CAMs_yes1, (width, height)), cv2.COLORMAP_JET)
heatmap_yes2 = cv2.applyColorMap(cv2.resize(CAMs_yes2, (width, height)), cv2.COLORMAP_JET)
heatmap_yes3 = cv2.applyColorMap(cv2.resize(CAMs_yes3, (width, height)), cv2.COLORMAP_JET)
heatmap_yes4 = cv2.applyColorMap(cv2.resize(CAMs_yes4, (width, height)), cv2.COLORMAP_JET)
# heatmap_no = cv2.applyColorMap(cv2.resize(CAMs_no, (width, height)), cv2.COLORMAP_JET)

result_yes1 = heatmap_yes1 * 0.2 + img * 0.5
result_yes2 = heatmap_yes2 * 0.2 + img * 0.5
result_yes3 = heatmap_yes3 * 0.2 + img * 0.5
result_yes4 = heatmap_yes4 * 0.2 + img * 0.5

cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[0])+classes[idx[0]]+'1.jpg', result_yes1)
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[0])+classes[idx[0]]+'2.jpg', result_yes2)
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[0])+classes[idx[0]]+'3.jpg', result_yes3)
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[0])+classes[idx[0]]+'4.jpg', result_yes4)
print(logit)

prm4 = peak_response_map(img_variable, out[4], 14)
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + classes[idx[0]]+'PRM4.jpg', cv2.resize(prm4, (512, 512)))

prm3 = peak_response_map(img_variable, out[3], 28)
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + classes[idx[0]]+'PRM3.jpg', cv2.resize(prm3, (512, 512)))

path = '/Users/suqi.lmh/weaksup/test/MURA/Shoulder/patient00064/study1_positive/image1_p'
# for j in range(4):
#     features = out[5+j].squeeze()
#     for i in range(features.size(0)):
#         CAMs = returnCAM(features[i], idx[0])
#         heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
#         result = heatmap * 0.2 + img * 0.5
#         cv2.imwrite(path + str(j+1) + '/' + str(i) + '.jpg', result)
# result_no = heatmap_no * 0.2 + img * 0.5
# cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[1])+classes[idx[1]]+'.jpg', result_no)


