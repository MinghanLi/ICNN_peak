# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
import os
import cv2
import argparse
import torch
import maxflow
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from model_test import Model
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from sklearn.cluster import KMeans
from PIL import Image
to_pil = ToPILImage()
to_tensor = ToTensor()


def returnCAM(feature_conv):
    # generate the class activation maps upsample to 256x256
    cam = feature_conv
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cam_img


def get_bbox(heatmap):
    heatmap = torch.Tensor(heatmap).view(1, 512, 512)
    thh = F.max_pool2d(heatmap, 512).squeeze()*0.7
    ind_mat = torch.ge(heatmap.squeeze(), thh)
    ind = []
    for i in range(512**2):
        coordx = int(i // 512)
        coordy = int(i % 512)
        if ind_mat[coordy][coordx] == 1:
            ind.append([coordy, coordx])
    ind = torch.Tensor(ind).t()
    minx, maxx = min(ind[1]), max(ind[1])
    miny, maxy = min(ind[0]), max(ind[0])
    return int(minx), int(miny), int(maxx), int(maxy), ind_mat


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('-net', type=str, default='vgg16')
parser.add_argument('--data_dir', default='MURA-v1.1', metavar='DIR', help='path to dataset')
parser.add_argument('--resume', default='peak_8vgg16ICNN_Luna16_circle.pth.tar', type=str)  #peak_8vgg16ICNN_Luna16_circle.pth.tar
parser.add_argument('-num_classes', type=int, default=2, help="how types")
parser.add_argument('--norm_template', type=int, default=1, help='Norm of the template')
parser.add_argument('-pai', type=bool, default=False)
global args
args = parser.parse_args()

# load pre-trained model
net = Model(args)
if args.resume:
    path = '/Users/suqi.lmh/weaksup/models_train/New/FCN/ICNN_peak/8_4'
    # path = '/Users/suqi.lmh/weaksup/models_train/New/FCN/ICNN_peak/skip_16_4'
    file_path = os.path.join(path, args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(file_path, map_location='cpu')
    from collections import OrderedDict
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_checkpoint[name] = v
    net.load_state_dict(new_checkpoint)

net.eval()

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
    ])
file_name = 'pos-8999-1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.jpg'
data_dir = os.path.join('/Users/suqi.lmh/weaksup/test/Luna16', file_name)
img_pil = Image.open(data_dir).convert("RGB")
img_size = img_pil.size
img_tensor = train_transforms(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0), requires_grad=True)
out = net(img_variable)
logit = out[0]
heatmap = out[1].data.numpy().squeeze()

# download the imagenet category list
# file = open('Labels_imageNet.txt')
# Labels = file.read()
classes = ['Negative', 'Positive']
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()
print('output ICNN.jpg for the top1 prediction: %.3f, %s' % (probs[0], classes[idx[0]]))

# generate class activation mapping for the top1 prediction
CAMs_yes = returnCAM(heatmap[idx[0]])
CAMs_no = returnCAM(heatmap[idx[1]])
img = cv2.imread(data_dir)
height, width, _ = img.shape
heatmap_yes = cv2.applyColorMap(CAMs_yes, cv2.COLORMAP_JET)
heatmap_no = cv2.applyColorMap(CAMs_no, cv2.COLORMAP_JET)

result_yes = heatmap_yes * 0.2 + img * 0.5
cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[0])+classes[idx[0]]+'.jpg', result_yes)
# result_no = heatmap_no * 0.2 + img * 0.5
# cv2.imwrite(data_dir.strip('.png')+'ICNN'+args.net + str(probs[1])+classes[idx[1]]+'.jpg', result_no)

# *** peak response maps ***
scale = 16
class_response_maps = out[2]
peak_idx = F.max_pool2d(class_response_maps, kernel_size=scale, return_indices=True)[1].squeeze()
grad_output = torch.zeros(1, 2, scale, scale)
grad_output[0, 0, int(peak_idx[0]/scale), peak_idx[0] % scale] = 1
grad_output[0, 0, int(peak_idx[1]/scale), peak_idx[1] % scale] = 1
class_response_maps.backward(grad_output, retain_graph=True)
prm = img_variable.grad.detach().sum(1).clone()      # .clamp(min=mean())
peak_response_maps = returnCAM(prm.squeeze().data.numpy())  # / prm.sum()
thr = int(torch.Tensor(peak_response_maps.reshape(1, -1)).mode()[0])
peak_response_maps = (peak_response_maps.clip(min=thr) - thr) * 5
prm = cv2.resize(peak_response_maps, (512, 512))
prm_rgb = cv2.applyColorMap(prm, cv2.COLORMAP_JET)
result_prm = prm_rgb * 0.2 + img * 0.5
cv2.imwrite(data_dir.strip('.png')+'ICNN'+'PRM.jpg', result_prm)

# *** segmentation ***
minx, miny, maxx, maxy, labelmap_bb = get_bbox(prm)
region = np.asarray(img_pil.crop((minx, miny, maxx, maxy)).convert('L'))
# prm_region = prm[minx:maxx, miny:maxy]
# edgs = cv2.Canny(region, 50, 100)
g = maxflow.Graph[int]()
nodeids = g.add_grid_nodes(region.shape)
g.add_grid_edges(nodeids, 20)
g.add_grid_tedges(nodeids, region, region.max()-region)
g.maxflow()
sgm = g.get_grid_segments(nodeids)
label_sgm = np.int_(np.logical_not(sgm))
labelmap_bb[miny:maxy, minx:maxx] = torch.Tensor(label_sgm*255)
img_mask = cv2.applyColorMap(labelmap_bb.numpy(), cv2.COLORMAP_JET)
result_sgm = img_mask * 0.3 + img * 0.5
cv2.imwrite(data_dir.strip('.png')+'ICNN_seg'+'.jpg', result_sgm)

# *** embedding vector + kmeans ***
embed_vec = F.batch
embed_vec = embed_vec.squeeze().permute(1, 2, 0).view(-1, 1024).data.numpy()
cluster = KMeans(n_clusters=2).fit(embed_vec)
cluster_map = cluster.labels_.view(512, 512)
labelmap_bb[miny:maxy, minx:maxx] = torch.Tensor(cluster_map*255)
img_mask = cv2.applyColorMap(labelmap_bb.numpy(), cv2.COLORMAP_JET)
result_sgm = img_mask * 0.3 + img * 0.5
cv2.imwrite(data_dir.strip('.png')+'ICNN_kmeans'+'.jpg', result_sgm)

# ***  bbox ***
img_bb = cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
cv2.imwrite(data_dir.strip('.png')+'ICNN_bb'+'.jpg', img)

# ### plot feature ###
# path = '/Users/suqi.lmh/weaksup/test/Luna16/pos_8999_c'
# for j in range(4):
#     features = out[3+j].squeeze()
#     for i in range(features.size(0)):
#         CAMs = returnCAM(features[i].data.numpy())
#         # heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
#         # result = heatmap * 0.2 + img * 0.5
#         cv2.imwrite(path + str(j+1) + '/' + str(i) + '.jpg', CAMs)
