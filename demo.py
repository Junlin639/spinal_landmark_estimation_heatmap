from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
from models import *

parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('-img', '--image', default='spine', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-c', '--checkpoint', default='checkpoint/00/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()

def load_model():
    model = resnet(136)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    out_size = 256
    model = load_model()
    model = model.eval()
    import glob,random

    path1 = '/home/felix/data/AASCE/boostnet_labeldata/data/training/*.jpg'
    path2 = '/home/felix/data/AASCE/boostnet_labeldata/data/test/*.jpg'
    path3 = '/home/felix/data/AASCE/boostnet_labeldata/data/test2/*.jpg'
    sam = random.sample(glob.glob(path3),5)
    for filename in sam:
        name = os.path.basename(filename)
        img = cv2.imread(filename)
        img = cv2.resize(img,(128,256))
        raw_img = img
        img = img/255.0
        img = (img-np.mean(img))/np.std(img)
        img = img.transpose((2, 0, 1))
        img = img.reshape((1,) + img.shape)
        input = torch.from_numpy(img).float()
        input= torch.autograd.Variable(input)
        out = model(input).cpu().data.numpy()
        out = out.reshape(-1,2)
        raw_img = cv2.resize(raw_img,(128,256))
        for i in range(68):
            cv2.circle(raw_img,(int(out[i][1]*128),int(out[i][0]*256)),2,(255,0,0),-1)
        cv2.imwrite('result_{a}.png'.format(a=name),raw_img)
        print(name)
    print('done!')

