# -*- coding: utf-8 -*-

from __future__ import print_function, division
import random
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms, utils
import scipy.io as scio

class SpinalDataset(Dataset):
    """Spinal Landmarks dataset."""
    def __init__(self, csv_file, img_dir, landmark_dir, transform=None,rgb = True):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.landmark_dir = landmark_dir
        print(len(self.landmarks_frame))
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.landmarks_frame.ix[idx, 0])
        landmarks_name = os.path.join(self.landmark_dir, self.landmarks_frame.ix[idx, 0] + '.mat')
        image = cv2.imread(img_name)
        if self.rgb:
            image = image[...,::-1]
        land_data = scio.loadmat(landmarks_name)['p2'].astype(np.float)
        landmarks = np.empty(land_data.shape)
        landmarks[:, 0] = land_data[:, 1] / 1
        landmarks[:, 1] = land_data[:, 0] / 1
        shapes = np.asarray(image.shape[0:2])
        sample = {'image': image, 'landmarks': landmarks, 'shapes': shapes}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        #img = transform.resize(image, (new_h, new_w))
        img = cv2.resize(image,(new_w,new_h))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_h / h, new_w / w]

        return {'image': img, 'landmarks': landmarks, 'shapes': sample['shapes']}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [top, left]

        return {'image': image, 'landmarks': landmarks, 'shapes': sample['shapes']}

class SmartRandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, zoom_scale = 3):
        assert isinstance(zoom_scale, (int, float))
        self.zoom_scale = zoom_scale
    def get_random_rect(self,min_x,min_y,max_x,max_y,w,h):
        rec_w = max_x  - min_x
        rec_h = max_y  - min_y
        scale = (self.zoom_scale-1)/2.0
        b_min_x = min_x - rec_w*scale if min_x - rec_w*scale >0 else 0
        b_min_y = min_y - rec_h*scale if min_y - rec_h*scale >0 else 0
        b_max_x = max_x + rec_w*scale if max_x + rec_w*scale <w else w
        b_max_y = max_y + rec_h*scale if max_y + rec_h*scale <h else h
        #r_min_x = np.random.randint(int(b_min_x),int(min_x)) if b_min_x<min_x else int(min_x)
        #r_min_y = np.random.randint(int(b_min_y),int(min_y)) if b_min_y<min_y else int(min_y)
        #r_max_x = np.random.randint(int(max_x),int(b_max_x)) if b_max_x > max_x else int(max_x)
        #r_max_y = np.random.randint(int(max_y),int(b_max_y)) if b_max_y > max_y else int(max_y)
        return b_min_x,b_min_y,b_max_x,b_max_y

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        min_xy  = np.min(landmarks,axis= 0)
        max_xy  = np.max(landmarks,axis= 0)
        min_x,min_y,max_x,max_y = self.get_random_rect(min_xy[0],min_xy[1],max_xy[0],max_xy[1],w,h)
        image = image[int(min_y): int(max_y),
                int(min_x):int(max_x)]

        landmarks = landmarks - [min_x, min_y]

        return {'image': image, 'landmarks': landmarks, 'shapes': sample['shapes']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        return

    def __call__(self, sample):
        image, land_data = sample['image'], sample['landmarks']

        image = image.transpose((2, 0, 1))
        landmarks = np.empty(land_data.shape)
        landmarks[:, 0] = land_data[:, 0] / image.shape[1]
        landmarks[:, 1] = land_data[:, 1] / image.shape[2]
        land_data =landmarks.reshape(-1)
        return {'image': torch.from_numpy(image).float().div(255),
                'landmarks': torch.from_numpy(land_data).float(),
                'shapes': sample['shapes']}

class RandomFlip(object):
    def __call__(self, sample):
        image = sample['image']
        landmarks = sample['landmarks']
        if random.random()<0.5:
            image = cv2.flip(image,1)
            landmarks[:,1] = image.shape[1]-landmarks[:,1]
        return {'image': image, 'landmarks': landmarks, 'shapes': sample['shapes']}

class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image = sample['image']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        sample['image'] = image
        return sample

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, sample):
        image = sample['image']
        if random.randint(0,2):
            swap = self.perms[random.randint(0,len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
            sample['image'] = image
        return sample

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, sample):
        if random.randint(0,2):
            image = sample['image']
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            sample['image'] = image
        return sample

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        image = sample['image']
        if random.randint(0,2):
            delta = random.uniform(-self.delta, self.delta)
            np.add(image, delta, out=image, casting="unsafe")
            #image += delta
            sample['image'] = image
        return sample

if __name__ == '__main__':
    datapath = '/home/felix/data/AASCE/boostnet_labeldata/'
    transform_train = transforms.Compose([
        Rescale((256,128)),
        RandomFlip(),
        RandomLightingNoise(),
        ToTensor(),
        #Normalize([ 0.225, 0.225, 0.225,], [ 0.153, 0.153, 0.153,]),
    ])
    trainset = SpinalDataset(
        csv_file = datapath + '/labels/training/filenames.csv', transform=transform_train,
        img_dir = datapath + '/data/training/', landmark_dir = datapath + '/labels/training/')

    sam = random.sample(range(480), 10)
    for m in range(10):
        num = sam[m]
        sample = trainset[num]
        image = sample['image'].data.numpy()
        landmark = sample['landmarks'].data.numpy()
        landmark = landmark.reshape(-1, 2)
        img = np.uint8(np.transpose(image, (1,2,0))*255)
        assert np.max(img)<=255
        assert np.min(img)>=0
        for i in range(68):
            img = cv2.circle(img,(int(landmark[i][1]*128),int(landmark[i][0]*256)),1,(255,0,0))
        cv2.imwrite('imshow{a}.png'.format(a=num),img)
        print(m)
