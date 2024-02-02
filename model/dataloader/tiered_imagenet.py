from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

# Set the appropriate paths of the datasets here.
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
s

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

def load_pickle(data):
    with open(data, 'rb') as f:
        data = pickle.load(f)
    return data['images'], data['labels']

def filter(args, indices):
    train_images, train_labels = load_pickle(args.data_path+'/train_data.pkl')
    test_images, test_labels = load_pickle(args.data_path+'/test_data.pkl')




    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    unique_train_labels = list(set(train_labels))
    unique_test_labels = list(set(test_labels))
    filter_data = []
    filter_labels = []
    for index in indices:
        if index in unique_train_labels:
            data  = train_images[train_labels==index]
            labels = train_labels[train_labels==index]
        elif index in unique_test_labels:
            data = test_images[test_labels==index]
            labels = test_labels[test_labels==index]
        else:
            print("Abnormal")
        if (len(data)==0 or len(labels)==0):
            print("Abnormal")
        filter_data.extend(data)
        filter_labels.extend(labels)
    return np.array(filter_data), np.array(filter_labels)



def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data





class tieredImageNet(data.Dataset):
    def __init__(self, setname, args, augment=False):
        assert(setname=='train' or setname=='val' or setname=='test')

        if setname=='train' or setname=='test':
            if setname=='train':

                openset = np.array(list(set(np.load("class_info/tieredimagenet/train_openset.npy"))))
                closeset = np.array(list(set(np.load("class_info/tieredimagenet/train_closeset.npy"))))
            elif setname=='test':
                openset = np.load("class_info/tieredimagenet/test_openset.npy")
                closeset = np.load("class_info/tieredimagenet/test_closeset.npy")
               
            open_data, open_labels = filter(args, openset)
            close_data, close_labels = filter(args, closeset)
            self.data = np.concatenate([open_data, close_data])
            label = np.concatenate([open_labels, close_labels])
            

        elif setname=='val':
            openset = np.load("class_info/tieredimagenet/val_openset.npy")
            closeset = np.load("class_info/tieredimagenet/val_closeset.npy")
            self.data, label = load_pickle(args.data_path+'/val_data.pkl')
           
       
        self.label = list(label)
        self.num_class = len(set(label))


        self.cluster_info = {}

        for a in list(set(self.label)):
            if a in openset:
                self.cluster_info[a] = 1
            elif a in closeset:
                self.cluster_info[a] = 0
           
      
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomCrop(84, padding=8),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'ResNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])                   
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data)