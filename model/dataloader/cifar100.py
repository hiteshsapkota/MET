from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms




def load_pickle(data):
    with open(data, 'rb') as f:
        data = pickle.load(f)
    return data['images'], data['labels']


def process_data(data):
    all_images = []
    for j in range(0, len(data)):
        image = Image.fromarray(data[j].astype('uint8'), 'RGB')#Image.fromarray(np.uint8(data[j])).convert('RGB')
        image = image.resize((84, 84))
        image = np.asarray(image)
        #image = image.reshape(3, 224, 224)
        all_images.append(image)
    all_images = np.array(all_images)
   
    return all_images


class Cifar100(data.Dataset):
    def __init__(self, setname, args, augment=False):
        assert(setname=='train' or setname=='val' or setname=='test')
        if setname=='train':
            feat_labels = load_data(args.data_path+'/train.pkl')
            openset = np.load("class_info/cifar100/train_openset.npy")
            closeset = np.load("class_info/cifar100/train_closeset.npy")

        elif setname=='val':
            feat_labels = load_data(args.data_path+'/val.pkl')
            openset = np.load("class_info/cifar100/val_openset.npy")
            closeset = np.load("class_info/cifar100/val_closeset.npy")

        elif setname=='test':
            feat_labels = load_data(args.data_path+'/test.pkl')
            openset = np.load("class_info/cifar100/test_openset.npy")
            closeset = np.load("class_info/cifar100/test_closeset.npy")
            
        

        data, label = feat_labels['feat'], feat_labels['labels']
        self.data = process_data(data)
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
                transforms_list)   

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

        elif args.backbone_class=='VGG16':
            self.transform = transforms.Compose(
                transforms_list)   

        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        #transform_img = Image.fromarray(img)#.astype(np.uint8)).convert('RGB')
        #print(transform_img.shape)
        img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data)