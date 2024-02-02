import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os 
import pandas as pd 
np.random.seed(1)

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
IMAGE_PATH1 = osp.join(ROOT_PATH, 'data/miniimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')

def identity(x):
    return x

class MiniImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        im_size = args.orig_imsize
        csv_path = osp.join(SPLIT_PATH, 'altered_'+setname + '.csv')
        cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )
        self.name_class_map = np.load("name_class_map.npy", allow_pickle = True).item()

        self.use_im_cache = ( im_size != -1 ) # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path, setname)
                
                self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label }, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data  = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label, self.img_names = self.parse_csv(csv_path, setname, self.name_class_map)

        self.num_class = len(set(self.label))
    


        

        if setname=='train':
            openset = np.load("class_info/miniimagenet/train_openset.npy")
            closeset = np.load("class_info/miniimagenet/train_closeset.npy")
            
        elif setname == 'val':
            openset = np.load("class_info/miniimagenet/val_openset.npy")
            closeset = np.load("class_info/miniimagenet/val_closeset.npy")
        else:
            openset = np.load("class_info/miniimagenet/test_openset.npy")
            closeset = np.load("class_info/miniimagenet/test_closeset.npy")
        
        self.cluster_info = {}
        for idx in openset:
            self.cluster_info[idx] = 1
        for idx in closeset:
            self.cluster_info[idx] = 0
        

        

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
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

    def parse_csv(self, csv_path, setname, name_class_map):
        data = pd.read_csv(csv_path)
        names, cls_names = list(data['image_name']), list(data['label'])

        data = []
        label = []
        img_names = []
        

        self.wnids = []
        for j in range(len(names)):
            name, wnid = names[j], cls_names[j]
            path = osp.join(IMAGE_PATH1, name)
            class_no = name_class_map[wnid]
            data.append(path)
            label.append(class_no)
            img_names.append(name)
           
       
        return data, label, img_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label, img_name = self.data[i], self.label[i], self.img_names[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))
        
        return image, label, img_name