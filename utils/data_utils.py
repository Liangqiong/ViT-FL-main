import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from skimage.transform import resize
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp


import torch
from torchvision import transforms

import torch.utils.data as data

Image.LOAD_TRUNCATED_IMAGES = True

CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


class DatasetFLViT(data.Dataset):
    def __init__(self, args, phase ):
        super(DatasetFLViT, self).__init__()
        self.phase = phase

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:

            self.transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])



        if args.dataset == "cifar10" or args.dataset == 'CelebA':
            data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle=True)
            data_all = data_all.item()


            self.data_all = data_all[args.split_type]

            if self.phase == 'train':
                if args.dataset == 'cifar10':
                    self.data = self.data_all['data'][args.single_client]
                    self.labels = self.data_all['target'][args.single_client]
                else:
                    self.data = self.data_all['train'][args.single_client]['x']
                    self.labels = data_all['labels']
            else:
                if args.dataset == 'cifar10':

                    self.data = data_all['union_' + phase]['data']
                    self.labels = data_all['union_' + phase]['target']

                else:
                    if args.split_type == 'real' and phase == 'val':
                        self.data = self.data_all['val'][args.single_client]['x']
                    elif args.split_type == 'central' or phase == 'test':
                        self.data = list(data_all['central']['val'].keys())

                    self.labels = data_all['labels']

        # for Retina dataset
        elif args.dataset =='Retina' :
            if self.phase == 'test':
                args.single_client = os.path.join(args.data_path, 'test.csv')
            elif self.phase == 'val':
                args.single_client = os.path.join(args.data_path, 'val.csv')

            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
            self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})

            self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                          open(os.path.join(args.data_path, 'labels.csv'))}

            args.loadSize = 256
            args.fineSize_w = 224
            args.fineSize_h = 224
            self.transform = None


        self.args = args


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.args.dataset == 'cifar10':
            img, target = self.data[index], self.labels[index]
            img = Image.fromarray(img)

        elif self.args.dataset == 'CelebA':
            name = self.data[index]
            target = self.labels[name]
            path = os.path.join(self.args.data_path, 'img_align_celeba', name)
            img = Image.open(path).convert('RGB')
            target = np.asarray(target).astype('int64')
        elif self.args.dataset == 'Retina':
            index = index % len(self.img_paths)

            path = os.path.join(self.args.data_path, 'train-all', self.img_paths[index])
            name = self.img_paths[index]

            ## use new way
            img = np.load(path)
            target = self.labels[name]
            target = np.asarray(target).astype('int64')

            if self.phase == 'train':
                if random.random() < 0.5:  # flip
                    img = np.fliplr(img).copy()
                else:  # not flip
                    img = np.array(img)
                img = resize(img, (self.args.loadSize, self.args.loadSize))
                w_offset = random.randint(0, max(0, self.args.loadSize - self.args.fineSize_w - 1))
                h_offset = random.randint(0, max(0, self.args.loadSize - self.args.fineSize_h - 1))
                img = img[w_offset:w_offset + self.args.fineSize_w, h_offset:h_offset + self.args.fineSize_h]

            else:
                img = resize(img, (self.args.loadSize, self.args.loadSize))
                img = np.array(img)
                img = img[(self.args.loadSize - self.args.fineSize_w) // 2:(self.args.loadSize - self.args.fineSize_w) // 2 + self.args.fineSize_w,
                      (self.args.loadSize - self.args.fineSize_h) // 2:(self.args.loadSize - self.args.fineSize_h) // 2 + self.args.fineSize_h]

            img = torch.from_numpy(img).float()  # send to torch
            img = (1 + 1) / 255 * (img - 255) + 1

            if img.dim() < 3:
                img = torch.stack((img, img, img))
            else:
                img = img.permute(2,1,0)


        if self.transform is not None:
            img = self.transform(img)


        return img,  target


    def __len__(self):
        if self.args.dataset == 'Retina' :
            return len(self.img_paths)

        else:
            return len(self.data)


def create_dataset_and_evalmetrix(args):

    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']

    if args.dataset == 'cifar10':

        # get the client with number
        data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle=True)
        data_all = data_all.item()

        data_all = data_all[args.split_type]
        args.dis_cvs_files = [key for key in data_all['data'].keys() if 'train' in key]
        args.clients_with_len = {name: data_all['data'][name].shape[0] for name in args.dis_cvs_files}

    elif args.dataset == 'Retina':
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
        args.clients_with_len = {}

        for single_client in args.dis_cvs_files:
            img_paths = list({line.strip().split(',')[0] for line in
                              open(os.path.join(args.data_path, args.split_type, single_client))})
            args.clients_with_len[single_client] = len(img_paths)


    elif args.dataset == 'CelebA':
        data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle=True)
        # data_all = np.load(os.path.join(args.data_path, args.dataset + '.npy'), allow_pickle = True)
        data_all = data_all.item()
        args.dis_cvs_files = list(data_all[args.split_type]['train'].keys())

        if args.split_type == 'real':
            args.clients_with_len = {name: len(data_all['real']['train'][name]['x']) for name in
                                     data_all['real']['train']}


    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = []
        args.current_test_acc[single_client] = []
        args.best_eval_loss[single_client] = 9999






