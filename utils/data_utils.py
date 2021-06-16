import os
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
import torch.utils.data as data



class DatasetFLViT(data.Dataset):
    def __init__(self, args, phase ):
        super(DatasetFLViT, self).__init__()
        self.phase = phase


        if args.dataset == "cifar10" or args.dataset == 'CelebA':
            # data_all = np.load(os.path.join(args.data_path, args.dataset + '.npy'), allow_pickle = True)
            data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle = True)
            data_all = data_all.item()

            self.data_all = data_all[args.split_type]

            if self.phase == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])


                if args.dataset == 'cifar10':
                    self.data = self.data_all['data'][args.single_client]
                    self.labels = self.data_all['target'][args.single_client]
                else:
                    self.data = self.data_all['train'][args.single_client]['x']
                    self.labels = data_all['labels']
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
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
        elif args.dataset =='Retina':
            args.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                          open(os.path.join(args.data_path, 'labels.csv'))}

            args.loadSize = 256
            args.fineSize_w = 224
            args.fineSize_h = 224

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

        if self.transform is not None:
            img = self.transform(img)


        return img,  target


    def __len__(self):
        return len(self.data)



def create_dataset_and_evalmetrix(args):

    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = ['central']
    else:

        if args.dataset == 'cifar10':

            # get the client with number
            data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle = True)
            data_all = data_all.item()

            data_all = data_all[args.split_type]
            args.dis_cvs_files =[key for key in data_all['data'].keys() if 'train' in key]
            args.clients_with_len = {name: data_all['data'][name].shape[0] for name in args.dis_cvs_files}

        elif args.dataset == 'Retina':
            args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
        elif args.dataset == 'CelebA':
            data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle = True)
            # data_all = np.load(os.path.join(args.data_path, args.dataset + '.npy'), allow_pickle = True)
            data_all = data_all.item()
            args.dis_cvs_files = list(data_all[args.split_type]['train'].keys())
            if args.split_type == 'real':
                args.clients_with_len = {name: len(data_all['real']['train'][name]['x']) for name in  data_all['real']['train']}

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






