# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
from math import ceil

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.scheduler import setup_scheduler
from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.util import valid
from utils.start_config import initization_configure

def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Prepare dataset
    create_dataset_and_evalmetrix(args)

    testset = DatasetFLViT(args, phase = 'test' )
    test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=8)

    # if not CelebA then get the union val dataset,
    if not args.dataset in ['CelebA']:
        valset = DatasetFLViT(args, phase = 'val' )
        val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=8)

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

        print("===============Not implemented optimization type, we used default adamw optimizer ===============")

    tot_step_per_round = [ceil(value / args.batch_size) for value in args.clients_with_len.values()]
    args.t_total = sum(tot_step_per_round) * args.max_communication_rounds * args.E_epoch

    scheduler = setup_scheduler(args, optimizer, t_total=args.t_total)
    if args.num_classes == 1:
        # loss_fct = torch.nn.L1Loss()
        loss_fct = torch.nn.MSELoss()
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    # print('For debugging usage, t_total', args.t_total)

    # Train!
    print("=============== Running training ===============")

    model.zero_grad()
    args.global_step = 0

    for epoch in range(args.max_communication_rounds):

        model.train()
        if args.decay_type == 'step':
            scheduler.step()

        ## iterative each client


        for single_client in args.dis_cvs_files:
            print('Train the client', single_client, 'of communication round', epoch)
            args.single_client = single_client

            trainset = DatasetFLViT(args, phase='train')
            train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, num_workers=args.num_workers)
            if args.dataset == 'CelebA':
                valset = DatasetFLViT(args, phase='val')
                val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=args.num_workers)

            for inner_epoch in range(args.E_epoch):
                for step, batch in enumerate(train_loader):
                    args.global_step += 1
                    batch = tuple(t.to(args.device) for t in batch)
                    x, y = batch
                    if args.num_classes == 1:
                        y = y.float()

                    predict = model(x)
                    loss = loss_fct(predict.view(-1, args.num_classes), y.view(-1))

                    loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if not args.decay_type == 'step':
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()

                    writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=args.global_step)
                    writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=args.global_step)
                    args.learning_rate_record.append(optimizer.param_groups[0]['lr'])

                    if (step +1) % 10 == 0:
                        message = "Client: %s inner epoch: %d step: %d (%d), round: %d (%d) loss: %2.2f lr:  %f" % ( single_client,
                        inner_epoch, step, len(train_loader), epoch,    args.max_communication_rounds,   loss.item()    , optimizer.param_groups[0]['lr']

                        )
                        print(message)

                valid(args, model, val_loader, test_loader, TestFlag=True)
                model.train()

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)
        args.record_val_acc = args.record_val_acc.append(args.current_acc, ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))
        args.record_test_acc = args.record_test_acc.append(args.current_test_acc, ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

    writer.close()
    print("================End training! ================ ")





def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="Swin-tiny",  help="Basic Name of this run with detailed network-architecture selection")
    parser.add_argument("--FL_platform", type = str, default="Swin-CWT", choices=["Swin-CWT", 'ViT-CWT',"ResNet-CWT", "Swin-CWT", "EfficientNet-CWT"],  help="Choose of different FL platform.")
    parser.add_argument("--dataset", choices=["cifar10", "Retina","CelebA"], default="cifar10", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")

    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")
    parser.add_argument("--cfg",  type=str, default="configs/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to args file for Swin-FL',)

    parser.add_argument('--Pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/swin_tiny_patch4_window7_224.pth", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=32, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default= 3e-3, type=float, help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")

    ## FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int, help="Total communication rounds.")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real" ,"central"], default="split_2", help="Which data partitions to use")


    args = parser.parse_args()

    # Initialization

    model = initization_configure(args)

    # Training, Validating, and Testing
    train(args, model)

    # Show final performance

    message = '\n \n ==============Start showing final performance ================= \n'
    message += 'Final union test accuracy is: %2.5f with std: %2.5f \n' %  \
                   (np.asarray(list(args.current_test_acc.values())).mean(),  np.asarray(list(args.current_test_acc.values())).std())
    message += "================ End ================ \n"

    with open(args.file_name, 'a+') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)

if __name__ == "__main__":
    main()
