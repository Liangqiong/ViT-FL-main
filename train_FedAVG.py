# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.data_utils import DatasetFLViT, create_dataset_and_evalmetrix
from utils.util import Partial_Client_Selection, valid, average_model
from utils.start_config import initization_configure

def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))


    # Prepare dataset
    create_dataset_and_evalmetrix(args)

    testset = DatasetFLViT(args, phase = 'test' )
    # test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.batch_size, num_workers=8)

    # if not CelebA then get the union val dataset,
    if not args.dataset == 'CelebA':
        valset = DatasetFLViT(args, phase = 'val' )
        # val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size, num_workers=8)


    # Configuration for FedAVG, prepare model, optimizer, scheduler
    model_all, optimizer_all, scheduler_all = Partial_Client_Selection(args, model)
    model_avg = deepcopy(model).cpu()

    # Train!
    print("=============== Running training ===============")
    loss_fct = torch.nn.CrossEntropyLoss()
    tot_clients = args.dis_cvs_files
    epoch = -1
    # For debug
    # print(args.t_total)

    while True:
        epoch += 1
        # randomly select partial clients
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            cur_selected_clients = args.proxy_clients
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]

        val_loader_proxy_clients = {}

        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens

            trainset = DatasetFLViT(args, phase='train')
            train_loader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.batch_size, num_workers=args.num_workers)
            if args.dataset == 'CelebA':
                valset = DatasetFLViT(args, phase='val')
                val_loader_proxy_clients[proxy_single_client] = DataLoader(valset, sampler=SequentialSampler(valset), batch_size=args.batch_size,
                                          num_workers=args.num_workers)
            else:
                # for Retina and Cifar10 datasets we use union validation dataset
                val_loader_proxy_clients[proxy_single_client] = val_loader


            model = model_all[proxy_single_client]
            model = model.to(args.device).train()
            optimizer = optimizer_all[proxy_single_client]
            scheduler = scheduler_all[proxy_single_client]
            if args.decay_type == 'step':
                scheduler.step()

            print('Train the client', cur_single_client, 'of communication round', epoch)

            for inner_epoch in range(args.E_epoch):
                for step, batch in enumerate(train_loader):  # batch = tuple(t.to(args.device) for t in batch)
                    args.global_step_per_client[proxy_single_client] += 1
                    batch = tuple(t.to(args.device) for t in batch)

                    x, y = batch
                    predict = model(x)
                    loss = loss_fct(predict.view(-1, args.num_classes), y.view(-1))

                    loss.backward()

                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if not args.decay_type == 'step':
                        scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()

                    writer.add_scalar(proxy_single_client + '/lr', scalar_value=optimizer.param_groups[0]['lr'],
                                      global_step=args.global_step_per_client[proxy_single_client])
                    writer.add_scalar(proxy_single_client + '/loss', scalar_value=loss.item(),
                                      global_step=args.global_step_per_client[proxy_single_client])


                    args.learning_rate_record[proxy_single_client].append(optimizer.param_groups[0]['lr'])

                    if (step+1 ) % 10 == 0:
                        print(cur_single_client, step,':', len(train_loader),'inner epoch', inner_epoch, 'round', epoch,':',
                              args.max_communication_rounds, 'loss', loss.item(), 'lr', optimizer.param_groups[0]['lr'])

            # we use frequent transfer of model between GPU and CPU due to limitation of GPU memory
            model.to('cpu')

        ## ---- model average and eval

        # average model
        average_model(args, model_avg, model_all)
        # then evaluate
        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            args.single_client = cur_single_client
            model = model_all[proxy_single_client]
            model.to(args.device)
            valid(args, model, val_loader_proxy_clients[proxy_single_client], test_loader, TestFlag=True)
            model.cpu()

        args.record_val_acc = args.record_val_acc.append(args.current_acc, ignore_index=True)
        args.record_val_acc.to_csv(os.path.join(args.output_dir, 'val_acc.csv'))
        args.record_test_acc = args.record_test_acc.append(args.current_test_acc, ignore_index=True)
        args.record_test_acc.to_csv(os.path.join(args.output_dir, 'test_acc.csv'))

        np.save(args.output_dir + '/learning_rate.npy', args.learning_rate_record)

        tmp_round_acc = [val for val in args.current_test_acc.values() if not val == []]
        writer.add_scalar("test/average_accuracy", scalar_value=np.asarray(tmp_round_acc).mean(), global_step=epoch)

        if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:
            break

    writer.close()
    print("================End training! ================ ")


def main():
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="ViT-small",  help="Basic Name of this run with detailed network-architecture selection. ")
    parser.add_argument("--FL_platform", type = str, default="ViT-FedAVG", choices=[ "Swin-FedAVG", "ViT-FedAVG", "Swin-FedAVG", "EfficientNet-FedAVG", "ResNet-FedAVG"],  help="Choose of different FL platform. ")
    parser.add_argument("--dataset", choices=["cifar10", "Retina"], default="cifar10", help="Which dataset.")
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
    parser.add_argument("--gpu_ids", type=str, default='2', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization") #99999

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # parser.add_argument("--learning_rate", default=3e-2, type=float, choices=[5e-4, 3e-2, 1e-3],  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # 1e-5 for ViT central

    ## FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=-1, choices=[10, -1], type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real", "central"], default="split_3", help="Which data partitions to use")


    args = parser.parse_args()

    # Initialization

    model = initization_configure(args)

    # Training, Validating, and Testing
    train(args, model)


    message = '\n \n ==============Start showing final performance ================= \n'
    message += 'Final union test accuracy is: %2.5f  \n' %  \
                   (np.asarray(list(args.current_test_acc.values())).mean())
    message += "================ End ================ \n"


    with open(args.file_name, 'a+') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)



if __name__ == "__main__":
    main()
