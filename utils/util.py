from __future__ import absolute_import, division, print_function
import os
import numpy as np
from copy import deepcopy

import torch

from utils.scheduler import setup_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(args.output_dir, "%s_%s_checkpoint.bin" % (args.name, client_name))

    torch.save(model_to_save.state_dict(), model_checkpoint)
    # print("Saved model checkpoint to [DIR: %s]", args.output_dir)



def inner_valid(args, model, test_loader):
    eval_losses = AverageMeter()

    print("++++++ Running Validation of client", args.single_client, "++++++")
    model.eval()
    all_preds, all_label = [], []

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if not args.Use_ResNet:
                logits = model(x)[0]
            else:
                logits = model(x)

            if args.num_classes > 1:
                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())

            if args.num_classes > 1:
                preds = torch.argmax(logits, dim=-1)
            else:

                preds = logits

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
    all_preds, all_label = all_preds[0], all_label[0]
    eval_result = simple_accuracy(all_preds, all_label)
    model.train()

    return eval_result, eval_losses

def valid(args, model, val_loader,  test_loader = None, TestFlag = False):
    # Validation!
    eval_result, eval_losses = inner_valid(args, model, val_loader)

    print("Valid Loss: %2.5f" % eval_losses.avg, "Valid Accuracy: %2.5f" % eval_result)
    if args.dataset == 'CelebA':
        if args.best_eval_loss[args.single_client] > eval_losses.val:
            # if args.best_acc[args.single_client] < eval_result:
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print("The updated best acc of client", args.single_client, args.best_acc[args.single_client])

            if TestFlag:
                test_result, eval_losses = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print('We also update the test acc of client', args.single_client, 'as',
                      args.current_test_acc[args.single_client])
        else:
            print("Donot replace previous best acc of client", args.best_acc[args.single_client])
    else: # we use different metrics
        if args.best_acc[args.single_client] < eval_result:
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print("The updated best acc of client", args.single_client, args.best_acc[args.single_client])

            if TestFlag:
                test_result, eval_losses = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print('We also update the test acc of client', args.single_client, 'as',
                      args.current_test_acc[args.single_client])
        else:
            print("Donot replace previous best acc of client", args.best_acc[args.single_client])

    args.current_acc[args.single_client] = eval_result



def Partial_Client_Selection(args, model):

    # Select partial clients join in FL train
    if args.num_local_clients == -1: # all the clients joined in the train
        args.proxy_clients = args.dis_cvs_files
        args.num_local_clients =  len(args.dis_cvs_files)# update the true number of clients
    else:
        args.proxy_clients = ['train_' + str(i) for i in range(args.num_local_clients)]

    # Generate model for each client
    model_all = {}
    optimizer_all = {}
    scheduler_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    for proxy_single_client in args.proxy_clients:
        model_all[proxy_single_client] = deepcopy(model).cpu()
        optimizer_all[proxy_single_client] = torch.optim.SGD(model_all[proxy_single_client].parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

        # get the total decay steps first
        if not args.dataset == 'CelebA':
            args.t_total[proxy_single_client] = args.clients_with_len[proxy_single_client] *  args.num_steps / args.batch_size * args.E_epoch
        else:
            # just approximate to make sure average communication round for each client is args.num_steps
            args.t_total[proxy_single_client] = sum(args.clients_with_len.values()) / (args.num_local_clients-1) *  \
                                                args.num_steps / args.batch_size * args.E_epoch
        scheduler_all[proxy_single_client] = setup_scheduler(args, optimizer_all[proxy_single_client], t_total=args.t_total[proxy_single_client])
        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    return model_all, optimizer_all, scheduler_all




def average_model(args,  model_avg, model_all):
    model_avg.cpu()
    print('Calculate the model avg----')
    params = dict(model_avg.named_parameters())

    # for name, value in model_state_dict.items():
    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(np.array(single_client_weight)).float()

            if client == 0:
                tmp_param_data = dict(model_all[single_client].named_parameters())[
                                     name].data * single_client_weight
            else:
                tmp_param_data = tmp_param_data + \
                                 dict(model_all[single_client].named_parameters())[
                                     name].data * single_client_weight
        params[name].data.copy_(tmp_param_data)

    print('Update each client model parameters----')

    for single_client in args.proxy_clients:
        tmp_params = dict(model_all[single_client].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)
