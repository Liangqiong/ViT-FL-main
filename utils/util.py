from __future__ import absolute_import, division, print_function
import os
import math
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_absolute_error, mean_squared_error


import torch

from utils.scheduler import setup_scheduler


## for optimizaer

from torch import optim as optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin




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
    if not args.num_classes == 1:
        eval_result = simple_accuracy(all_preds, all_label)
    else:
        # eval_result =  mean_absolute_error(all_preds, all_label)
        eval_result =  mean_squared_error(all_preds, all_label)

    model.train()

    return eval_result, eval_losses

def metric_evaluation(args, eval_result):
    if args.num_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False
    return Flag

def valid(args, model, val_loader,  test_loader = None, TestFlag = False):
    # Validation!
    eval_result, eval_losses = inner_valid(args, model, val_loader)

    print("Valid Loss: %2.5f" % eval_losses.avg, "Valid metric: %2.5f" % eval_result)
    if args.dataset == 'CelebA':
        if args.best_eval_loss[args.single_client] > eval_losses.val:
            # if args.best_acc[args.single_client] < eval_result:
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])

            if TestFlag:
                test_result, eval_losses = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print('We also update the test acc of client', args.single_client, 'as',
                      args.current_test_acc[args.single_client])
        else:
            print("Donot replace previous best metric of client", args.best_acc[args.single_client])
    else:  # we use different metrics
        # if args.best_acc[args.single_client] < eval_result:
        if metric_evaluation(args, eval_result):
            if args.save_model_flag:
                save_model(args, model)

            args.best_acc[args.single_client] = eval_result
            args.best_eval_loss[args.single_client] = eval_losses.val
            print("The updated best metric of client", args.single_client, args.best_acc[args.single_client])

            if TestFlag:
                test_result, eval_losses = inner_valid(args, model, test_loader)
                args.current_test_acc[args.single_client] = test_result
                print('We also update the test acc of client', args.single_client, 'as',
                      args.current_test_acc[args.single_client])
        else:
            print("Donot replace previous best metric of client", args.best_acc[args.single_client])

    args.current_acc[args.single_client] = eval_result


def optimization_fun(args, model):

    # Prepare optimizer, scheduler
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)

        print("===============Not implemented optimization type, we used default adamw optimizer ===============")
    return optimizer


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
        optimizer_all[proxy_single_client] = optimization_fun(args, model_all[proxy_single_client])

        # get the total decay steps first
        if not args.dataset == 'CelebA':
            args.t_total[proxy_single_client] = args.clients_with_len[proxy_single_client] *  args.max_communication_rounds / args.batch_size * args.E_epoch
        else:
            # just approximate to make sure average communication round for each client is args.max_communication_rounds
            tmp_rounds = [math.ceil(len/32) for len in args.clients_with_len.values()]
            args.t_total[proxy_single_client]= sum(tmp_rounds)/(args.num_local_clients-1) *  args.max_communication_rounds
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
