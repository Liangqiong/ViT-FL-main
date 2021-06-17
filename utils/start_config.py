import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as torch_models

from models.modeling import VisionTransformer, CONFIGS


def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.FL_platform, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'

    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '


    ## save to disk of current log

    args.file_name = os.path.join(args.output_dir, 'log_file.txt')

    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')

    print(message)


def initization_configure(args):
    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")


    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.device == 'cpu':
        torch.cuda.manual_seed(args.seed)

    # Set model type related parameters
    if args.FL_platform == "ResNet-CWT" or args.FL_platform == 'ResNet-FedAVG':
        args.Use_ResNet = True
    else:
        args.Use_ResNet = False

    if args.dataset == 'cifar10':
        args.num_classes = 10
    else:
        args.num_classes = 2


    # Prepare model
    config = CONFIGS[args.model_type]


    if not args.Use_ResNet:
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
        model.load_from(np.load(args.pretrained_dir))
        model.to(args.device)

    else:

        model = torch_models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, args.num_classes)
        model.to(args.device)



    # set output parameters
    args.name = args.net_name + '_' + args.split_type + '_lr_' + args.decay_type + '_' + str(args.learning_rate) + \
                '_WUP_'  + str(args.warmup_steps) + '_Round_' + str(args.max_communication_rounds) + '_Eepochs_' + str(args.E_epoch) + '_Seed_' + str(args.seed)
    args.output_dir = os.path.join('output', args.FL_platform, args.dataset, args.name)
    os.makedirs(args.output_dir, exist_ok=True)

    print_options(args, model)



    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}

    return model


