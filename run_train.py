import os
import sys
import argparse
import json
import random
import torch

import matplotlib
matplotlib.use("Agg")

from constants import DEFORMATOR_TYPE_DICT, DEFORMATOR_LOSS_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
from graphs.stylegan.constants import *
from models.gan_load import make_big_gan, make_proggan, make_external, make_stylegan
from latent_deformator import LatentDeformator
from latent_shift_predictor import ResNetShiftPredictor, LeNetShiftPredictor
from trainer import Trainer, Params

from options.train_options import TrainOptions
import graphs
from utils import util
from utils.util import EasyDict


def main():
    tOption = TrainOptions()
    
    for key, val in Params().__dict__.items():
        tOption.parser.add_argument('--{}'.format(key), type=type(val), default=val)

    tOption.parser.add_argument('--args', type=str, default=None, help='json with all arguments')
    tOption.parser.add_argument('--out', type=str, default='./output')
    tOption.parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys(), default='StyleGAN')
    tOption.parser.add_argument('--gan_weights', type=str, default=None)
    tOption.parser.add_argument('--target_class', type=int, default=239)
    tOption.parser.add_argument('--json', type=str)

    tOption.parser.add_argument('--deformator', type=str, default='proj',
                        choices=DEFORMATOR_TYPE_DICT.keys())
    tOption.parser.add_argument('--deformator_random_init', type=bool, default=False)

    tOption.parser.add_argument('--shift_predictor_size', type=int)
    tOption.parser.add_argument('--shift_predictor', type=str,
                        choices=['ResNet', 'LeNet'], default='ResNet')
    tOption.parser.add_argument('--shift_distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

    tOption.parser.add_argument('--seed', type=int, default=2)
    tOption.parser.add_argument('--device', type=int, default=0)
    
    tOption.parser.add_argument('--continue_train', type=bool, default=False)
    tOption.parser.add_argument('--deformator_path', type=str, default='output/models/deformator_90000.pt')
    tOption.parser.add_argument('--shift_predictor_path', type=str, default='output/models/shift_predictor_190000.pt')

    args = tOption.parse()
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.args is not None:
        with open(args.args) as args_json:
            args_dict = json.load(args_json)
            args.__dict__.update(**args_dict)

    # save run params
    #if not os.path.isdir(args.out):
    #    os.makedirs(args.out)
    #with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
    #    json.dump(args.__dict__, args_file)
    #with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
    #    command_file.write(' '.join(sys.argv))
    #    command_file.write('\n')

    # init models
    if args.gan_weights is not None:
        weights_path = args.gan_weights
    else:
        weights_path = WEIGHTS[args.gan_type]

    if args.gan_type == 'BigGAN':
        G = make_big_gan(weights_path, args.target_class).eval()
    elif args.gan_type == 'StyleGAN':
        G = make_stylegan(weights_path, net_info[args.stylegan.dataset]['resolution']).eval()
    elif args.gan_type == 'ProgGAN':
        G = make_proggan(weights_path).eval()
    else:
        G = make_external(weights_path).eval()

    #判断是对z还是w做latent code
    if args.model =='stylegan':
        assert(args.stylegan.latent in ['z', 'w']), 'unknown latent space'
        if args.stylegan.latent == 'z':
            target_dim = G.dim_z
        else:
            target_dim = G.dim_w


    if args.shift_predictor == 'ResNet':
        shift_predictor = ResNetShiftPredictor(args.direction_size, args.shift_predictor_size).cuda()
    elif args.shift_predictor == 'LeNet':
        shift_predictor = LeNetShiftPredictor(args.direction_size, 1 if args.gan_type == 'SN_MNIST' else 3).cuda()
    if args.continue_train:
        deformator = LatentDeformator(direction_size=args.direction_size, out_dim=target_dim, type=DEFORMATOR_TYPE_DICT[args.deformator]).cuda()
        deformator.load_state_dict(torch.load(args.deformator_path, map_location=torch.device('cpu')))

        shift_predictor.load_state_dict(torch.load(args.shift_predictor_path, map_location=torch.device('cpu')))
    else:
        deformator = LatentDeformator(direction_size=args.direction_size, out_dim=target_dim, 
            type=DEFORMATOR_TYPE_DICT[args.deformator], 
            random_init=args.deformator_random_init).cuda()

    # transform
    graph_kwargs = util.set_graph_kwargs(args)

    transform_type = ['zoom','shiftx','color','shifty']
    transform_model = EasyDict()
    for a_type in transform_type:
        model = graphs.find_model_using_name(args.model, a_type)
        g = model(**graph_kwargs)
        transform_model[a_type] = EasyDict(model=g)

    # training
    args.shift_distribution = SHIFT_DISTRIDUTION_DICT[args.shift_distribution_key]
    trainer = Trainer(params=Params(**args.__dict__), out_dir=args.out, out_json=args.json, continue_train=args.continue_train)
    trainer.train(G, deformator, shift_predictor, transform_model)


if __name__ == '__main__':
    main()
