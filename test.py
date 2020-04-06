import os
import sys
import argparse
import json
import random
import torch

import matplotlib
matplotlib.use("Agg")

from constants import DEFORMATOR_TYPE_DICT, DEFORMATOR_LOSS_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
from models.gan_load import make_stylegan
from latent_deformator import LatentDeformator
from latent_shift_predictor import ResNetShiftPredictor
from visualization import make_interpolation_chart, fig_to_image
from utils import make_noise


def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    
    parser.add_argument('--out', type=str, default='./output')
    parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys(), default='StyleGAN')
    parser.add_argument('--gan_weights', type=str, default=None)
    parser.add_argument('--json', type=str)

    parser.add_argument('--deformator', type=str, default='ortho',
                        choices=DEFORMATOR_TYPE_DICT.keys())
    parser.add_argument('--deformator_path', type=str, default='output/models/deformator_490000.pt')
    parser.add_argument('--images_dir', type=str, default='output/images/')

    parser.add_argument('--shift_predictor_size', type=int)
    parser.add_argument('--shift_predictor', type=str,
                        choices=['ResNet', 'LeNet'], default='ResNet')
    parser.add_argument('--shift_distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    
    # save run params
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    
    # init models
    if args.gan_weights is not None:
        weights_path = args.gan_weights
    else:
        weights_path = WEIGHTS[args.gan_type]

    if args.gan_type == 'BigGAN':
        G = make_big_gan(weights_path, args.target_class).eval()
    elif args.gan_type == 'StyleGAN':
        G = make_stylegan(weights_path)
    elif args.gan_type == 'ProgGAN':
        G = make_proggan(weights_path).eval()
    else:
        G = make_external(weights_path).eval()

    deformator = LatentDeformator(G.dim_z,type=DEFORMATOR_TYPE_DICT[args.deformator]).cuda()
    deformator.load_state_dict(torch.load(args.deformator_path, map_location=torch.device('cpu')))

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    z = make_noise(batch = 5,dim = G.dim_z).cuda()
    dims=[2, 9]

    fig = make_interpolation_chart(G, deformator=deformator, z = z,
                             shifts_r=10, shifts_count=3,
                             dims=dims, dims_count=10, texts=None, dpi=1024)
    fig_to_image(fig).convert("RGB").save(os.path.join(args.images_dir, 'batch_{}.jpg'.format(args.seed)))

if __name__ == '__main__':
    main()