import json
import numpy as np
import torch
from torch import nn
from models.BigGAN import utils
from models.BigGAN import BigGAN
from models.ProgGAN.model import Generator as ProgGenerator
from models.StyleGAN.stylegan_layers import  G_mapping,G_synthesis
from GANs.load import load_model_from_state_dict
from collections import OrderedDict
import math

class ConditionedBigGAN(nn.Module):
    def __init__(self, big_gan, target_classes=(239)):
        super(ConditionedBigGAN, self).__init__()
        self.big_gan = big_gan

        self.target_classes = nn.Parameter(torch.tensor(target_classes, dtype=torch.int64),
            requires_grad=False)

        self.dim_z = self.big_gan.dim_z

    def mixed_classes(self, batch_size):
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size)
        else:
            return torch.from_numpy(np.random.choice(self.target_classes.cpu(), [batch_size]))

    def forward(self, z, classes=None):
        if classes is None:
            classes = self.mixed_classes(z.shape[0]).to(z.device)
        return self.big_gan(z, self.big_gan.shared(classes))


def make_biggan_config(weights_root):
    with open('models/BigGAN/generator_config.json') as f:
        config = json.load(f)
    config['weights_root'] = weights_root
    return config


def make_big_gan(weights_root, target_class):
    config = make_biggan_config(weights_root)

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True

    G = BigGAN.Generator(**config)
    G.load_state_dict(torch.load(config['weights_root'], map_location=torch.device('cpu')),
                      strict=True)

    return ConditionedBigGAN(G, target_class).cuda()


def make_proggan(weights_root):
    model = ProgGenerator()
    model.load_state_dict(torch.load(weights_root))
    model.cuda()

    setattr(model, 'dim_z', [512, 1, 1])
    return model

def make_stylegan(weights_root, resolution):
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=resolution))    
    ]))

    g_all.load_state_dict(torch.load(weights_root, map_location=torch.device('cpu')))
    g_all.cuda()

    # test, set dim z and w , w+
    setattr(g_all, 'dim_z', [512])
    setattr(g_all, 'dim_w', [512])
    times = 2*(math.log(resolution, 2) - 1)
    setattr(g_all, 'dim_w+', [times, 512])

    return g_all


def make_external(gan_dir):
    gan = load_model_from_state_dict(gan_dir)
    G = gan.model.eval()
    setattr(G, 'dim_z', gan.distribution.dim)

    return G
