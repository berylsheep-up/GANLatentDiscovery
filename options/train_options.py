# note: currently doesn't handle recursively nested groups and conflicting
# option strings

import argparse
import oyaml as yaml
import sys
import os
from collections import OrderedDict

import sys
sys.path.append('resources/stylegan')
import copy
import dnnlib
from dnnlib import EasyDict
import config
from metrics import metric_base

class TrainOptions():
    def __init__(self):
        self.initialized = False
        self.parser = parser = argparse.ArgumentParser("Training Parser")

    def initialize(self):
        parser = self.parser
        parser.add_argument('--config_file', type=argparse.FileType(mode='r'), help="configuration yml file")
        self.parser.add_argument('--overwrite_config', action='store_true', help="overwrite config files if they exist")
        self.parser.add_argument('--model', default='biggan', help="pretrained model to use, e.g. biggan, stylegan")
        parser.add_argument('--transform', default="zoom", help="transform operation, e.g. zoom, shiftx, color, rotate2d"),
        parser.add_argument('--num_samples', type=int, default=20000, help='number of latent z samples')
        parser.add_argument('--loss', type=str, default='l2', help='loss to use for training', choices=['l2', 'lpips'])
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
        parser.add_argument('--walk_type', type=str, default='NNz', choices=['NNz', 'linear'], help='type of latent z walk')
        parser.add_argument('--models_dir', type=str, default="./models", help="output directory for saved checkpoints")
        parser.add_argument('--model_save_freq', type=int, default=400, help="saves checkpoints after this many batches")
        parser.add_argument('--name', type=str, help="experiment name, saved within models_dir")
        parser.add_argument('--suffix', type=str, help="suffix for experiment name")
        parser.add_argument('--prefix', type=str, help="prefix for experiment name")
        parser.add_argument("--gpu", default="", type=str, help='GPUs to use (leave blank for CPU only)')

        # NN walk parameters
        group = parser.add_argument_group('nn', 'parameters used to specify NN walk')
        group.add_argument('--eps', type=float, help="step size of each NN block")
        group.add_argument('--num_steps', type=int, help="number of NN blocks")

        # color transformation parameters
        group = parser.add_argument_group('color', 'parameters used for color walk')
        group.add_argument('--channel', type=int, help="which channel to modify; if unspecified, modifies all channels for linear walks, and luminance for NN walks")

        # biggan walk parameters
        group = parser.add_argument_group('biggan', 'parameters used for biggan walk')
        group.add_argument('--category', type=int, help="which category to train on; if unspecified uses all categories")

        # stylegan walk parameters
        # Official training configs for StyleGAN, targeted mainly for anime.

        if 1:
            desc          = 'sgan'                                                                 # 包含在结果子目录名称中的描述字符串。
            train         = EasyDict(run_func_name='train.joint_train')         # 训练过程设置。
            G             = EasyDict(func_name='training.networks_stylegan.G_style')               # 生成网络架构设置。
            D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # 判别网络架构设置。
            G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # 生成网络优化器设置。
            D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # 判别网络优化器设置。
            G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating_steer')           # 生成损失设置。
            D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # 判别损失设置。
            dataset       = EasyDict()                                                             # 数据集设置，在后文确认。
            sched         = EasyDict()                                                             # 训练计划设置，在后文确认。
            grid          = EasyDict(size='4k', layout='random')                                   # setup_snapshot_image_grid()相关设置。
            metrics       = [metric_base.fid50k]                                                   # 指标方法设置。
            submit_config = dnnlib.SubmitConfig()                                                  # dnnlib.submit_run()相关设置。
            tf_config     = {'rnd.np_random_seed': 1000}                                           # tflib.init_tf()相关设置。

            # 数据集。
            desc += '-character';     dataset = EasyDict(tfrecord_dir='character');                 #train.mirror_augment = True
            
            # GPU数量。
            desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
            #desc += '-2gpu'; submit_config.num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
            #desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
            #desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

            # 默认设置。
            train = EasyDict(total_kimg = 25000)
            sched.lod_initial_resolution = 8
            sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
            sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

            kwargs = EasyDict(is_train=True)
            kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
            kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
            kwargs.submit_config = copy.deepcopy(submit_config)
            kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
            kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
            kwargs.submit_config.run_desc = desc

        else:
            kwargs = EasyDict(is_train=False)

        group = parser.add_argument_group('stylegan', 'parameters used for stylegan walk')
        group.add_argument('--dataset', default="anime", help="which dataset to use for pretrained stylegan, e.g. cars, cats, celebahq")
        group.add_argument('--latent', default="w", help="which latent space to use; z or w")
        group.add_argument('--truncation_psi', default=1.0, help="truncation for NN walk in w")
        group.add_argument('--train_args', default=kwargs, help="kwargs for training stylegan")

        # pgan walk parameters
        group = parser.add_argument_group('pgan', 'parameters used for pgan walk')
        group.add_argument('--dset', default="celebahq", help="which dataset to use for pretrained pgan")

        self.initialized = True
        return self.parser

    def print_options(self, opt):
        opt_dict = OrderedDict()
        message = ''
        message += '----------------- Options ---------------\n'
        # top level options
        grouped_k = []
        for k, v in sorted(vars(opt).items()):
            if isinstance(v, argparse.Namespace):
                grouped_k.append((k, v))
                continue
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            opt_dict[k] = v
        # grouped options
        for k, v in grouped_k:
            message += '{} '.format(k).ljust(20, '-')
            message += '\n'
            opt_dict[k] = OrderedDict()
            for k1, v1 in sorted(vars(v).items()):
                comment = ''
                default = self.parser.get_default(k1)
                if v1 != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<30}{}\n'.format(str(k1), str(v1), comment)
                opt_dict[k][k1] = v1
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if hasattr(opt, 'output_dir'):
                expr_dir = opt.output_dir
        else:
            expr_dir ='./'
        os.makedirs(expr_dir, exist_ok=True)

        if not opt.overwrite_config:
            assert(not os.path.isfile(os.path.join(expr_dir, 'opt.txt'))), 'config file exists, use --overwrite_config'
            assert(not os.path.isfile(os.path.join(expr_dir, 'opt.yml'))), 'config file exists, use --overwrite_config'

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        file_name = os.path.join(expr_dir, 'opt.yml')
        with open(file_name, 'wt') as opt_file:
            opt_dict['overwrite_config'] = False
            yaml.dump(opt_dict, opt_file, default_flow_style=False)

    def _flatten_to_toplevel(self, data):
        args = {}
        for k, v in data.items():
            if isinstance(v, dict):
                args.update(self._flatten_to_toplevel(v))
            else:
                args[k] = v
        return args

    def parse(self, print_opt=True):
        ''' use update_fn() to do additional modifications on args
            before printing
        '''
        # initialize parser with basic options
        if not self.initialized:
            self.initialize()

        # parse options
        opt = self.parser.parse_args()

        # get arguments specified in config file
        if opt.config_file:
            data = yaml.load(opt.config_file, Loader=yaml.FullLoader)
            data = self._flatten_to_toplevel(data)
        else:
            data = {}

        # determine which options were specified
        # explicitly with command line args
        option_strings = {}
        for action_group in self.parser._action_groups:
            for action in action_group._group_actions:
                for option in action.option_strings:
                    option_strings[option] = action.dest
        specified_options = set([option_strings[x] for x in
                                 sys.argv if x in option_strings])

        # make hierarchical namespace wrt groups
        # positional and optional arguments in toplevel
        args = {}
        for group in self.parser._action_groups:
        # by default, take the result from argparse
            # unless was specified in config file and not in command line
            group_dict={a.dest: data[a.dest] if a.dest in data
                        and a.dest not in specified_options
                        else getattr(opt, a.dest, None)
                        for a in group._group_actions}
            if group.title == 'positional arguments' or \
               group.title == 'optional arguments':
                args.update(group_dict)
            else:
                args[group.title] = argparse.Namespace(**group_dict)

        opt = argparse.Namespace(**args)
        delattr(opt, 'config_file')

        # output directory
        if opt.name:
            output_dir = opt.name
        else:
            output_dir = '_'.join([opt.model, opt.transform, opt.walk_type,
                                    'lr'+str(opt.learning_rate), opt.loss])
            if opt.model == 'biggan':
                subopt = opt.biggan
                if subopt.category:
                    output_dir += '_cat{}'.format(subopt.category)
            elif opt.model == 'stylegan':
                subopt = opt.stylegan
                output_dir += '_{}'.format(subopt.dataset)
                output_dir += '_{}'.format(subopt.latent)
            elif opt.model == 'pgan':
                subopt = opt.pgan
                output_dir += '_{}'.format(subopt.dset)
            if opt.walk_type.startswith('NN'):
                subopt = opt.nn
                if subopt.eps:
                    output_dir += '_eps{}'.format(subopt.eps)
                if subopt.num_steps:
                    output_dir += '_nsteps{}'.format(subopt.num_steps)
            if opt.transform.startswith('color') and opt.color.channel is not None:
                output_dir += '_chn{}'.format(opt.color.channel)


        if opt.suffix:
            output_dir += opt.suffix
        if opt.prefix:
            output_dir = opt.prefix + output_dir

        opt.output_dir = os.path.join(opt.models_dir, output_dir)


        # write the configurations to disk
        if print_opt:
            self.print_options(opt)

        self.opt = opt
        return opt