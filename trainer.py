import os
import json
from enum import Enum
import torch
from torch import nn
from tensorboardX import SummaryWriter

from utils.util import make_noise
from train_log import MeanTracker
from visualization import make_interpolation_chart, fig_to_image
from latent_deformator import DeformatorType, normal_projection_stat


class DeformatorLoss(Enum):
    L2 = 0,
    RELATIVE = 1,
    STAT = 2,
    NONE = 3,


class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,


class Params(object):
    def __init__(self, **kwargs):
        self.global_deformation = False
        self.deformation_loss = DeformatorLoss.NONE
        self.shift_scale = 6.0
        self.min_shift = 0.25
        self.shift_distribution = ShiftDistribution.UNIFORM

        self.deformator_lr = 0.0001
        self.shift_predictor_lr = 0.0001
        self.n_steps = 5*int(1e+5)
        self.batch_size = 6
        self.max_latent_ind = 120

        self.label_weight = 2.0
        self.shift_weight = 0.5
        self.transform_weight = 1

        self.deformation_loss_weight = 2.0
        self.z_norm_loss_low_bound = 1.1
        self.z_mean_weight = 200.0
        self.z_std_weight = 200.0

        self.steps_per_log = 10
        self.steps_per_save = 10000
        self.steps_per_img_log = 1000
        self.steps_per_backup = 1000

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val


class Trainer(object):
    def __init__(self, params=Params(), out_dir='', out_json=None,
                 verbose=False):
        if verbose:
            print('Trainer inited with:\n{}'.format(str(params.__dict__)))
        self.p = params
        self.log_dir = out_dir
        self.cross_entropy = nn.CrossEntropyLoss()

        tb_dir = os.path.join(out_dir, 'tensorboard')
        self.models_dir = os.path.join(out_dir, 'models')
        self.images_dir = os.path.join(out_dir, 'images')
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.checkpoint = os.path.join(out_dir, 'checkpoint.pt')
        self.writer = SummaryWriter(tb_dir)
        self.out_json = out_json if out_json is not None else os.path.join(out_dir, 'stat.json')
        self.fixed_test_noise = None

    def make_shifts(self, latent_dim, shift=None):
        '''
        Returns:
            target_indices: one hot's index
            shifts: z_shift one hot处的值，即偏移量
            z_shift: one hot 向量
        '''
        target_indices = torch.randint(0, self.p.max_latent_ind, [self.p.batch_size], device='cuda')
        if shift is not None:
            shifts = shift
        elif self.p.shift_distribution == ShiftDistribution.NORMAL:
            # 返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。
            shifts =  torch.randn(target_indices.shape, device='cuda')
        elif self.p.shift_distribution == ShiftDistribution.UNIFORM:
            # 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.p.shift_scale * shifts
        shifts[(shifts < self.p.min_shift) & (shifts > 0)] = self.p.min_shift
        shifts[(shifts > -self.p.min_shift) & (shifts < 0)] = -self.p.min_shift

        if isinstance(latent_dim, int):
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.p.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def log_train(self, step, should_print=True, stats=()):
        if should_print:
            out_text = '{}% [step {}]'.format(int(100 * step / self.p.n_steps), step)
            for named_value in stats:
                out_text += (' | {}: {:.2f}'.format(*named_value))
            print(out_text)
        for named_value in stats:
            self.writer.add_scalar(named_value[0], named_value[1], step)

        with open(self.out_json, 'w') as out:
            stat_dict = {named_value[0]: named_value[1] for named_value in stats}
            json.dump(stat_dict, out)

    def log_interpolation(self, G, deformator, step):
        noise = make_noise(1, G.dim_z).cuda()
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
        for z, prefix in zip([noise, self.fixed_test_noise], ['rand', 'fixed']):
            fig = make_interpolation_chart(
                G, deformator, z=z, shifts_r=3 * self.p.shift_scale, shifts_count=3, dims_count=15,
                dpi=500)

            self.writer.add_figure('{}_deformed_interpolation'.format(prefix), fig, step)
            fig_to_image(fig).convert("RGB").save(
                os.path.join(self.images_dir, '{}_{}.jpg'.format(prefix, step)))

    def start_from_checkpoint(self, deformator, shift_predictor):
        step = 0
        if os.path.isfile(self.checkpoint):
            state_dict = torch.load(self.checkpoint)
            step = state_dict['step']
            deformator.load_state_dict(state_dict['deformator'])
            shift_predictor.load_state_dict(state_dict['shift_predictor'])
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, deformator, shift_predictor, step):
        state_dict = {
            'step': step,
            'deformator': deformator.state_dict(),
            'shift_predictor': shift_predictor.state_dict(),
        }
        torch.save(state_dict, self.checkpoint)

    def save_models(self, deformator, shift_predictor, step):
        torch.save(deformator.state_dict(),
                   os.path.join(self.models_dir, 'transform_deformator_{}.pt'.format(step)))
        torch.save(shift_predictor.state_dict(),
                   os.path.join(self.models_dir, 'transform_shift_predictor_{}.pt'.format(step)))

    def log_accuracy(self, G, deformator, shift_predictor, step):
        deformator.eval()
        shift_predictor.eval()

        accuracy = validate_classifier(G, deformator, shift_predictor, trainer=self)
        self.writer.add_scalar('accuracy', accuracy.item(), step)

        deformator.train()
        shift_predictor.train()
        return accuracy

    def log(self, G, deformator, shift_predictor, step, avgs):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, True, [avg.flush() for avg in avgs])

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(G, deformator, step)

        if step % self.p.steps_per_backup == 0 and step > 0:
            self.save_checkpoint(deformator, shift_predictor, step)
            accuracy = self.log_accuracy(G, deformator, shift_predictor, step)
            print('Step {} accuracy: {:.3}'.format(step, accuracy.item()))

        if step % self.p.steps_per_save == 0 and step > 0:
            self.save_models(deformator, shift_predictor, step)

    def cal_transform_loss(transform_model, criterion, out_img, color_channel = -1):
        if color_channel != -1:
            target_img, mask_out, numel_mask = transform_model.get_target_np(out_img.cpu().numpy(), transform_model.alpha_for_target, color_channel)
        else:
            target_img, mask_out, numel_mask = transform_model.get_target_np(out_img.cpu().numpy(), transform_model.alpha_for_target)
        target_tensor = torch.tensor(target_fn, device='cuda', dtype=torch.float32)
        mask_tensor = torch.tensor(mask_out, device='cuda', dtype=torch.float32)
        numel_mask_tensor = torch.tensor(numel_mask, device='cuda', dtype=torch.float32)
        loss = criterion(out_img * mask_tensor, target_tensor * mask_tensor) / numel_mask_tensor
        return loss

    def train(self, G, deformator, shift_predictor, transform_model):
        G.cuda().eval()
        deformator.cuda().train()
        shift_predictor.cuda().train()

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=self.p.deformator_lr) \
            if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
        shift_predictor_opt = torch.optim.Adam(
            shift_predictor.parameters(), lr=self.p.shift_predictor_lr)

        avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'),\
               MeanTracker('shift_loss'), MeanTracker('deformator_loss'), MeanTracker('transform_loss')
        avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss, avg_deformator_loss, avg_transform_loss = avgs

        recovered_step = self.start_from_checkpoint(deformator, shift_predictor)
        for step in range(recovered_step, self.p.n_steps, 1):
            G.zero_grad()
            deformator.zero_grad()
            shift_predictor.zero_grad()

            z = make_noise(self.p.batch_size, G.dim_z).cuda()
            z_orig = torch.clone(z)
            target_indices, shifts, z_shift = self.make_shifts(G.dim_z)

            for t_model in transform_model:
                alpha_for_graph, alpha_for_target = t_model.model.get_train_alpha(minibatch = 1)
                transform_model.t_model.alpha_for_graph = alpha_for_graph
                transform_model.t_model.alpha_for_target = alpha_for_target

            # alpha替换shift
            for index, target_indice in enumerate(target_indices):
                if target_indice == 0:
                    shifts[index] = transform_model.color.alpha_for_graph[0][0]
                    z_shift[index][target_indice] = transform_model.color.alpha_for_target[0][0]
                elif target_indice == 1:
                    shifts[index] = transform_model.color.alpha_for_graph[0][1]
                    z_shift[index][target_indice] = transform_model.color.alpha_for_target[0][1]
                elif target_indice == 2:
                    shifts[index] = transform_model.color.alpha_for_graph[0][2]
                    z_shift[index][target_indice] = transform_model.color.alpha_for_target[0][2]
                elif target_indice == 3:
                    shifts[index] = transform_model.zoom.alpha_for_graph[0]
                    z_shift[index][target_indice] = transform_model.zoom.alpha_for_target[0]
                elif target_indice == 4:
                    shifts[index] = transform_model.shiftx.alpha_for_graph[0]
                    z_shift[index][target_indice] = transform_model.shiftx.alpha_for_target[0]
                elif target_indice == 5:
                    shifts[index] = transform_model.shifty.alpha_for_graph[0]
                    z_shift[index][target_indice] = transform_model.shifty.alpha_for_target[0]

            # Deformation

            if self.p.global_deformation:
                z_shifted = deformator(z + z_shift)
                z = deformator(z)
            else:
                z_shifted = z + deformator(z_shift)
            imgs = G(z)
            imgs_shifted = G(z_shifted)

            criterion = nn.MSELoss(reduction='sum')
            transform_loss = 0
            for index, target_indice in enumerate(target_indices):
                if target_indice < 3:
                    transform_loss += self.cal_transform_loss(transform_model.color, criterion, imgs[index], color_channel=target_indice)
                elif target_indice == 3:
                    transform_loss += self.cal_transform_loss(transform_model.zoom, criterion, imgs[index])
                elif target_indice == 4:
                    transform_loss += self.cal_transform_loss(transform_model.shiftx, criterion, imgs[index])
                elif target_indice == 5:
                    transform_loss += self.cal_transform_loss(transform_model.shifty, criterion, imgs[index])
            transform_loss = self.p.transform_weight * transform_loss


            logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
            logit_loss = self.p.label_weight * self.cross_entropy(logits, target_indices)
            shift_loss = self.p.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))
            # Loss

            # deformator penalty
            if self.p.deformation_loss == DeformatorLoss.STAT:
                z_std, z_mean = normal_projection_stat(z)
                z_loss = self.p.z_mean_weight * torch.abs(z_mean) + \
                    self.p.z_std_weight * torch.abs(1.0 - z_std)

            elif self.p.deformation_loss == DeformatorLoss.L2:
                z_loss = self.p.deformation_loss_weight * torch.mean(torch.norm(z, dim=1))
                if z_loss < self.p.z_norm_loss_low_bound * torch.mean(torch.norm(z_orig, dim=1)):
                    z_loss = torch.tensor([0.0], device='cuda')

            elif self.p.deformation_loss == DeformatorLoss.RELATIVE:
                deformation_norm = torch.norm(z - z_shifted, dim=1)
                z_loss = self.p.deformation_loss_weight * torch.mean(torch.abs(deformation_norm - shifts))

            else:
                z_loss = torch.tensor([0.0], device='cuda')

            # total loss
            loss = logit_loss + shift_loss + z_loss + transform_loss
            loss.backward()

            if deformator_opt is not None:
                deformator_opt.step()
            shift_predictor_opt.step()

            # update statistics trackers
            avg_correct_percent.add(torch.mean(
                    (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
            avg_loss.add(loss.item())
            avg_label_loss.add(logit_loss.item())
            avg_shift_loss.add(shift_loss)
            avg_deformator_loss.add(z_loss.item())
            avg_transform_loss.add(transform_loss.item())

            self.log(G, deformator, shift_predictor, step, avgs)


@torch.no_grad()
def validate_classifier(G, deformator, shift_predictor, params_dict=None, trainer=None):
    n_steps = 100
    if trainer is None:
        trainer = Trainer(params=Params(**params_dict), verbose=False)

    percents = torch.empty([n_steps])
    for step in range(n_steps):
        z = make_noise(trainer.p.batch_size, G.dim_z).cuda()
        target_indices, shifts, z_shift = trainer.make_shifts(G.dim_z)

        if trainer.p.global_deformation:
            z_shifted = deformator(z + z_shift)
            z = deformator(z)
        else:
            z_shifted = z + deformator(z_shift)
        imgs = G(z)
        imgs_shifted = G(z_shifted)

        logits, _ = shift_predictor(imgs, imgs_shifted)
        percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

    return percents.mean()
