import os
import sys
import logging
from glob import glob
import numpy as np
import cv2 as cv
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from shutil import copyfile
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from modules.dataset import LLFFDataset
from modules.renderer import DirectRenderer, ProLiFRenderer, ProLiFEmbRenderer, \
                             NeRFPatchRenderer, NeRFRenderer, RSENRenderer
from modules.supervisor import Regularizer, ClipSupervisor, PatchSupervisor, PixelSupervisor

from modules.fields.prolif import construct_adam_state_dict_wn


class Runner:
    def __init__(self, conf, mode='train', is_continue=False, case='CASE_NAME', device='cuda'):
        self.device = torch.device(device)
        self.conf = conf
        self.writer = None
        self.base_dir = conf['device']['base_dir']
        self.base_exp_dir = conf['base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = LLFFDataset(conf['LLFFDataset'])

        self.iter_step = 0

        # Training parameters
        train_conf = conf['train']
        self.end_iter = train_conf['end_iter']
        self.save_freq = train_conf['save_freq']
        self.report_freq = train_conf['report_freq']
        self.val_freq = train_conf['val_freq']
        self.vis_freq = train_conf['vis_freq']
        self.vis_down_level = train_conf['vis_down_level']
        self.video_freq = train_conf['video_freq']
        self.tsboard_freq = train_conf['tsboard_freq']
        self.render_batch_size = train_conf['render_batch_size']

        # learning rate
        self.learning_rate = train_conf['learning_rate']
        self.learning_rate_alpha = train_conf['learning_rate_alpha']

        self.mile_stones = train_conf['mile_stones']
        self.sub_div_inputs = train_conf['sub_div_inputs']
        self.sub_div_outputs = train_conf['sub_div_outputs']

        self.ms_idx = 0
        self.mode = mode


        # renderer
        renderer_list = ['ProLiFRenderer', 'ProLiFEmbRenderer', 'DirectRenderer', 'NeRFRenderer',
                         'NeRFPatchRenderer', 'RSENRenderer', 'NeuLFRenderer']
        self.renderer = None
        for r_name in renderer_list:
            if r_name in conf:
                assert self.renderer is None
                self.renderer = getattr(sys.modules[__name__], r_name)(**conf[r_name])

        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr=self.learning_rate)

        # supervisors
        supervisor_list = ['Regularizer', 'PixelSupervisor', 'PatchSupervisor', 'ClipSupervisor']
        self.supervisors = []
        for s_name in supervisor_list:
            if s_name in conf:
                self.supervisors.append(getattr(sys.modules[__name__], s_name)(self.dataset, self.renderer, **conf[s_name]))

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = glob(os.path.join(self.base_exp_dir, 'checkpoints', '*.pth'))
            model_list = []
            for model_path in model_list_raw:
                model_name = os.path.split(model_path)[-1]
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

            if latest_model_name is not None:
                logging.info('Find checkpoint: {}'.format(latest_model_name))
                self.load_checkpoint(latest_model_name)

    def train(self):
        # Backup codes and configs for debug
        self.file_backup()

        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        for iter_i in tqdm(range(res_step)):
            loss = 0.
            for supervisor in self.supervisors:
                cur_loss, cur_loss_dict = supervisor.get_loss()
                loss = loss + cur_loss
                for key in cur_loss_dict:
                    self.writer.add_scalar('{}/{}'.format(supervisor.__class__.__name__, key),
                                           cur_loss_dict[key], self.iter_step)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            for supervisor in self.supervisors:
                supervisor.update_progress(self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter: {:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.val_freq == 0:
                self.calc_validation(self.writer)

            if self.iter_step % self.vis_freq == 0:
                self.visualize_image(resolution_level=self.vis_down_level)

            connect = False
            if self.mile_stones[self.ms_idx + 1] <= self.iter_step < self.end_iter:
                connect = True
                self.add_connections()
                self.ms_idx += 1
                self.render_path(self.dataset.render_poses)
            elif self.iter_step % self.video_freq == 0:
                self.render_path(self.dataset.render_poses)

            self.update_learning_rate()

            if self.iter_step % self.save_freq == 0 or connect or self.iter_step >= self.end_iter:
                self.save_checkpoint()

    def add_connections(self, update_optim=True):
        assert isinstance(self.renderer, ProLiFRenderer) or isinstance(self.renderer, ProLiFEmbRenderer)
        new_state_dict = None
        if update_optim:
            # update Adams' state dict
            assert self.ms_idx == self.renderer.stage
            field_confs = []
            if isinstance(self.renderer, ProLiFRenderer):
                field_conf = {
                    'org_n_sub_fields': self.renderer.field.n_sub_fields,
                    'n_layers': self.renderer.field.n_layers,
                    'skips': self.renderer.field_conf.skips,
                    'skip_dim': self.renderer.field_conf.d_hidden * 2 ** self.renderer.stage,
                    'sub_div_inputs': self.sub_div_inputs[self.ms_idx],
                    'sub_div_outputs': self.sub_div_outputs[self.ms_idx],
                }
                field_confs.append(field_conf)
            else:
                geo_field_conf = {
                    'org_n_sub_fields': self.renderer.geo_field.n_sub_fields,
                    'n_layers': self.renderer.geo_field.n_layers,
                    'skips': self.renderer.geo_field_conf.skips,
                    'skip_dim': self.renderer.geo_field_conf.d_hidden * 2 ** self.renderer.stage,
                    'sub_div_inputs': self.sub_div_inputs[self.ms_idx],
                    'sub_div_outputs': self.sub_div_outputs[self.ms_idx],
                    'input_coord_dim': self.renderer.geo_weights_conf.d_in,
                    'output_coord_dim': self.renderer.geo_weights_conf.d_out,
                }
                field_confs.append(geo_field_conf)
                app_field_conf = {
                    'org_n_sub_fields': self.renderer.app_field.n_sub_fields,
                    'n_layers': self.renderer.app_field.n_layers,
                    'skips': self.renderer.app_field_conf.skips,
                    'skip_dim': self.renderer.app_field_conf.d_hidden * 2 ** self.renderer.stage,
                    'sub_div_inputs': self.sub_div_inputs[self.ms_idx],
                    'sub_div_outputs': False,
                    'input_coord_dim': self.renderer.app_weights_conf.d_in,
                    'output_coord_dim': self.renderer.app_weights_conf.d_out,
                }
                field_confs.append(app_field_conf)
                rgb_field_conf = {
                    'org_n_sub_fields': self.renderer.rgb_field.n_sub_fields,
                    'n_layers': self.renderer.rgb_field.n_layers,
                    'skips': self.renderer.rgb_field_conf.skips,
                    'skip_dim': self.renderer.rgb_field_conf.d_hidden * 2 ** self.renderer.stage,
                    'sub_div_inputs': False,
                    'sub_div_outputs': self.sub_div_outputs[self.ms_idx],
                    'input_coord_dim': self.renderer.rgb_weights_conf.d_in,
                    'output_coord_dim': self.renderer.rgb_weights_conf.d_out,
                }
                field_confs.append(rgb_field_conf)
            new_state_dict = construct_adam_state_dict_wn(self.optimizer.state_dict(), field_confs)

        del self.optimizer
        self.renderer.merge(sub_div_inputs=self.sub_div_inputs[self.ms_idx],
                            sub_div_outputs=self.sub_div_outputs[self.ms_idx])
        self.optimizer = torch.optim.Adam(self.renderer.parameters(), lr=self.learning_rate)
        if update_optim:
            self.optimizer.load_state_dict(new_state_dict)

        self.update_learning_rate()

    def get_image_perm(self):
        image_list = [i for i in range(self.dataset.n_images) if i % 8 != 0]
        image_list = torch.tensor(image_list).cuda()
        image_list = image_list[torch.randperm(len(image_list))]
        return image_list

    def update_learning_rate(self):
        alpha = self.learning_rate_alpha

        progress = self.iter_step / self.end_iter
        alpha = self.learning_rate_alpha
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        if self.writer is not None:
            self.writer.add_scalar('Statistics/lr', self.learning_rate * learning_factor, self.iter_step)

    def file_backup(self):
        dir_lis = self.conf['recording']
        rec_dir = os.path.join(self.base_exp_dir, 'rec')
        for dir_name in dir_lis:
            def_dir_name = os.path.join(self.base_dir, dir_name)
            file_lis = glob(def_dir_name)
            for file_name in file_lis:
                new_file_name = file_name.replace(self.base_dir, rec_dir)
                os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
                copyfile(file_name, new_file_name)

        OmegaConf.save(self.conf, os.path.join(rec_dir, 'config.yaml'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.iter_step = checkpoint['iter_step']
        while self.iter_step >= self.mile_stones[self.ms_idx + 1] and self.mile_stones[self.ms_idx + 1] < self.end_iter:
            self.add_connections(update_optim=False)
            self.ms_idx += 1

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.renderer.load_state_dict(checkpoint['renderer'])
        self.update_learning_rate()

        for supervisor in self.supervisors:
            supervisor.update_progress(self.iter_step)

    def save_checkpoint(self):
        checkpoint = {
            'renderer': self.renderer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>8d}.pth'.format(self.iter_step)))

    def calc_validation(self, writer, visualize=False):
        all_psnr = []
        out_dir = os.path.join(self.base_exp_dir, 'validation', '{}'.format(self.iter_step))
        for i in tqdm(range(0, self.dataset.n_images, 8)):
            with torch.no_grad():
                pred_image, depth_image = self.render_image(i)
            pred_image = pred_image / 255.0
            gt_image = self.dataset.images[i].cpu().numpy()
            mse = ((pred_image - gt_image)**2).mean()
            psnr = 20.0 * np.log10(1.0 / (np.sqrt(mse)))
            all_psnr.append(psnr)

            if visualize:
                os.makedirs(out_dir, exist_ok=True)
                # out_image = np.concatenate([pred_image * 255.0, gt_image * 255.0], axis=1)
                cv.imwrite(os.path.join(out_dir, '{:0>3d}_gt.png'.format(i)), gt_image * 255.0)
                cv.imwrite(os.path.join(out_dir, '{:0>3d}_me.png'.format(i)), pred_image * 255.0)
                cv.imwrite(os.path.join(out_dir, '{:0>3d}_depth.png'.format(i)), depth_image)
        psnr = np.array(all_psnr).mean()
        print('Val PSNR:', psnr)
        if writer is not None:
            writer.add_scalar('Validation/test_psnr', psnr, self.iter_step)

    def render_image(self, idx, resolution_level=1):
        out_colors_lfn = []
        out_depths_lfn = []

        if isinstance(self.dataset, LLFFDataset):
            assert idx >= 0
            coord = self.dataset.coords_of_camera(idx, down_level=resolution_level)
        else:
            coord = self.dataset.rand_coords_patch_from_rand_pose(self.dataset.H, self.dataset.W)
        coord = coord.reshape(-1, 4).split(self.render_batch_size)

        for coord_batch in coord:
            # render_depths:
            render_result = self.renderer.full_query(coord_batch, perturb=False, idx=idx, mode='eval')
            depth = render_result['depths'].detach().cpu().numpy()
            out_depths_lfn.append(depth)

            # render colors
            colors = render_result['colors'].detach().cpu().numpy()
            out_colors_lfn.append(colors)


        H = self.dataset.H // resolution_level
        W = self.dataset.W // resolution_level

        out_depths_lfn = np.round(np.concatenate(out_depths_lfn, axis=0).reshape([H, W, 1]) * 255)\
                         .clip(0, 255).astype(np.uint8)
        out_depths_lfn = cv.applyColorMap(out_depths_lfn, cv.COLORMAP_JET)

        out_colors_lfn = np.round(np.concatenate(out_colors_lfn, axis=0).reshape([H, W, 3]) * 255)\
                         .clip(0, 255).astype(np.uint8)
        return out_colors_lfn, out_depths_lfn

    def render_image_from_pose(self, c2w, resolution_level=1, render_depth=True):
        coord = self.dataset.coords_from_pose(c2w, down_level=resolution_level)
        coord = coord.reshape(-1, 4).split(self.render_batch_size)

        out_colors_lfn = []
        out_depths_lfn = []
        for coord_batch in coord:
            # render_depths:
            render_results = self.renderer.full_query(coord_batch, perturb=False, idx=0, mode='eval')
            if render_depth:
                depth = render_results['depths'].detach().cpu().numpy()
                out_depths_lfn.append(depth)
            # render colors
            colors = render_results['colors'].detach().cpu().numpy()
            out_colors_lfn.append(colors)
            del render_results

        H = self.dataset.H // resolution_level
        W = self.dataset.W // resolution_level
        if render_depth:
            out_depths_lfn = (np.concatenate(out_depths_lfn, axis=0).reshape([H, W, 1]) * 255).clip(0, 255).astype(
                np.uint8)
            out_depths_lfn = cv.applyColorMap(out_depths_lfn, cv.COLORMAP_JET)

        out_colors_lfn = (
                    np.concatenate(out_colors_lfn, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)
        return out_colors_lfn, out_depths_lfn

    def visualize_image(self, idx=-1, resolution_level=1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        # visualize image
        with torch.no_grad():
            image, depth = self.render_image(idx, resolution_level=resolution_level)

        os.makedirs(os.path.join(self.base_exp_dir, 'render_images'), exist_ok=True)
        if isinstance(self.dataset, LLFFDataset):
            ref_img = self.dataset.images[idx].detach().cpu().numpy() * 255.0
            ref_img = cv.resize(ref_img, (self.dataset.W // resolution_level, self.dataset.H // resolution_level))
            img = np.concatenate([depth, image, ref_img.astype(np.uint8)], axis=1)
        else:
            img = np.concatenate([depth, image], axis=1)
        cv.imwrite(os.path.join(self.base_exp_dir, 'render_images', '{:0>8d}_{}.jpg'.format(self.iter_step, idx)), img)

    def render_path(self, render_poses, cat_closest_image=False):
        images = []
        for i, c2w in enumerate(tqdm(render_poses)):
            rgb, depths = self.render_image_from_pose(c2w, render_depth=True, resolution_level=self.vis_down_level)
            if cat_closest_image:
                closest_rgb = self.dataset.closest_image(c2w, resolution_level=self.vis_down_level)
                images.append(np.concatenate([closest_rgb, rgb, depths], axis=1))
            else:
                images.append(np.concatenate([rgb, depths], axis=1))

        out_dir = os.path.join(self.base_exp_dir, 'video')
        os.makedirs(out_dir, exist_ok=True)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(out_dir, '{:0>8d}.mp4'.format(self.iter_step)), fourcc, 30,
            (w, h))
        for image in images:
            writer.write(image)

        writer.release()

    @torch.no_grad()
    def visualize_weight(self):
        out_dir = os.path.join(self.base_exp_dir, 'weights_vis', '{}'.format(self.iter_step))
        os.makedirs(out_dir, exist_ok=True)
        for i in range(self.renderer.field.n_layers):
            weights = []
            for j in range(self.renderer.field.n_sub_fields):
                weight_v = self.renderer.field.sub_fields[j].layers[i].weight_v.data.clone()
                weight_g = self.renderer.field.sub_fields[j].layers[i].weight_g.data.clone()
                weight = weight_g * weight_v / torch.linalg.norm(weight_v, ord=2, dim=-1, keepdim=True)
                weights.append(weight.cpu().numpy())
            weights = np.concatenate(weights, axis=1)
            weights = (np.abs(weights) / np.max(np.abs(weights)) * 255.0).astype(np.uint8)
            weights = cv.applyColorMap(weights, cv.COLORMAP_JET)
            cv.imwrite(os.path.join(out_dir, 'layer_{}.png'.format(i)), weights)

    def vis_epi(self):
        # Visualize epipolar image
        v, t = 0., 0.
        res = 512
        u_samples = torch.linspace(-1, 1, res)
        s_samples = torch.linspace(-1, 1, res)

        u_samples, s_samples = torch.meshgrid(u_samples, s_samples, indexing='ij')
        coords = torch.stack([u_samples,
                              torch.ones_like(u_samples) * v, s_samples,
                              torch.ones_like(s_samples) * t], dim=-1).reshape(-1, 4)
        coords = coords.split(self.render_batch_size)

        out_colors = []
        for coords_batch in coords:
            color = self.renderer(coords_batch, mode='eval').detach().cpu().numpy()
            out_colors.append(color)

        out_colors = np.concatenate(out_colors, axis=0).reshape([res, res, 3]) * 255.0

        out_dir = os.path.join(self.base_exp_dir, 'epi_images')
        os.makedirs(out_dir, exist_ok=True)
        cv.imwrite(os.path.join(out_dir, '{:0>8d}.png'.format(self.iter_step)), out_colors)

    def calc_model_size(self):
        model = self.renderer
        param_size = 0.
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0.
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))


@hydra.main(config_path="confs", config_name="default")
def my_app(cfg: DictConfig) -> None:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mode = cfg['mode']
    runner = Runner(cfg, mode=mode, case='fortress', is_continue=cfg['is_continue'])

    if mode == 'train':
        runner.train()
    elif mode == 'vis_weight':
        runner.visualize_weight()
    elif mode == 'video':
        runner.render_path(runner.dataset.render_poses, cat_closest_image=True)
    elif mode == 'validate':
        runner.calc_validation(writer=None, visualize=True)
    elif mode == 'vis_epi':
        runner.vis_epi()
    elif mode == 'calc_model_size':
        runner.calc_model_size()

if __name__ == "__main__":
    my_app()
