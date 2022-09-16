import torch
import numpy as np
import os
import cv2 as cv
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def recenter_poses(poses, c2w=None):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    if c2w is None:
        c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


class LLFFDataset:
    def __init__(self, conf, bd_factor=.75, path_zflat=False):
        self.conf = conf
        self.device = torch.device('cuda')
        base_dir = conf['data_dir']
        self.base_dir = base_dir
        self.pose_arr = np.load(os.path.join(base_dir, 'poses_bounds.npy'))
        bounds = self.pose_arr[:, -2:].transpose([1, 0])

        poses = self.pose_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        h, w, f = poses[:, 4, 0]
        self.cx = w * 0.5
        self.cy = h * 0.5

        if 'target_width' in conf:
            # For the Shiny object dataset
            factor = w / conf['target_width']
        else:
            factor = conf['factor']

        sfx = ''
        resized = False
        if 'factor' in conf and conf['factor'] > 1:
            sfx = '_{}'.format(conf['factor'])
            img_dir = os.path.join(self.base_dir, 'images' + sfx)
            if not os.path.exists(img_dir):
                sfx = ''
            else:
                resized = True    # The images have been resized.
        img_dir = os.path.join(self.base_dir, 'images' + sfx)

        assert os.path.exists(img_dir), '{} does not exist, returning'.format(img_dir)

        img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if
                     f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        assert poses.shape[-1] == len(img_files), 'Mismatch between imgs and poses'

        poses[:, 4, :] = poses[:, 4, :] * 1. / factor
        h = int(np.round(h / factor))
        w = int(np.round(w / factor))

        self.cx = self.cx / factor
        self.cy = self.cy / factor

        if resized:
            images = [cv.imread(f) / 255. for f in img_files]
            assert images[0].shape[0] == h and images[0].shape[1] == w
        else:
            images = [cv.resize(cv.imread(f), (w, h), interpolation=cv.INTER_AREA)[..., :3] / 255. for f in img_files]
        images = np.stack(images, -1)

        # ------------------------

        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0).astype(np.float32)
        bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)

        sc = 1. if bd_factor is None else 1. / (bounds.min() * bd_factor)

        poses[:, :3, 3] *= sc
        bounds *= sc

        poses = recenter_poses(poses, c2w=None)
        c2w = poses_avg(poses)

        print('recentered', c2w.shape)

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.

        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #          zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

        self.poses = poses
        self.render_poses = np.stack(render_poses, axis=0).astype(np.float32)
        self.images = images
        self.bounds = bounds

        # -----------------------------------------------------------------------------------------------------
        rots = self.poses[:, :3, :3]
        eulers = []
        for i in range(len(rots)):
            rot = Rot.from_matrix(rots[i])
            euler = rot.as_euler(seq='xyz')
            eulers.append(euler)
        eulers = np.stack(eulers, axis=0)
        self.euler_min = np.min(eulers, axis=0).astype(np.float32)
        self.euler_max = np.max(eulers, axis=0).astype(np.float32)
        self.cam_pos_min = np.min(self.poses[:, :3, 3], axis=0).astype(np.float32)
        self.cam_pos_max = np.max(self.poses[:, :3, 3], axis=0).astype(np.float32)

        # -----------------------------------------------------------------------------------------------------
        self.hwf = self.poses[0, :3, -1]
        self.poses = torch.from_numpy(self.poses[:, :3, :4]).to(self.device)
        self.render_poses = torch.from_numpy(self.render_poses).to(self.device)[:, :3, :4]
        self.images_np = self.images.copy()
        self.images = torch.from_numpy(self.images.astype(np.float32)).cpu()

        self.n_images = len(self.images)
        self.H = int(self.hwf[0])
        self.W = int(self.hwf[1])
        self.focal = float(self.hwf[2])

    def interpolate_pose(self, pose_0, pose_1, ratio):
        pose_0 = np.concatenate([pose_0, np.array([0, 0, 0, 1], dtype=np.float32)[None]], axis=0)
        pose_1 = np.concatenate([pose_1, np.array([0, 0, 0, 1], dtype=np.float32)[None]], axis=0)

        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        key_rots = [rot_0, rot_1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        return pose[:3, :4]

    def rand_pose(self):
        euler = np.random.rand(3) * (self.euler_max - self.euler_min) + self.euler_min
        pos = np.random.rand(3) * (self.cam_pos_max - self.cam_pos_min) + self.cam_pos_min
        rot = Rot.from_euler('xyz', euler)
        pose = np.diag([1., 1., 1., 1.]).astype(np.float32)
        pose[:3, :3] = rot.as_matrix().astype(np.float32)
        pose[:3, 3] = pos
        pose = torch.from_numpy(pose).to(self.device)

        return pose

    def rand_coords_patch_from_rand_pose(self, patch_h, patch_w, stride=1):
        pose = self.rand_pose()
        rays_o, rays_d = self.rand_rays_patch_from_pose(patch_h, patch_w, pose, stride=stride)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        coords = self.rays2coords(rays_o, rays_d)
        return coords

    def rand_rays_of_camera(self, idx, batch_size):
        i = torch.randint(low=0, high=self.H, size=[batch_size])
        j = torch.randint(low=0, high=self.W, size=[batch_size])
        dirs = torch.stack([(j - self.cx) / self.focal, -(i - self.cy) / self.focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * self.poses[idx, :3, :3], -1)
        rays_o = self.poses[idx, :3, -1].expand(rays_d.shape)
        return rays_o, rays_d, i, j

    def rand_coords_data_of_camera(self, idx, batch_size):
        rays_o, rays_d, i, j = self.rand_rays_of_camera(idx, batch_size)
        coord = self.rays2coords(rays_o, rays_d)
        color = self.images[idx][(i, j)]

        return coord, color


    def rays_of_camera(self, idx, down_level=1):
        i, j = torch.meshgrid(torch.linspace(0, self.W - 1, self.W // down_level),
                              torch.linspace(0, self.H - 1, self.H // down_level), indexing='ij')
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i - self.cx) / self.focal, -(j - self.cy) / self.focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * self.poses[idx, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = self.poses[idx, :3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def coords_of_camera(self, idx, raw=False, down_level=1):
        rays_o, rays_d = self.rays_of_camera(idx, down_level=down_level)  # Just get statistics
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        coord = self.rays2coords(rays_o, rays_d)
        return coord

    def coords_data_of_camera(self, idx, down_level=1):
        coords = self.coords_of_camera(idx, down_level=down_level)
        assert self.H % down_level == 0 and self.W % down_level == 0
        color = cv.resize(self.images_np[idx], (self.W // down_level, self.H // down_level), interpolation=cv.INTER_LINEAR)
        color = torch.from_numpy(color).to(self.device).reshape(-1, 3)

        return coords, color

    def rays_from_pose(self, pose, down_level=1):
        i, j = torch.meshgrid(torch.linspace(0, self.W - 1, self.W // down_level),
                              torch.linspace(0, self.H - 1, self.H // down_level), indexing='ij')
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i - self.cx) / self.focal, -(j - self.cy) / self.focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = pose[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def coords_from_pose(self, pose, down_level=1):
        rays_o, rays_d = self.rays_from_pose(pose, down_level=down_level)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        coord = self.rays2coords(rays_o, rays_d)
        return coord

    def rand_rays_patch_from_pose(self, patch_h, patch_w, pose, stride=1, h_low=-1, w_low=-1):
        if h_low < 0:
            h_low = np.random.rand() * (self.H - patch_h * stride)
        if w_low < 0:
            w_low = np.random.rand() * (self.W - patch_w * stride)
        i, j = torch.meshgrid(torch.linspace(w_low, w_low + patch_w * stride - 1, patch_w),
                              torch.linspace(h_low, h_low + patch_h * stride - 1, patch_h), indexing='ij')
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i - self.cx) / self.focal, -(j - self.cy) / self.focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = pose[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def rand_coords_data_patch_of_camera(self, idx, patch_h, patch_w, batch_size=1, stride=1):
        h_low = np.random.randint(self.H - patch_h * stride + 1, size=batch_size)
        w_low = np.random.randint(self.W - patch_w * stride + 1, size=batch_size)
        rays_o = []
        rays_d = []
        for i in range(batch_size):
            cur_rays_o, cur_rays_d = self.rand_rays_patch_from_pose(patch_h, patch_w, self.poses[idx],
                                                                    h_low=h_low[i], w_low=w_low[i], stride=stride)
            rays_o.append(cur_rays_o)
            rays_d.append(cur_rays_d)

        rays_o = torch.stack(rays_o, dim=0).reshape(-1, 3)
        rays_d = torch.stack(rays_d, dim=0).reshape(-1, 3)

        coords = self.rays2coords(rays_o.reshape(-1, 3), rays_d.reshape(-1, 3))
        colors = [
            self.images[idx][h_low[i]: h_low[i] + patch_h * stride, w_low[i]: w_low[i] + patch_w * stride]
            for i in range(batch_size)
        ]
        colors_numpy = [
            cv.resize(colors[i].cpu().numpy(), (patch_w, patch_h), interpolation=cv.INTER_AREA)
            for i in range(batch_size)
        ]
        colors = [ torch.from_numpy(colors_numpy[i]) for i in range(batch_size) ]
        color = torch.stack(colors, dim=0).to(self.device).reshape(-1, 3)
        # color = self.images[idx][h_low: h_low + patch_h, w_low: w_low + patch_w].to(self.device).reshape(-1, 3)

        return coords, color

    def rand_rays_from_pose(self, batch_size, pose, extend=1.0):
        i = ((torch.rand(batch_size) * 2.0 - 1.0) * extend + 1.0) * self.H * 0.5
        j = ((torch.rand(batch_size) * 2.0 - 1.0) * extend + 1.0) * self.W * 0.5
        dirs = torch.stack([(j - self.cx) / self.focal, -(i - self.cy) / self.focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)
        rays_o = pose[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def rand_coords_from_pose(self, pose, batch_size, extend=1.2):
        rays_o, rays_d = self.rand_rays_from_pose(batch_size, pose, extend=extend)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        coord = self.rays2coords(rays_o, rays_d)
        return coord

    def rand_coords_from_rand_pose(self, batch_size, extend=1.0):
        pose = self.rand_pose()
        return self.rand_coords_from_pose(pose, batch_size, extend=extend)

    def ndc_rays_of_camera(self, idx, down_level=1):
        rays_o, rays_d = self.rays_of_camera(idx, down_level=down_level)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        return self.ndc_rays(rays_o, rays_d)

    def ndc_rays(self, rays_o, rays_d):
        # Shift ray origins to near plane
        near = 1.
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection
        W = self.W
        H = self.H
        focal = self.focal
        o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]

        d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)
        return rays_o, rays_d

    def rays2coords(self, rays_o, rays_d):
        ndc_o, ndc_d = self.ndc_rays(rays_o, rays_d)
        coord = torch.cat([ndc_o[:, :2], (ndc_o + ndc_d)[:, :2]], dim=-1)
        return coord

    def closest_image(self, c2w, resolution_level=1):
        dis = 1e9
        idx = -1
        for i in range(self.n_images):
            cur_dis = ((c2w - self.poses[i])[:3, 3]**2).sum().item()
            if dis > cur_dis:
                dis = cur_dis
                idx = i
        h = self.H // resolution_level
        w = self.W // resolution_level
        img = (self.images[idx].cpu().numpy() * 255.0).astype(np.uint8)
        img = cv.resize(img, (w, h))
        return img
