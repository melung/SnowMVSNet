from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from torchvision import transforms
import torch

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, interval_scale=1.06, crop_h=512, crop_w=640, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = 192  # Hardcode
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        self.rt = kwargs.get("rt", False)
        self.use_raw_train = kwargs.get("use_raw_train", False)
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.stage_info = {"scale":[0.125,0.25,0.5,1.0]}
        self.img_mean = [0.5, 0.5, 0.5]
        self.img_std = [0.5, 0.5, 0.5]
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        # print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)
    def crop_cam(self, intrinsics, h, w, new_h=None, new_w=None, base=8):
        if new_h is None or new_w is None:
            new_h = h // base * base
            new_w = w // base * base

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
            new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
            return new_intrinsics
        else:
            return intrinsics
        
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval
    
    def scale_cam(self, intrinsics, h=None, w=None, max_h=None, max_w=None, scale=None):
        if scale:
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0, :] *= scale
            new_intrinsics[1, :] *= scale
        elif h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0, :] *= scale
            new_intrinsics[1, :] *= scale
        return new_intrinsics
    
    def read_depth(self, filename, scale):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32) * scale
    
    def read_img(self, filename, color_mode=None):
        if color_mode == "BGR":
            img = cv2.imread(filename)
        elif color_mode == "RGB" or color_mode is None:
            img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        #print(np.shape(img))
        return np_img
    def norm_img(self, np_img, self_norm=False, img_mean=None, img_std=None):
        if self_norm:
            var = np.var(np_img, axis=(0, 1), keepdims=True)
            mean = np.mean(np_img, axis=(0, 1), keepdims=True)
            np_img = (np_img - mean) / (np.sqrt(var) + 1e-7)
            return np_img
        else:
            # scale 0~255 to 0~1
            np_img = np_img / 255.
            if (img_mean is not None) and (img_std is not None):
                # scale with given mean and std
                img_mean = np.array(img_mean, dtype=np.float32)
                img_std = np.array(img_std, dtype=np.float32)
                np_img = (np_img - img_mean) / img_std
        return np_img


    def crop_img(self, img, new_h=None, new_w=None, base=8):
        h, w = img.shape[:2]

        if new_h is None or new_w is None:
            new_h = h // base * base
            new_w = w // base * base

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            img = img[start_h:finish_h, start_w:finish_w]
        return img
    
    def crop_img_any(self, img, start_h, start_w, new_h, new_w):
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        img = img[start_h:finish_h, start_w:finish_w]
        return img
    
    def crop_cam_any(self, intrinsics, start_h, start_w):
        new_intrinsics = np.copy(intrinsics)
        # principle point:
        new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
        new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
        return new_intrinsics
    
    def prepare_img(self, hr_img):
        h, w = hr_img.shape
        if not self.use_raw_train:
            #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
            #downsample
            hr_img_ds = cv2.resize(hr_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            h, w = hr_img_ds.shape
            target_h, target_w = 512, 640
            start_h, start_w = (h - target_h)//2, (w - target_w)//2
            hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]
        elif self.use_raw_train:
            hr_img_crop = hr_img[h//2-1024//2:h//2+1024//2, w//2-1280//2:w//2+1280//2]  # 1024, 1280, c
        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(np_img, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": np_img,
        }
        return np_img_ms
    
    def scale_img(self, img, max_h=None, max_w=None, scale=None, interpolation=cv2.INTER_LINEAR): 
            h, w = img.shape[:2]
            if scale:
                new_w, new_h = int(scale * w), int(scale * h)
                img = cv2.resize(img, [new_w, new_h], interpolation=interpolation)
            elif h > max_h or w > max_w:
                scale = 1.0 * max_h / h
                if scale * w > max_w:
                    scale = 1.0 * max_w / w
                new_w, new_h = int(scale * w), int(scale * h)
                img = cv2.resize(img, [new_w, new_h], interpolation=interpolation)
            return img

    def read_depth_hr(self, filename, scale):
        # read pfm depth file
        #w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth_lr, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views

        if self.mode == 'train' and self.rt:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            #scale = random.uniform(0.8, 1.25)
            scale = 1
        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        rotation_matrices = []
        
        imgs = []
        stage_num = len([0.125,0.25,0.5,1.0])
        proj_matrices = {str(i):[] for i in range(stage_num)}
        cams = {str(i):[] for i in range(stage_num)}
        ref_imgs = {str(i):None for i in range(stage_num)}
        ref_cams = {str(i):None for i in range(stage_num)}
        depths = {str(i):None for i in range(stage_num)}
        masks = {str(i):None for i in range(stage_num)}
        rand_start_h = None
        rand_start_w = None
        
        
        
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath,'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(vid)
            
            img = self.read_img(img_filename, color_mode="RGB")

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            extrinsics[:3,3] *= scale

            if i == 0:
                
                depth_max = depth_interval * self.ndepths + depth_min
                #print(depth_max)
                depth_values = np.array([depth_min * scale, depth_max * scale], dtype=np.float32)
                
                mask = self.read_img(mask_filename)
                mask = self.norm_img(mask)
                mask[mask > 0.0] = 1.0
                
                depth = self.read_depth(depth_filename, scale)
                depth_min_max = np.array([depth_values[0], depth_values[-1]], dtype=np.float32)
                
                depth_range = depth_min_max[1] - depth_min_max[0]

            ori_shape = img.shape
            img = self.crop_img(img=img, new_h=1024, new_w=1280) # 1024, 1280
            intrinsics = self.crop_cam(intrinsics=intrinsics, h=ori_shape[0], w=ori_shape[1], new_h=1024, new_w=1280)
            if i == 0:  # reference view
                mask = self.crop_img(img=mask, new_h=1024, new_w=1280) # 1024, 1280
                depth = self.crop_img(img=depth, new_h=1024, new_w=1280) # 1024, 1280

            h_o = img.shape[0]
            w_o = img.shape[1]
            if i == 0:  # reference view
                mask_o = mask.copy()
                if self.mode == "train":
                    while True:
                        rand_start_h = torch.randint(0, h_o - self.crop_h, [1]).item() # top
                        rand_start_w = torch.randint(0, w_o - self.crop_w, [1]).item() # left
                        tmp_mask = self.crop_img_any(img=mask_o, start_h=rand_start_h, start_w=rand_start_w, 
                            new_h=self.crop_h, new_w=self.crop_w)
                        # begin stages
                        for stage_id in range(stage_num):
                            stage_scale = self.stage_info["scale"][stage_id]
                            stage_mask = self.scale_img(img=tmp_mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                            masks[str(stage_id)] = stage_mask
                        if np.any(masks[str(0)] > 0.0):
                            break
                if self.mode == "val":
                    rand_start_h = (h_o - self.crop_h) // 2
                    rand_start_w = (w_o - self.crop_w) // 2
                    tmp_mask = self.crop_img_any(img=mask_o, start_h=rand_start_h, start_w=rand_start_w, 
                        new_h=self.crop_h, new_w=self.crop_w)
                    # begin stages
                    for stage_id in range(stage_num):
                        stage_scale = self.stage_info["scale"][stage_id]
                        stage_mask = self.scale_img(img=tmp_mask, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                        masks[str(stage_id)] = stage_mask

            img = self.crop_img_any(img=img, start_h=rand_start_h, start_w=rand_start_w, 
                new_h=self.crop_h, new_w=self.crop_w)
            intrinsics = self.crop_cam_any(intrinsics=intrinsics, start_h=rand_start_h, start_w=rand_start_w)
            if i == 0:  # reference view
                depth = self.crop_img_any(img=depth, start_h=rand_start_h, start_w=rand_start_w, 
                    new_h=self.crop_h, new_w=self.crop_w)

            # begin stages
            for stage_id in range(stage_num):
                stage_scale = self.stage_info["scale"][stage_id]
                stage_intrinsics = self.scale_cam(intrinsics=intrinsics, scale=stage_scale)
                #stage_proj_mat = extrinsics.copy()
                stage_proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
                
                #stage_proj_mat[:3, :4] = np.matmul(stage_intrinsics, stage_proj_mat[:3, :4])
                stage_proj_mat[0, :4, :4] = extrinsics
                stage_proj_mat[1, :3, :3] = stage_intrinsics
                
                proj_matrices[str(stage_id)].append(stage_proj_mat)

                stage_cam = np.zeros([2, 4, 4], dtype=np.float32)
                stage_cam[0, :4, :4] = extrinsics
                stage_cam[1, :3, :3] = stage_intrinsics
                cams[str(stage_id)].append(stage_cam)
                
            
                if i == 0:  # reference view
                    stage_ref_img = img.copy()
                    stage_ref_img = self.scale_img(stage_ref_img, scale=stage_scale, interpolation=cv2.INTER_LINEAR)
                    stage_ref_img = np.array(stage_ref_img, dtype=np.uint8)
                    ref_imgs[str(stage_id)] = stage_ref_img
                    ref_cams[str(stage_id)] = stage_cam
                    stage_depth = self.scale_img(img=depth, scale=stage_scale, interpolation=cv2.INTER_NEAREST)
                    depths[str(stage_id)] = stage_depth
            
            img = self.norm_img(img, self_norm=False, img_mean=self.img_mean, img_std=self.img_std)
            imgs.append(img.transpose(2,0,1))
            rotation_matrices.append(extrinsics[:3,:3])

        # binary_tree[0] is level, binary_tree[1] is key
          
        depth_min = depth_min_max[0]

        #imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        #print("img shape", np.shape(img.transpose(2,0,1)))
        rotation_matrices = np.stack(rotation_matrices)
        
        depth_ms = {
            "stage1": depths[str(0)],
            "stage2": depths[str(1)],
            "stage3": depths[str(2)],
            "stage4": depths[str(3)]
        }
        
        proj_matrices_ms = {
            "stage1": np.stack(proj_matrices[str(0)], axis=0),
            "stage2": np.stack(proj_matrices[str(1)], axis=0),
            "stage3": np.stack(proj_matrices[str(2)], axis=0),
            "stage4": np.stack(proj_matrices[str(3)], axis=0)
        }
        mask_ms = {
            "stage1": masks[str(0)],
            "stage2": masks[str(1)],
            "stage3": masks[str(2)],
            "stage4": masks[str(3)]
        }
        #print("mask shape ", np.shape(masks[str(0)]))
        #print("img",imgs.shape())
        return {"imgs": imgs,  # Nv C H W
                "proj_matrices": proj_matrices_ms,  # 4 stage of Nv 2 4 4
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask_ms,
                "R" : rotation_matrices}