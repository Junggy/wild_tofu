import torch,os,glob
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision
import cv2
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch import nn

ToF_to_RS = torch.from_numpy( np.array([[0.9995575011766835, 0.007698921328204006, 0.02873201092521153, 0.04250089873790845],
                                        [-0.006708546718660541, 0.9993853535885749, -0.03440799955769654, 0.06625787988547474],
                                        [-0.028979255379458894, 0.03420002402076673, 0.998994775318986, -0.002945568589809818],
                                        [0, 0, 0, 1]])).float()

RS_to_ToF = torch.linalg.inv(ToF_to_RS)

K_rs = torch.from_numpy(np.array([[458.9553651696184, 0.0, 308.15745544433594],
                                       [0.0, 458.9553651696184, 262.75226974487305],
                                       [0.0, 0.0, 1.0]])).float()

K_tof = torch.from_numpy(np.array([[558.1649169921875, 0.0, 320.84777429724636],
                                        [0.0, 571.637451171875, 252.1800136421225],
                                        [0.0, 0.0, 1.0]])).float()

right_to_left = torch.from_numpy(np.array([[1,0,0,0.08518229882400204],
                                               [0,1,0,0],
                                               [0,0,1,0],
                                               [0,0,0,1]])).float()
left_to_right = torch.linalg.inv(right_to_left)

ToF_to_right = torch.matmul(left_to_right,ToF_to_RS)

class ToFDataset(Dataset):
    def __init__(self, base_list, split, img_size, crop_size=None, filtered=False,flip=False,test=False):

        self.split = split
        self.filtered = filtered
        self.crop_size = crop_size
        self.h,self.w = img_size
        self.generate_mesh_grid()
        self.generate_intrinsic_conversion_factor()
        self.flip = flip
        self.test = test

        if not self.test:
            self.depth_names, self.depth_warped_names, self.left_names, self.right_names, self.tof_names = self.read_file_names_from_multi_sequence(base_list)
            print(len(self.depth_names),len(self.depth_warped_names),len(self.left_names),len(self.right_names),len(self.tof_names))
            assert len(self.depth_names) == len(self.depth_warped_names) == len(self.left_names) == len(self.right_names) == len(self.tof_names)
            self.transforms = [TF.adjust_brightness,TF.adjust_contrast,TF.adjust_saturation,TF.adjust_hue]

        elif self.test:
            self.left_names, self.tof_names = self.read_file_names_test()
            print(len(self.left_names),len(self.tof_names))
            assert len(self.left_names) == len(self.tof_names)


    def transform_func(self,factor,img):

        for idx in range(4):
            img = self.transforms[factor[0][idx]](img,factor[1+factor[0][idx]])

        return img

    def __getitem__(self,idx):

        # do augmentation later
        if not self.test:
            depth = self.intrinsic_transform(torch.from_numpy(np.expand_dims(cv2.imread(self.depth_names[idx],-1).astype(np.float32)/1000,0)),"nn")
            depth_warped = torch.from_numpy(np.expand_dims(cv2.imread(self.depth_warped_names[idx],-1).astype(np.float32)/1000,0))
            right = self.intrinsic_transform(read_image(self.right_names[idx]) / 255, "bi")
        left = self.intrinsic_transform(read_image(self.left_names[idx]) / 255,"bi")


        tof_pre = read_image(self.tof_names[idx]) / 255
        tof = torch.zeros([4, 480, 640])
        tof[0] = tof_pre[0, :480, :]  # s
        tof[2] = tof_pre[0, 480 * 1:480 * 2, :]  # c
        tof[1] = tof_pre[0, 480 * 2:480 * 3, :]  # -s
        tof[3] = tof_pre[0, 480 * 3:480 * 4, :]  # c

        if self.test:
            fname = self.left_names[idx].split("\\")[-1]
            return left, tof, fname
        else:
            if self.crop_size != None:
                top_left = (torch.randint(0,int(self.h-self.crop_size[0]),(1,))[0],torch.randint(0,int(self.w-self.crop_size[1]),(1,))[0])
                depth = self.random_crop(depth,top_left)
                depth_warped = self.random_crop(depth_warped,top_left)
                left = self.random_crop(left,top_left)
                right = self.random_crop(right,top_left)
                tof = self.random_crop(tof,top_left)

                if self.flip:
                    if torch.rand(1)[0] > 0.5:
                        left = torch.flip(left,[2])
                        right = torch.flip(right, [2])
                        tof   = torch.flip(tof,[2])
                        depth = torch.flip(depth,[2])
                        depth_warped = torch.flip(depth_warped,[2])

            if self.split == "train":
                trans_ = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
                factors = torchvision.transforms.ColorJitter.get_params(trans_.brightness, trans_.contrast,
                                                                                     trans_.saturation, trans_.hue)
                left = self.transform_func(factors,left)
                right = self.transform_func(factors,right)

            assert (self.depth_names[idx].split("/")[-1] == self.depth_warped_names[idx].split("/")[-1] \
                    == self.left_names[idx].split("/")[-1]  == self.right_names[idx].split("/")[-1]), \
                    print(self.depth_names[idx].split("/")[-1], self.depth_warped_names[idx].split("/")[-1],
                          self.left_names[idx].split("/")[-1], self.right_names[idx].split("/")[-1])

            fname_ = self.depth_names[idx].split("/")
            scene, fname = fname_[-5],fname_[-1]
            return depth, depth_warped,left, right, tof, (scene,fname)

    def __len__(self):
        return len(self.tof_names)

    def random_crop(self,img,top_left):
        top,left = top_left
        crop_h,crop_w = self.crop_size
        img_cropped = img[:,top:top+crop_h,left:left+crop_w]
        return img_cropped

    def read_file_names_test(self):

        left_names = glob.glob("testset/RGB/*.png")
        tof_names = glob.glob("testset/i-ToF/*.png")

        return left_names, tof_names

    def read_file_names_from_multi_sequence(self,base_list):

        depth_names        = []
        depth_warped_names = []
        left_names         = []
        right_names        = []
        tof_names          = []

        for each_base in base_list:
            if self.split == "train":
                depth_train, depth_warped_train, left_train, right_train, tof_train = self.read_file_names(base=each_base, split="train",filtered=self.filtered)

                depth_names += depth_train
                depth_warped_names += depth_warped_train
                left_names += left_train
                right_names += right_train
                tof_names += tof_train

                depth_train, depth_warped_train, left_train, right_train, tof_train = self.read_file_names(base=each_base, split="val",filtered=self.filtered)

                depth_names += depth_train
                depth_warped_names += depth_warped_train
                left_names += left_train
                right_names += right_train
                tof_names += tof_train

            elif self.split == "val":
                depth_val, depth_warped_val, left_val, right_val, tof_val = self.read_file_names(base=each_base, split="val",filtered=self.filtered)

                depth_names += depth_val
                depth_warped_names += depth_warped_val
                left_names += left_val
                right_names += right_val
                tof_names += tof_val

        return depth_names, depth_warped_names, left_names, right_names, tof_names

    def read_file_names(self,base, split="train",filtered=False):
        depth_warped_base = base + "/_gt_warped/"
        depth_base = base + "/Depth/image_01/data/"
        if filtered:
            ToF_base = base + "/_kpn_filtered/"
        else:
            ToF_base = base + "/i-ToF/image_00/data/"
        left_base = base + "/RGB/image_02/data/"
        right_base = base + "/RGB/image_03/data/"

        split_name = base + "/split/test_files.txt"

        with open(split_name, 'r') as f:
            val_idx = [int(each_name.split(" ")[-1].split(".")[0]) for each_name in f.read().strip().split("\n")]

        val_idx.sort()

        n_depth = len(os.listdir(depth_base))
        n_tof = len(os.listdir(ToF_base))
        n_left = len(os.listdir(left_base))
        n_right = len(os.listdir(right_base))

        print(base, n_depth, n_tof, n_left, n_right)
        assert (n_depth == n_tof == n_left == n_right)

        depth_file_names = []
        depth_warped_file_names = []
        ToF_file_names = []
        left_file_names = []
        right_file_names = []

        idcs = [int(each_elem.split(".")[0]) for each_elem in os.listdir(depth_base) if each_elem.endswith("png")]

        if not base.endswith("20201027-181503"):
            print(base,"no offset")
            if split == "train":
                for i in idcs:
                    if i not in val_idx:
                        depth_file_names.append(depth_base + "{:010d}.png".format(i))
                        depth_warped_file_names.append(depth_warped_base + "{:010d}.png".format(i))
                        ToF_file_names.append(ToF_base + "{:010d}.png".format(i))
                        left_file_names.append(left_base + "{:010d}.png".format(i))
                        right_file_names.append(right_base + "{:010d}.png".format(i))
            else:
                for i in val_idx:
                    depth_file_names.append(depth_base + "{:010d}.png".format(i))
                    depth_warped_file_names.append(depth_warped_base + "{:010d}.png".format(i))
                    ToF_file_names.append(ToF_base + "{:010d}.png".format(i))
                    left_file_names.append(left_base + "{:010d}.png".format(i))
                    right_file_names.append(right_base + "{:010d}.png".format(i))
        else:
            print(base,"offset")
            if split == "train":
                for i in idcs:
                    if i not in val_idx and i+1 in idcs:
                        depth_file_names.append(depth_base + "{:010d}.png".format(i))
                        depth_warped_file_names.append(depth_warped_base + "{:010d}.png".format(i))
                        ToF_file_names.append(ToF_base + "{:010d}.png".format(i+1))
                        left_file_names.append(left_base + "{:010d}.png".format(i))
                        right_file_names.append(right_base + "{:010d}.png".format(i))

            else:
                for i in val_idx:
                    if i+1 in idcs:
                        depth_file_names.append(depth_base + "{:010d}.png".format(i))
                        depth_warped_file_names.append(depth_warped_base + "{:010d}.png".format(i))
                        ToF_file_names.append(ToF_base + "{:010d}.png".format(i+1))
                        left_file_names.append(left_base + "{:010d}.png".format(i))
                        right_file_names.append(right_base + "{:010d}.png".format(i))

        return depth_file_names, depth_warped_file_names, left_file_names, right_file_names, ToF_file_names

    def generate_mesh_grid(self):
        y_target,x_target = self.h,self.w
        y = np.arange(y_target)
        x = np.arange(x_target)
        x_grid,y_grid = np.meshgrid(x,y)

        homo = np.ones_like(x_grid)

        tof_mesh_grid = torch.from_numpy(np.stack([x_grid, y_grid, homo], -1))
        self.target_mesh_homo_flatten = tof_mesh_grid.view((-1, 3)).T

    def generate_intrinsic_conversion_factor(self):

        target_cam_can = torch.matmul(torch.linalg.inv(K_tof), self.target_mesh_homo_flatten.float())

        source_cam_pixel_pre = torch.matmul(K_rs, target_cam_can)

        source_cam_pixel = torch.stack([source_cam_pixel_pre[1, :], source_cam_pixel_pre[0, :]], 1)
        self.conversion_source = source_cam_pixel

    def intrinsic_transform(self,img,interpolation="nn"):

        x_target_grid = self.target_mesh_homo_flatten[0, :].view([480,640])
        y_target_grid = self.target_mesh_homo_flatten[1, :].view([480,640])
        target = torch.zeros([img.size()[0], 480, 640])

        if interpolation == "nn":
            conversion_source_in = torch.round(self.conversion_source).int()

            y_source = conversion_source_in[:, 0].view([480,640])
            x_source = conversion_source_in[:, 1].view([480,640])

            target[:,y_target_grid.long(),x_target_grid.long()] = img[:,y_source.long(),x_source.long()]

        elif interpolation == "bi":
            y = self.conversion_source[:, 0].view([480,640])
            x = self.conversion_source[:, 1].view([480,640])

            fy = torch.floor(y)
            fx = torch.floor(x)
            cy = fy+1
            cx = fx+1

            a00 = ((y - fy) * (x - fx)).view((1,480,640))
            a01 = ((y - fy) * (cx - x)).view((1,480,640))
            a10 = ((cy - y) * (x - fx)).view((1,480,640))
            a11 = ((cy - y) * (cx - x)).view((1,480,640))

            p00 = torch.zeros([3, 480, 640])
            p01 = torch.zeros([3, 480, 640])
            p10 = torch.zeros([3, 480, 640])
            p11 = torch.zeros([3, 480, 640])

            p00[:,y_target_grid.long(),x_target_grid.long()] = img[:,fy.long(),fx.long()]
            p01[:,y_target_grid.long(),x_target_grid.long()] = img[:,fy.long(),cx.long()]
            p10[:,y_target_grid.long(),x_target_grid.long()] = img[:,cy.long(),fx.long()]
            p11[:,y_target_grid.long(),x_target_grid.long()] = img[:,cy.long(),cx.long()]

            target = a11 * p00 + a00 * p11 + a10 * p01 + a01 * p10

        return target

def forward_warp(depth,source,target_shape,T_source_target,K_tar,K_sc):

    c, y_tg, x_tg = target_shape
    b, c, y_sc, x_sc = source.size()

    y = np.arange(y_sc)
    x = np.arange(x_sc)

    x_grid, y_grid = np.meshgrid(x, y)
    homo = np.ones_like(x_grid)

    sc_mesh_homo_pre = torch.from_numpy(np.stack([x_grid,y_grid,homo], -1))
    sc_mesh_homo_flatten = torch.tile(sc_mesh_homo_pre.view((-1, 3)).T.float(),[1,b])
    depth_flatten = depth.view([-1, 1]).T

    source_cam_can = torch.matmul(torch.linalg.inv(K_sc), sc_mesh_homo_flatten) * depth_flatten

    source_cam_can_h = torch.cat([source_cam_can, torch.ones((1, b * y_sc * x_sc ))], 0)

    source_cam_world = torch.matmul(torch.eye(4), source_cam_can_h)
    target_cam_world = torch.matmul(T_source_target, source_cam_world)
    target_cam_can_h = torch.matmul(torch.eye(4), target_cam_world)

    target_cam_can = target_cam_can_h[:3, :] / (target_cam_can_h[2:3, :] + 1e-10)

    target_cam_pixel_pre = torch.matmul(K_tar, target_cam_can)
    target_cam_pixel = torch.stack([target_cam_pixel_pre[1, :], target_cam_pixel_pre[0, :]], 1)

    target_cam_pixel = torch.round(target_cam_pixel).int()

    y_pre = target_cam_pixel[:, 0]
    x_pre = target_cam_pixel[:, 1]

    mask = ((x_pre < x_tg) * (x_pre >= 0) * (y_pre < y_tg) * (y_pre >= 0)).view((b,y_sc*x_sc))
    # depth_post = mask.reshape((n_batch, y_sc, x_sc)) * depth

    y = torch.clip(target_cam_pixel[:, 0], 0, y_tg - 1).long()
    x = torch.clip(target_cam_pixel[:, 1], 0, x_tg - 1).long()

    x_sc_grid = sc_mesh_homo_flatten[0, :].long()
    y_sc_grid = sc_mesh_homo_flatten[1, :].long()

    target = torch.zeros([b,c,y_tg,x_tg])
    for b_idx in range(b):
        start = b_idx*y_sc*x_sc
        end =  (b_idx+1)*y_sc*x_sc
        target[b_idx,:,y[start:end],x[start:end]] = source[b_idx,:,y_sc_grid[start:end],x_sc_grid[start:end]]*mask[b_idx]

    return target


# def _inverse_warp(depth, source, target_shape, target_meshgrid_flatten, T_target_source, K_tar, K_sc):
#
#     y = np.arange(img_size[0])
#     x = np.arange(img_size[1])
#
#     x_grid, y_grid = np.meshgrid(x, y)
#     homo = np.ones_like(x_grid)
#
#     tg_mesh_homo_pre = torch.from_numpy(np.stack([x_grid, y_grid, homo], -1))
#     # target_meshgrid_flatten =  torch.tile(tg_mesh_homo_pre.view((-1, 3)).T.float(), [1, 4]).cuda()
#
#     target_meshgrid_flatten = torch.stack([tg_mesh_homo_pre.view((-1, 3)).T.float() for i in range(8)], 0).cuda()
#
#     c, y_tg, x_tg = target_shape
#     b, c, y_sc, x_sc = source.size()
#
#     tg_mesh_homo_flatten = target_meshgrid_flatten
#
#     depth_flatten = depth.view([b, 1, -1])
#     target_cam_can = torch.matmul(torch.stack([torch.linalg.inv(K_tar) for _ in range(b)], 0),
#                                   tg_mesh_homo_flatten) * depth_flatten
#
#     target_cam_can_h = torch.cat([target_cam_can, torch.ones((b, 1, y_sc * x_sc)).cuda()], 1)
#
#     target_cam_world = torch.matmul(torch.eye(4).cuda(), target_cam_can_h)
#     source_cam_world = torch.matmul(T_target_source, target_cam_world)
#     source_cam_can_h = torch.matmul(torch.eye(4).cuda(), source_cam_world)
#
#     source_cam_can = source_cam_can_h[:, :3, :] / (source_cam_can_h[:, 2:3, :] + 1e-10)
#
#     source_cam_pixel_pre = torch.matmul(K_sc, source_cam_can)
#
#     source_cam_pixel = torch.stack([source_cam_pixel_pre[:, 1, :], source_cam_pixel_pre[:, 0, :]], 1)
#     source_cam_pixel = torch.round(source_cam_pixel).int()
#
#     y_pre = source_cam_pixel[:, 0]
#     x_pre = source_cam_pixel[:, 1]
#
#     mask = ((x_pre < x_sc) * (x_pre >= 0) * (y_pre < y_sc) * (y_pre >= 0)).view((b, 1, y_tg, x_tg)).cuda()
#
#     y = torch.clip(y_pre, 0, y_sc - 1).long().view([b, y_sc, x_sc])
#     x = torch.clip(x_pre, 0, x_sc - 1).long().view([b, y_sc, x_sc])
#     b_sc = torch.stack([i * torch.ones([y_sc, x_sc]) for i in range(b)], 0).long()
#
#     x_tg_grid = tg_mesh_homo_flatten[:, 0, :].long().view([b, y_tg, x_tg])
#     y_tg_grid = tg_mesh_homo_flatten[:, 1, :].long().view([b, y_tg, x_tg])
#     b_tg = torch.stack([i * torch.ones([y_tg, x_tg]) for i in range(b)], 0).long()
#
#     target = torch.zeros([b, 3, y_tg, x_tg]).cuda()
#     target[b_tg, :, y_tg_grid, x_tg_grid] = source[b_sc, :, y, x]
#
#     return target * mask


def inverse_warp_(depth,source,target_shape,target_meshgrid_flatten,T_target_source,K_tar,K_sc):

    c, y_tg, x_tg = target_shape
    b, c, y_sc, x_sc = source.size()

    y = np.arange(y_tg)
    x = np.arange(x_tg)

    tg_mesh_homo_flatten = target_meshgrid_flatten

    depth_flatten = depth.view([-1, 1]).T

    target_cam_can = torch.matmul(torch.linalg.inv(K_tar), tg_mesh_homo_flatten) * depth_flatten

    target_cam_can_h = torch.cat([target_cam_can, torch.ones((1,b * y_sc * x_sc )).cuda()], 0)

    target_cam_world = torch.matmul(torch.eye(4).cuda(), target_cam_can_h)
    source_cam_world = torch.matmul(T_target_source, target_cam_world)
    source_cam_can_h = torch.matmul(torch.eye(4).cuda(), source_cam_world)

    source_cam_can = source_cam_can_h[:3, :] / (source_cam_can_h[2:3, :] + 1e-10)

    source_cam_pixel_pre = torch.matmul(K_sc, source_cam_can)
    source_cam_pixel = torch.stack([source_cam_pixel_pre[1, :], source_cam_pixel_pre[0, :]], 1)

    source_cam_pixel = torch.round(source_cam_pixel).int()

    y_pre = source_cam_pixel[:, 0]
    x_pre = source_cam_pixel[:, 1]

    mask = ((x_pre < x_sc) * (x_pre >= 0) * (y_pre < y_sc) * (y_pre >= 0)).view((b,y_tg*x_tg)).cuda()
    # depth_post = mask.reshape((n_batch, y_sc, x_sc)) * depth

    y = torch.clip(source_cam_pixel[:, 0], 0, y_sc - 1).long()
    x = torch.clip(source_cam_pixel[:, 1], 0, x_sc - 1).long()

    x_tg_grid = tg_mesh_homo_flatten[0, :].long()
    y_tg_grid = tg_mesh_homo_flatten[1, :].long()

    target = torch.zeros([b,c,y_tg,x_tg]).cuda()
    for b_idx in range(b):
        start = b_idx*y_sc*x_sc
        end =  (b_idx+1)*y_sc*x_sc
        target[b_idx,:,y_tg_grid[start:end],x_tg_grid[start:end]] = source[b_idx,:,y[start:end],x[start:end]]*mask[b_idx]

    return target

def save_validation_images(rgb_depths,tof_depths,fnames,n_batch,val_dir,val_step,dmax=12,is_test=False):

    if not is_test:
        scene_names, file_names = fnames
    else:
        file_names = fnames

    if not is_test:
        val_step_dir = os.path.join(val_dir,str(int(val_step)))
        if not os.path.exists(val_step_dir):
            os.mkdir(val_step_dir)

    rgb_depths_np = rgb_depths.cpu().detach().numpy()
    tof_depths_np = tof_depths.cpu().detach().numpy()

    for each_idx in range(n_batch):

        file_name = file_names[each_idx]

        if not is_test:
            scene_name = scene_names[each_idx]
            val_scene = os.path.join(val_step_dir, scene_name)
        else:
            val_scene = val_dir

        if not os.path.exists(val_scene):
            os.mkdir(val_scene)

        rgb_dir = os.path.join(val_scene, "rgb")
        tof_dir = os.path.join(val_scene, "tof")

        if not os.path.exists(rgb_dir):
            os.mkdir(rgb_dir)
        if not os.path.exists(tof_dir):
            os.mkdir(tof_dir)

        rgb_fname = os.path.join(rgb_dir,file_name)
        tof_fname = os.path.join(tof_dir,file_name)

        each_rgb_depth = (rgb_depths_np[each_idx]/dmax*255).astype(np.uint8)
        each_tof_depth = (tof_depths_np[each_idx]/dmax*255).astype(np.uint8)

        cv2.imwrite(rgb_fname,each_rgb_depth.squeeze())
        cv2.imwrite(tof_fname,each_tof_depth.squeeze())

