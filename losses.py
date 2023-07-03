import torch,os
from torch import nn
from utilities import *
import kornia
import torch.nn.functional as F

class InverseWarp():

    def __init__(self, img_size_target, img_size_source, K_tar,K_sc, n_batch):

        self.y_tg, self.x_tg, self.c = img_size_target
        self.y_sc, self.x_sc, self.c = img_size_source

        self.b = n_batch

        y = np.arange(img_size_target[0])
        x = np.arange(img_size_target[1])

        x_grid, y_grid = np.meshgrid(x, y)
        homo = np.ones_like(x_grid)

        target_mesh_homo_pre = torch.from_numpy(np.stack([x_grid, y_grid, homo], -1))
        self.tg_mesh_homo_flatten = torch.stack([target_mesh_homo_pre.view((-1, 3)).T.float() for i in range(self.b)], 0).cuda()


        self.b_tg = torch.stack([i * torch.ones([self.y_tg, self.x_tg]) for i in range(self.b)], 0).long()
        self.b_sc = torch.stack([i * torch.ones([self.y_sc, self.x_sc]) for i in range(self.b)], 0).long()

        self.K_tar = K_tar
        self.K_sc = K_sc

        self.max_pool = nn.MaxPool2d((5,5),stride=(1,1),padding=2)

    def stereo_self_supervised_loss(self, depths, target, source, T_target_source, inter=True):

        warped = []
        loss_pre = []
        loss = 0

        for idx,each_depth in enumerate(depths):
            each_warped_right = self.inverse_warp(each_depth, source, T_target_source, inter)
            mask = -self.max_pool(-(each_warped_right  > 0).type('torch.DoubleTensor')).cuda()
            each_loss_pre = torch.sum(torch.abs(target - each_warped_right),dim=1,keepdim=True) * mask
            # if idx <= 1:
            loss += torch.sum(each_loss_pre)
            loss_pre.append(each_loss_pre)
            warped.append(each_warped_right)

        return warped, loss_pre, loss

    def tof_rgb_self_supervised_loss(self, depths, left, right, T_tof_to_left, T_tof_to_right, inter=True):

        warped_from_left = []
        warped_from_right = []

        loss_pre = []
        loss = 0

        for idx,each_depth in enumerate(depths):

            each_warped_from_left  = self.inverse_warp(each_depth,  left, T_tof_to_left, inter)
            each_warped_from_right = self.inverse_warp(each_depth, right, T_tof_to_right, inter)

            warped_from_left.append(each_warped_from_left)
            warped_from_left.append(each_warped_from_right)

            mask = (-self.max_pool(-(each_warped_from_left  > 0).type('torch.DoubleTensor')).cuda()) * \
                   (-self.max_pool(-(each_warped_from_right > 0).type('torch.DoubleTensor')).cuda())

            each_loss_pre = torch.sum(torch.abs(each_warped_from_left - each_warped_from_right),dim=1,keepdim=True) * mask

            loss += torch.sum(each_loss_pre)
            loss_pre.append(each_loss_pre)

        return warped_from_left, warped_from_right, loss_pre, loss, mask

    def inverse_warp(self, depth, source, T_target_source, inter=True):

        depth_flatten = depth.view([self.b, 1, -1])
        target_cam_can = torch.matmul(torch.stack([torch.linalg.inv(self.K_tar) for _ in range(self.b)], 0),
                                      self.tg_mesh_homo_flatten) * depth_flatten

        target_cam_can_h = torch.cat([target_cam_can, torch.ones((self.b, 1, self.y_tg * self.x_tg)).cuda()], 1)

        target_cam_world = torch.matmul(torch.eye(4).cuda(), target_cam_can_h)
        source_cam_world = torch.matmul(T_target_source, target_cam_world)
        source_cam_can_h = torch.matmul(torch.eye(4).cuda(), source_cam_world)

        source_cam_can = source_cam_can_h[:, :3, :] / (source_cam_can_h[:, 2:3, :] + 1e-10)

        source_cam_pixel_pre = torch.matmul(self.K_sc, source_cam_can)

        source_cam_pixel = torch.stack([source_cam_pixel_pre[:, 1, :], source_cam_pixel_pre[:, 0, :]], 1)

        if not inter:
            source_cam_pixel = torch.round(source_cam_pixel).int()

            y_pre = source_cam_pixel[:, 0]
            x_pre = source_cam_pixel[:, 1]

            mask = ((x_pre < self.x_sc) * (x_pre >= 0) * (y_pre < self.y_sc) * (y_pre >= 0)).view((self.b, 1, self.y_sc, self.x_sc)).cuda()

            y = torch.clip(y_pre, 0, self.y_sc - 1).long().view([self.b, self.y_sc, self.x_sc])
            x = torch.clip(x_pre, 0, self.x_sc - 1).long().view([self.b, self.y_sc, self.x_sc])

            x_tg_grid = self.tg_mesh_homo_flatten[:, 0, :].long().view([self.b, self.y_tg, self.x_tg])
            y_tg_grid = self.tg_mesh_homo_flatten[:, 1, :].long().view([self.b, self.y_tg, self.x_tg])

            target = torch.zeros([self.b, self.c, self.y_tg, self.x_tg]).cuda()
            target[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_sc, :, y, x]

        else:
        ###
            y_pre = source_cam_pixel[:, 0]
            x_pre = source_cam_pixel[:, 1]
            mask = ((x_pre < self.x_sc-1) * (x_pre >= 0) * (y_pre < self.y_sc-1) * (y_pre >= 0)).view(
                (self.b, 1, self.y_tg, self.x_tg)).cuda()

            y = torch.clip(y_pre, 0, self.y_sc - 2).view([self.b, self.y_sc, self.x_sc])
            x = torch.clip(x_pre, 0, self.x_sc - 2).view([self.b, self.y_sc, self.x_sc])

            fy = torch.floor(y).long()
            fx = torch.floor(x).long()
            cy = fy + 1
            cx = fx + 1

            a00 = ((y - fy) * (x - fx)).view((self.b, 1, self.y_sc, self.x_sc))
            a01 = ((y - fy) * (cx - x)).view((self.b, 1, self.y_sc, self.x_sc))
            a10 = ((cy - y) * (x - fx)).view((self.b, 1, self.y_sc, self.x_sc))
            a11 = ((cy - y) * (cx - x)).view((self.b, 1, self.y_sc, self.x_sc))

            p00 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p00.clone()
            p01 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p01.clone()
            p10 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p10.clone()
            p11 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p11.clone()

            x_tg_grid = self.tg_mesh_homo_flatten[:, 0, :].long().view([self.b, self.y_tg, self.x_tg])
            y_tg_grid = self.tg_mesh_homo_flatten[:, 1, :].long().view([self.b, self.y_tg, self.x_tg])

            p00[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, fy, fx]
            p01[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, fy, cx]
            p10[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, cy, fx]
            p11[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, cy, cx]

            target = a11 * p00 + a00 * p11 + a10 * p01 + a01 * p10

        return target * mask

class UpscaleImages(nn.Module):

    def __init__(self,size):
        super(UpscaleImages, self).__init__()

        self.scale_8_to_1 = torch.nn.Upsample(size, mode="nearest")
        self.scale_4_to_1 = torch.nn.Upsample(size, mode="nearest")
        self.scale_2_to_1 = torch.nn.Upsample(size, mode="nearest")

    def forward(self,preds):

        pred_1_8, pred_1_4, pred_1_2, pred_1_1 = preds
        return [self.scale_8_to_1(pred_1_8),self.scale_4_to_1(pred_1_4),self.scale_2_to_1(pred_1_2),pred_1_1]

class MultiscaleL1Loss(nn.Module):

    def __init__(self,dmin,dmax):
        super(MultiscaleL1Loss, self).__init__()

        self.dmin = dmin
        self.dmax = dmax

    def l1_loss(self,gt,pred):

        loss_pre = torch.abs(torch.clip(gt,self.dmin,self.dmax)-pred)*(gt>self.dmin)
        loss = torch.sum(loss_pre)

        return loss_pre,loss

    def forward(self,gt,up_preds):

        pred_1_8,pred_1_4,pred_1_2,pred_1_1 = up_preds

        loss_1_8_pre, loss_1_8 = self.l1_loss(gt,pred_1_8)
        loss_1_4_pre, loss_1_4 = self.l1_loss(gt,pred_1_4)
        loss_1_2_pre, loss_1_2 = self.l1_loss(gt,pred_1_2)
        loss_1_1_pre, loss_1_1 = self.l1_loss(gt,pred_1_1)

        loss_pre = [loss_1_8_pre,loss_1_4_pre,loss_1_2_pre,loss_1_1_pre]
        loss = loss_1_8 + loss_1_4 + loss_1_2 + loss_1_1

        return [loss_pre,loss]

def edge_aware_loss(rgb,pred,mask=None,factor=1):

    rgb_grad = kornia.spatial_gradient(rgb,normalized=False)
    rgb_edge = torch.sum(torch.abs(rgb_grad),dim=[1])

    pred_grad = kornia.spatial_gradient(pred,normalized=False)
    pred_edge = torch.sum(torch.abs(pred_grad),dim=[1])

    loss_pre = (pred_edge[:,0]*torch.exp(-factor*rgb_edge[:,0]) + pred_edge[:,1]*torch.exp(-factor*rgb_edge[:,1]))#*mask
    loss = torch.sum(loss_pre)

    return [loss_pre,loss]

def edge_aware_loss2(rgb,pred,mask=None,facator=1):

    rgb_grad = calculate_edge(rgb)
    rgb_edge = torch.sum(torch.abs(rgb_grad),dim=1)
    pred_grad = calculate_edge(pred)
    pred_edge = torch.sum(torch.abs(pred_grad),dim=1)

    loss_pre = (pred_edge[:,0]*torch.exp(-rgb_edge[:,0]) + pred_edge[:,1]*torch.exp(-rgb_edge[:,1]))#*mask
    loss = torch.sum(loss_pre)

    return [loss_pre,loss]

def calculate_edge(img):

    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    a = a.view((1, 1, 3, 3)).cuda()

    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    b = b.view((1, 1, 3, 3)).cuda()

    img_grad = []

    for c in range(img.size()[1]):

        G_x = F.conv2d(img[:,c:c+1], a, padding=1)
        G_y = F.conv2d(img[:,c:c+1], b, padding=1)
        grad = torch.cat([G_x,G_y],1) # b,2,w,h

        img_grad.append(grad)

    img_edge = torch.stack(img_grad,1)
    return img_edge