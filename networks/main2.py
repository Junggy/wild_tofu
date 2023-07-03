import torch,os
from torch import nn
from .resnet_18_encoder import ResNet18Encoder
from .decoder2 import Decoder
from .network_utils import *
import numpy as np

K = torch.from_numpy(np.array([[558.1649169921875, 0.0, 320.84777429724636],
                               [0.0, 571.637451171875, 252.1800136421225],
                               [0.0, 0.0, 1.0]])).float().cuda()

K_1_2 = torch.from_numpy(np.array([[558.1649169921875 / 2, 0.0, 320.84777429724636 / 2],
                                   [0.0, 571.637451171875 / 2, 252.1800136421225 / 2],
                                   [0.0, 0.0, 1.0]])).float().cuda()

K_1_4 = torch.from_numpy(np.array([[558.1649169921875 / 4, 0.0, 320.84777429724636 / 4],
                                   [0.0, 571.637451171875 / 4, 252.1800136421225 / 4],
                                   [0.0, 0.0, 1.0]])).float().cuda()

K_1_8 = torch.from_numpy(np.array([[558.1649169921875 / 8, 0.0, 320.84777429724636 / 8],
                                   [0.0, 571.637451171875 / 8, 252.1800136421225 / 8],
                                   [0.0, 0.0, 1.0]])).float().cuda()

K_1_16 = torch.from_numpy(np.array([[558.1649169921875 / 16, 0.0, 320.84777429724636 / 16],
                                   [0.0, 571.637451171875 / 16, 252.1800136421225 / 16],
                                   [0.0, 0.0, 1.0]])).float().cuda()

ToF_to_RS = torch.from_numpy( np.array([[0.9995575011766835, 0.007698921328204006, 0.02873201092521153, 0.04250089873790845],
                                        [-0.006708546718660541, 0.9993853535885749, -0.03440799955769654, 0.06625787988547474],
                                        [-0.028979255379458894, 0.03420002402076673, 0.998994775318986, -0.002945568589809818],
                                        [0, 0, 0, 1]])).float().cuda()

RS_to_ToF = torch.linalg.inv(ToF_to_RS)

class ToFNetFusion(nn.Module):
    def __init__(self,dmin,dmax,setup,fusion=[True,False,False,False]):
        super(ToFNetFusion, self).__init__()
        self.b = 1 / dmax
        self.a = (1 - self.b * dmin) / (dmin)
        print("a :",self.a, "b :", self.b)

        self.fusion = fusion
        module = {"tof_encoder": ResNet18Encoder("tof"),
                  "tof_decoder1_16": Decoder(self.a,self.b, 512, 256, 16, fusion[0]),
                  "tof_decoder1_8": Decoder(self.a, self.b, 256, 128, 8, False),
                  "tof_decoder1_4": Decoder(self.a, self.b, 128, 64, 4, False),
                  "tof_decoder1_2": Decoder(self.a, self.b, 64, 64, 2, False),
                  "tof_decoder1_1": Decoder(self.a, self.b, 64, 16, 1, False),

                  "rgb_encoder": ResNet18Encoder("rgb"),
                  "rgb_decoder1_16": Decoder(self.a,self.b, 512, 256, 16, fusion[0]),
                  "rgb_decoder1_8": Decoder(self.a, self.b, 256, 128, 8, fusion[1]),
                  "rgb_decoder1_4": Decoder(self.a, self.b, 128, 64, 4, fusion[2]),
                  "rgb_decoder1_2": Decoder(self.a, self.b, 64, 64, 2, fusion[3]),
                  "rgb_decoder1_1": Decoder(self.a, self.b, 64, 16, 1, False),
                  }
        try:
            batch = setup['train_batch']
        except:
            batch = setup["val_batch"]

        self.forward_warp_1_8 = ForwardWarp([int(480 / 8), int(640 / 8), 128], [int(480 / 8), int(640 / 8), 128], K_1_8, K_1_8,batch)
        self.forward_warp_1_4 =  ForwardWarp([int(480 / 4), int(640 / 4), 64], [int(480 / 4), int(640 / 4), 64], K_1_4, K_1_4, batch)
        self.forward_warp_1_2 =  ForwardWarp([int(480 / 2), int(640 / 2), 64], [int(480 / 2), int(640 / 2), 64], K_1_2, K_1_2, batch)

        for key,val in module.items():
            self.add_module(key,val)

    def forward(self,input):

        rgb =  input[:,:3,:,:]
        corr = input[:,3:,:,:]

        rgb_encoded = self.rgb_encoder(rgb)
        tof_encoded = self.tof_encoder(corr)

        tof_decoded_16 = self.tof_decoder1_16(tof_encoded[0],tof_encoded[1], rgb_encoded[0] if self.fusion[0] else False)
        rgb_decoded_16 = self.rgb_decoder1_16(rgb_encoded[0],rgb_encoded[1], tof_encoded[0] if self.fusion[0] else False)

        # tof_decoded_8,tof_depth_8 = self.tof_decoder1_8(tof_decoded_16,tof_encoded[2], rgb_encoded[1] if self.fusion[1] else False)
        tof_decoded_8,tof_depth_8 = self.tof_decoder1_8(tof_decoded_16,tof_encoded[2], False)
        rgb_decoded_8,rgb_depth_8 = self.rgb_decoder1_8(rgb_decoded_16,rgb_encoded[2], self.forward_warp_1_8.fw(tof_depth_8,tof_decoded_8,ToF_to_RS) if self.fusion[1] else False)

        # tof_decoded_4,tof_depth_4 = self.tof_decoder1_4(tof_decoded_8,tof_encoded[3], rgb_encoded[2] if self.fusion[2] else False)
        tof_decoded_4,tof_depth_4 = self.tof_decoder1_4(tof_decoded_8, tof_encoded[3], False)
        rgb_decoded_4,rgb_depth_4 = self.rgb_decoder1_4(rgb_decoded_8,rgb_encoded[3], self.forward_warp_1_4.fw(tof_depth_4,tof_decoded_4,ToF_to_RS) if self.fusion[2] else False)

        # tof_decoded_2,tof_depth_2 = self.tof_decoder1_2(tof_decoded_4,tof_encoded[4], rgb_encoded[3] if self.fusion[3] else False)
        tof_decoded_2,tof_depth_2 = self.tof_decoder1_2(tof_decoded_4, tof_encoded[4], False)
        rgb_decoded_2,rgb_depth_2 = self.rgb_decoder1_2(rgb_decoded_4,rgb_encoded[4], self.forward_warp_1_2.fw(tof_depth_2,tof_decoded_2,ToF_to_RS) if self.fusion[3] else False)

        tof_depth_1 = self.tof_decoder1_1(tof_decoded_2, False)
        rgb_depth_1 = self.rgb_decoder1_1(rgb_decoded_2, False)
        
        return [rgb_depth_8,rgb_depth_4,rgb_depth_2,rgb_depth_1], [tof_depth_8,tof_depth_4,tof_depth_2,tof_depth_1]

class ForwardWarp():

    def __init__(self, img_size_target, img_size_source, K_tar, K_sc, n_batch):

        self.y_tg, self.x_tg, self.c = img_size_target
        self.y_sc, self.x_sc, self.c = img_size_source

        self.b = n_batch

        y = np.arange(img_size_source[0])
        x = np.arange(img_size_source[1])

        x_grid, y_grid = np.meshgrid(x, y)
        homo = np.ones_like(x_grid)

        source_mesh_homo_pre = torch.from_numpy(np.stack([x_grid, y_grid, homo], -1))
        self.sc_mesh_homo_flatten = torch.stack([source_mesh_homo_pre.view((-1, 3)).T.float() for i in range(self.b)], 0).cuda()

        self.b_tg = torch.stack([i * torch.ones([self.y_tg, self.x_tg]) for i in range(self.b)], 0).long()
        self.b_sc = torch.stack([i * torch.ones([self.y_sc, self.x_sc]) for i in range(self.b)], 0).long()

        self.K_tar = K_tar
        self.K_sc = K_sc

        self.max_pool = nn.MaxPool2d((5,5),stride=(1,1),padding=2)

    def fw(self, depth, source, T_source_target, inter=False):

        depth_flatten = depth.view([self.b, 1, -1])
        source_cam_can = torch.matmul(torch.stack([torch.linalg.inv(self.K_sc) for _ in range(self.b)], 0),
                                      self.sc_mesh_homo_flatten) * depth_flatten

        source_cam_can_h = torch.cat([source_cam_can, torch.ones((self.b, 1, self.y_sc * self.x_sc)).cuda()], 1)

        source_cam_world = torch.matmul(torch.eye(4).cuda(), source_cam_can_h)
        target_cam_world = torch.matmul(T_source_target, source_cam_world)
        target_cam_can_h = torch.matmul(torch.eye(4).cuda(), target_cam_world)

        target_cam_can = target_cam_can_h[:, :3, :] / (target_cam_can_h[:, 2:3, :] + 1e-10)

        target_cam_pixel_pre = torch.matmul(self.K_sc, target_cam_can)

        target_cam_pixel = torch.stack([target_cam_pixel_pre[:, 1, :], target_cam_pixel_pre[:, 0, :]], 1)

        if not inter:
            target_cam_pixel = torch.round(target_cam_pixel).int()

            y_pre = target_cam_pixel[:, 0]
            x_pre = target_cam_pixel[:, 1]

            mask = ((x_pre < self.x_tg) * (x_pre >= 0) * (y_pre < self.y_tg) * (y_pre >= 0)).view((self.b, 1, self.y_tg, self.x_tg)).cuda()

            y = torch.clip(y_pre, 0, self.y_tg - 1).long().view([self.b, self.y_tg, self.x_tg])
            x = torch.clip(x_pre, 0, self.x_tg - 1).long().view([self.b, self.y_tg, self.x_tg])

            x_sc_grid = self.sc_mesh_homo_flatten[:, 0, :].long().view([self.b, self.y_sc, self.x_sc])
            y_sc_grid = self.sc_mesh_homo_flatten[:, 1, :].long().view([self.b, self.y_sc, self.x_sc])

            target = torch.zeros([self.b, self.c, self.y_tg, self.x_tg]).cuda()
            target[self.b_tg, :, y, x] = source[self.b_sc, :, y_sc_grid, x_sc_grid]

        else:
        ###
            pass
            # y_pre = source_cam_pixel[:, 0]
            # x_pre = source_cam_pixel[:, 1]
            # mask = ((x_pre < self.x_sc-1) * (x_pre >= 0) * (y_pre < self.y_sc-1) * (y_pre >= 0)).view(
            #     (self.b, 1, self.y_tg, self.x_tg)).cuda()
            #
            # y = torch.clip(y_pre, 0, self.y_sc - 2).view([self.b, self.y_sc, self.x_sc])
            # x = torch.clip(x_pre, 0, self.x_sc - 2).view([self.b, self.y_sc, self.x_sc])
            #
            # fy = torch.floor(y).long()
            # fx = torch.floor(x).long()
            # cy = fy + 1
            # cx = fx + 1
            #
            # a00 = ((y - fy) * (x - fx)).view((self.b, 1, self.y_sc, self.x_sc))
            # a01 = ((y - fy) * (cx - x)).view((self.b, 1, self.y_sc, self.x_sc))
            # a10 = ((cy - y) * (x - fx)).view((self.b, 1, self.y_sc, self.x_sc))
            # a11 = ((cy - y) * (cx - x)).view((self.b, 1, self.y_sc, self.x_sc))
            #
            # p00 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p00.clone()
            # p01 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p01.clone()
            # p10 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p10.clone()
            # p11 = torch.zeros([self.b, self.c, self.y_sc, self.x_sc]).cuda()#self.p11.clone()
            #
            # x_tg_grid = self.tg_mesh_homo_flatten[:, 0, :].long().view([self.b, self.y_tg, self.x_tg])
            # y_tg_grid = self.tg_mesh_homo_flatten[:, 1, :].long().view([self.b, self.y_tg, self.x_tg])
            #
            # p00[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, fy, fx]
            # p01[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, fy, cx]
            # p10[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, cy, fx]
            # p11[self.b_tg, :, y_tg_grid, x_tg_grid] = source[self.b_tg, :, cy, cx]
            #
            # target = a11 * p00 + a00 * p11 + a10 * p01 + a01 * p10

        return target * mask