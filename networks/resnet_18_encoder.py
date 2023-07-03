import torch,os
from torch import nn
from torchvision import datasets,models
from .resnet18_no_bn import resnet18_no_bn
from .network_utils import *

class ResNet18Encoder(nn.Module):
    def __init__(self,mode):
        super(ResNet18Encoder, self).__init__()

        if mode == "rgb":
            res_en = models.resnet18(pretrained=True)
        elif mode == "tof":
            res_en = resnet18_no_bn()
        else:
            raise

        self.set_encoder_layer(res_en,mode)

    def set_encoder_layer(self,net,mode):
        ## rgb_encoder
        if mode == "rgb":
            encoder_1_2 = nn.Sequential(
                net.conv1,
                net.bn1,
                net.relu
            )
        else:
            print("no-bn")
            encoder_1_2 = nn.Sequential(
                net.conv1,
                net.relu,
            )

        self.add_module("encoder_1_2", encoder_1_2)
        encoder_1_4 = nn.Sequential(
                                        net.maxpool,
                                        net.layer1
                                    )
        self.add_module("encoder_1_4",  encoder_1_4)
        self.add_module("encoder_1_8",  net.layer2)
        self.add_module("encoder_1_16", net.layer3)
        self.add_module("encoder_1_32", net.layer4)

    def forward(self,input):

        en_out_1_2  = self.encoder_1_2(input)
        en_out_1_4  = self.encoder_1_4(en_out_1_2)
        en_out_1_8  = self.encoder_1_8(en_out_1_4)
        en_out_1_16 = self.encoder_1_16(en_out_1_8)
        en_out_1_32 = self.encoder_1_32(en_out_1_16)

        return [en_out_1_32,en_out_1_16,en_out_1_8,en_out_1_4,en_out_1_2]
