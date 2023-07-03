import torch,os
from torch import nn
from .network_utils import *
from .fusion import *

class Decoder(nn.Module):
    def __init__(self,a,b,f_in,f_out,res,fusion,mode="nearest"):
        super(Decoder, self).__init__()

        self.a = a
        self.b = b
        self.is_fusion = fusion
        self.res = res

        self.set_decoder_layers(mode,f_in,f_out,res)

        if fusion:
            self.add_module("fusion",FusionBranch(fusion))

    def set_decoder_layers(self,mode,in_,out_,res):
        if mode == "nearest":
            align_corners = None
        else:
            align_corners = True

        if res != 1:
            decoder_1 = nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners),
                torch.nn.ELU(),
                )
            decoder_2 = nn.Sequential(
                conv3x3(in_+out_, out_),
                torch.nn.ELU()
            )
            self.add_module("decoder_1", decoder_1)
            self.add_module("decoder_2", decoder_2)

            if res != 16:
                depth = torch.nn.Sequential(
                    conv3x3(out_, 1),
                    torch.nn.Sigmoid()
                )
                self.add_module("depth", depth)

        else:
            decoder_1 = nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners),
                torch.nn.ELU(),
                conv3x3(in_, out_),
                torch.nn.ELU()
            )
            self.add_module("decoder_1", decoder_1)

            depth = torch.nn.Sequential(
                conv3x3(out_, 1),
                torch.nn.Sigmoid()
            )
            self.add_module("depth", depth)

    def forward(self,d_feature,e_feature,feature2=False):

        # timing of fusion? before concat? after concat? right before depth?
        if self.res == 16:
            # 32 -> up -> 16
            if not self.is_fusion:
                d_fused = d_feature
            else:
                d_fused = self.fusion(d_feature,feature2)

            d_up  = self.decoder_1(d_fused) # upsample
            d_concat = torch.cat([d_up,e_feature],1)
            d_out = self.decoder_2(d_concat)
            return d_out

        elif self.res == 1:
            d_out = self.decoder_1(d_feature)
            depth = 1/(self.a*self.depth(d_out) + self.b)
            return  depth

        else:
            # res/2 -> up -> res
            d_up  = self.decoder_1(d_feature) # upsample
            d_concat = torch.cat([d_up,e_feature],1)
            d_out = self.decoder_2(d_concat)

            if not self.is_fusion:
                d_out = d_out
            else:
                d_out = self.fusion(d_out,feature2)

            depth = 1/(self.a*self.depth(d_out) + self.b)

            return d_out, depth