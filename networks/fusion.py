import torch,os
from torch import nn
from .network_utils import *

class FusionBranch(nn.Module):

    def __init__(self,in_dim):
        super(FusionBranch, self).__init__()
        self.set_feature_fusion(in_dim)

    def set_feature_fusion(self,in_):

        out_ = int(in_/4)
        self.add_module("feature2_transform",nn.Sequential(
            conv3x3(in_,out_),
            torch.nn.ReLU(),
            conv1x1(out_,in_),
            # torch.nn.ReLU(),
            # conv1x1(out_,in_),
        ))

        self.add_module("calculate_weight",nn.Sequential(
            conv3x3(2*in_,out_),
            torch.nn.ReLU(),
            conv1x1(out_,in_),
            torch.nn.Sigmoid(),
        ))

    def forward(self,feature1,feature2):

        feature2_transformed = self.feature2_transform(feature2)
        weight_input = torch.cat([feature1,feature2_transformed],1)
        weight = self.calculate_weight(weight_input)
        fused_out = weight * feature2_transformed + feature1

        return fused_out