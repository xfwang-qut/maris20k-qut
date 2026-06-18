import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet
from .FreqDynConv import FCDConv
from .convs import SELayer, ECALayer, CBAM, ODConv2d, FDConv,SC

@MODELS.register_module()
class FDResNet(ResNet):
    def __init__(self, **kwargs):
        super(FDResNet, self).__init__(strides=(1, 1, 1, 1), **kwargs)
        self.FreqDyn = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            FCDConv(in_channels=256, dct_h=224, dct_w=224),
            FCDConv(in_channels=512, dct_h=112, dct_w=112),
            FCDConv(in_channels=1024, dct_h=56, dct_w=56)
        ])


    def forward(self, x, denoise=True):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = self.FreqDyn[i](x)
            # print(x.shape)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils import checkpoint as cp
#
# from mmcv.cnn import build_conv_layer
# from mmdet.registry import MODELS
# from mmdet.models.backbones import ResNet
# from mmdet.models.backbones.resnet import Bottleneck
# from .FreqDynConv import FreqDynConv
#
#
# class SmoothPriorBottleneck(Bottleneck):
#     """
#     简洁稳健版：conv2(out) + sigmoid(scale)*prior_proj(prior(out))
#     """
#     def __init__(self, train_first_stage=True, prior_scale_logit=-2.0, **kwargs):
#         super().__init__(**kwargs)
#         self.train_first_stage = train_first_stage
#
#         # stage1（planes=64）可选禁用 prior，保持与预训练一致并避免unused参数
#         self.use_prior = not (self.planes == 64 and not self.train_first_stage)
#
#         if self.use_prior:
#             self.prior_scale_logit = nn.Parameter(torch.tensor(prior_scale_logit, dtype=torch.float32))
#
#             conv_cfg = kwargs.get("conv_cfg", None)
#             planes = kwargs["planes"]
#             dilation = kwargs.get("dilation", 1)
#
#             self.prior = FreqDynConv(
#                 in_channels=planes,
#                 stride=self.conv2_stride,
#             )
#
#             self.prior_proj = build_conv_layer(
#                 conv_cfg,
#                 planes,
#                 planes,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#                 dilation=dilation,
#                 bias=False
#             )
#
#     def forward(self, x):
#         if not self.use_prior:
#             return super().forward(x)
#
#         def _inner_forward(x):
#             identity = x
#
#             out = self.conv1(x)
#             out = self.norm1(out)
#             out = self.relu(out)
#
#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv1_plugin_names)
#
#             direct_out = self.conv2(out)
#
#             scale = torch.sigmoid(self.prior_scale_logit)  # (0,1)
#             prior_out = self.prior_proj(self.prior(out))
#             out = direct_out + scale * prior_out
#
#             out = self.norm2(out)
#             out = self.relu(out)
#
#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv2_plugin_names)
#
#             out = self.conv3(out)
#             out = self.norm3(out)
#
#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv3_plugin_names)
#
#             if self.downsample is not None:
#                 identity = self.downsample(x)
#
#             out = out + identity
#             return out
#
#         if self.with_cp and x.requires_grad:
#             out = cp.checkpoint(_inner_forward, x)
#         else:
#             out = _inner_forward(x)
#
#         return self.relu(out)
#
#
# @MODELS.register_module()
# class FDResNet(ResNet):
#     arch_settings = {
#         50:  (SmoothPriorBottleneck, (3, 4, 6, 3)),
#         101: (SmoothPriorBottleneck, (3, 4, 23, 3)),
#         152: (SmoothPriorBottleneck, (3, 8, 36, 3)),
#     }
#
#     def __init__(self, **kwargs):
#         super().__init__(strides=(1, 1, 1, 1), **kwargs)
#         self.FreqDyn = nn.ModuleList([
#             nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             FreqDynConv(in_channels=256),
#             FreqDynConv(in_channels=512),
#             FreqDynConv(in_channels=1024),
#         ])
#
#     def forward(self, x, denoise=True):
#         if self.deep_stem:
#             x = self.stem(x)
#         else:
#             x = self.conv1(x)
#             x = self.norm1(x)
#             x = self.relu(x)
#
#         outs = []
#         for i, layer_name in enumerate(self.res_layers):
#             res_layer = getattr(self, layer_name)
#             x = self.FreqDyn[i](x)
#             x = res_layer(x)
#             if i in self.out_indices:
#                 outs.append(x)
#
#         return tuple(outs)
