import math
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
from einops import rearrange, einsum

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class DCT(nn.Module):

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(DCT, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = (
                            self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y))

        return dct_filter


# class FreqLayer(nn.Module):
#     """
#     Fixed-size DCT frequency modulation layer (FcaNet-style):
#     - Build DCT basis once at (dct_h, dct_w)
#     - For any input feature map, adaptive_avg_pool2d -> (dct_h, dct_w)
#     - Avoid rebuilding DCT filters when (h,w) changes
#     """
#     def __init__(self, channel, freq_sel_method='top16', dct_h=14, dct_w=14):
#         super().__init__()
#         self.channel = channel
#         self.freq_sel_method = freq_sel_method
#         self.dct_h = dct_h
#         self.dct_w = dct_w

#         # build fixed DCT layer once (no rebuild in forward)
#         mapper_x, mapper_y = get_freq_indices(self.freq_sel_method)
#         # map to (dct_h, dct_w) frequency space (same trick as his code)
#         mapper_x = [tx * (self.dct_h // 7) for tx in mapper_x]
#         mapper_y = [ty * (self.dct_w // 7) for ty in mapper_y]
#         self.dct_layer = DCT(self.dct_h, self.dct_w, mapper_x, mapper_y, self.channel)

#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels=channel, out_channels=channel // 4, kernel_size=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=channel // 4, out_channels=channel, kernel_size=1, bias=True),
#             # 如果你想当作门控用，可以加 Sigmoid：nn.Sigmoid()
#         )

#     def forward(self, x):
#         n, c, h, w = x.shape

#         # fixed pooling to avoid variable-size DCT rebuild
#         if (h != self.dct_h) or (w != self.dct_w):
#             x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
#         else:
#             x_pooled = x

#         # DCT projection -> [B, C] then -> [B,C,1,1]
#         y = self.dct_layer(x_pooled).view(n, c, 1, 1)
#         y = self.fc(y)
#         return y
    

class FreqLayer(nn.Module):
    def __init__(self, channel, freq_sel_method='top16'):
        super(FreqLayer, self).__init__()
        self.channel = channel
        self.freq_sel_method = freq_sel_method

        self.dct_h = None
        self.dct_w = None
        self.dct_layer = None

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel // 4, out_channels=channel, kernel_size=1, bias=True),
            # nn.Sigmoid()
        )

    def _build_dct_layer(self, h, w, device, dtype):
        self.dct_h = h
        self.dct_w = w

        mapper_x, mapper_y = get_freq_indices(self.freq_sel_method)
        mapper_x = [temp_x * (h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (w // 7) for temp_y in mapper_y]

        self.dct_layer = DCT(h, w, mapper_x, mapper_y, self.channel).to(device=device, dtype=dtype)

    def forward(self, x):
        n, c, h, w = x.shape
        if (self.dct_layer is None) or (self.dct_h != h) or (self.dct_w != w):
            self._build_dct_layer(h, w, x.device, x.dtype)

        x_pooled = x
        y = self.dct_layer(x_pooled).view(n, c, 1, 1)
        y = self.fc(y)
        return y


class FCDConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, groups=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups if groups is not None else in_channels//2
        # self.dct_h = dct_h
        # self.dct_w = dct_w

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.groups * kernel_size ** 2,
                      stride=stride,
                      kernel_size=kernel_size,
                      bias=False,
                      padding=self.pad,
                      groups=self.groups),
            nn.BatchNorm2d(self.groups * kernel_size ** 2),
        )
        # self.freq_mod_net = FreqLayer(in_channels, freq_sel_method='top16', dct_h=self.dct_h, dct_w=self.dct_w)

        self.freq_mod_net = FreqLayer(in_channels, freq_sel_method='top16')

        self.num_heads = self.groups
        assert in_channels % self.num_heads == 0, "in_channels 必须能被 groups 整除"
        head_dim = in_channels // self.num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.k_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.attn_proj = nn.Conv2d(49, kernel_size ** 2, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1

        w_mod = self.freq_mod_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
        weight = self.weight(x)

        weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow)

        weight = weight.expand(b, self.groups, c // self.groups, self.kernel_size ** 2, oh, ow)

        if self.stride != 1:
            x_qk = F.interpolate(x, size=(oh, ow), mode='bilinear', align_corners=False)
        else:
            x_qk = x

        Q = self.q_proj(x_qk) * self.scale
        K = self.k_proj(x_qk)

        Q = rearrange(Q, 'b (g c) h w -> b g c (h w)', g=self.num_heads)  # N = oh*ow
        K = rearrange(K, 'b (g c) h w -> b g c (h w)', g=self.num_heads)  # 49

        attn = einsum(Q, K, 'b g c n, b g c l -> b g n l')

        attn = rearrange(attn, 'b g n l -> b l g n').contiguous()        # [B, 49, G, N]
        attn = self.attn_proj(attn)                                      # [B, k², G, N]
        attn = rearrange(attn, 'b k g (h w) -> b g 1 k h w', h=oh, w=ow) # [B, G,1,k²,oh,ow]

        weight = w_mod *  weight
        weight = attn * weight
        weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)

        weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        pad_x = pad_x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return res


if __name__ == "__main__":
    model = FCDConv(16, kernel_size=3, stride=2)
    x = torch.randn(4, 16, 128, 128)
    y = model(x)
    print(y.shape)
