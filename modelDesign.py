'''
The content of this file could be self-defined. 
But please note the interface of the following function cannot be modified,
    - encFunction_1
    - decFunction_1
    - encFunction_2
    - decFunction_2
'''
# =======================================================================================================================
# =======================================================================================================================
# Package Importing
import random
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F


# =======================================================================================================================
# =======================================================================================================================
# Number to Bit Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


# =======================================================================================================================
# =======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


# =======================================================================================================================
# =======================================================================================================================
# Eigenvector Calculation Function Defining
def cal_eigenvector(channel):
    """
        Description:
            calculate the eigenvector on each subband
        Input:
            channel: np.array, channel in frequency domain,  shape [batch_size, rx_num, tx_num, subcarrier_num]
        Output:
            eigenvectors:  np.array, eigenvector for each subband, shape [batch_size, tx_num, subband_num]
    """
    subband_num = 13
    hf_ = np.transpose(channel, [0, 3, 1, 2])  # (batch,subcarrier_num,4,32)
    hf_h = np.conj(np.transpose(channel, [0, 3, 2, 1]))  # (batch,subcarrier_num,32,4)
    R = np.matmul(hf_h, hf_)  # (batch,subcarrier_num,32,32)
    R = R.reshape(R.shape[0], subband_num, -1, R.shape[2], R.shape[3]).mean(
        axis=2)  # average the R over each subband, (batch,13,32,32)
    [D, V] = np.linalg.eig(R)
    v = V[:, :, :, 0]
    eigenvectors = np.transpose(v, [0, 2, 1])
    return eigenvectors


# =======================================================================================================================
# =======================================================================================================================
# Loss Function Defining
class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, w_true, w_pre):
        cos_similarity = cosine_similarity_cuda(w_true.permute(0, 3, 2, 1), w_pre.permute(0, 3, 2, 1))
        if self.reduction == 'mean':
            cos_similarity_scalar = cos_similarity.mean()
        elif self.reduction == 'sum':
            cos_similarity_scalar = cos_similarity.sum()
        return 1 - cos_similarity_scalar


def cosine_similarity_cuda(w_true, w_pre):
    num_batch, num_sc, num_ant = w_true.size(0), w_true.size(1), w_true.size(2)
    w_true = w_true.reshape(num_batch * num_sc, num_ant, 2)
    w_pre = w_pre.reshape(num_batch * num_sc, num_ant, 2)
    w_true_re, w_true_im = w_true[..., 0], w_true[..., 1]
    w_pre_re, w_pre_im = w_pre[..., 0], w_pre[..., 1]
    numerator_re = (w_true_re * w_pre_re + w_true_im * w_pre_im).sum(-1)
    numerator_im = (w_true_im * w_pre_re - w_true_re * w_pre_im).sum(-1)
    denominator_0 = (w_true_re ** 2 + w_true_im ** 2).sum(-1)
    denominator_1 = (w_pre_re ** 2 + w_pre_im ** 2).sum(-1)
    cos_similarity = torch.sqrt(numerator_re ** 2 + numerator_im ** 2) / (
            torch.sqrt(denominator_0) * torch.sqrt(denominator_1))
    cos_similarity = cos_similarity ** 2
    return cos_similarity


# =======================================================================================================================
# =======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matInput, matLabel):
        self.input, self.label = matInput, matLabel

    def __getitem__(self, index):
        return self.input[index], self.label[index]

    def __len__(self):
        return self.input.shape[0]


class DatasetFolder_mixup(Dataset):
    def __init__(self, matInput, matLabel):
        self.input, self.label = matInput, matLabel

    def __getitem__(self, index):
        mixup_ratio = 0.3
        r = np.random.rand(1)
        if r < mixup_ratio:
            mix_idx = random.randint(0, self.input.shape[0] - 1)
            lam = np.random.rand(1)
            mix_input = np.zeros(self.input[index].shape, dtype='float32')
            mix_label = np.zeros(self.label[index].shape, dtype='float32')
            mix_input[:] = lam * self.input[index] + (1 - lam) * self.input[mix_idx]
            mix_label[:] = lam * self.label[index] + (1 - lam) * self.label[mix_idx]
        else:
            mix_input, mix_label = self.input[index], self.label[index]
        return mix_input, mix_label

    def __len__(self):
        return self.input.shape[0]


class DatasetFolder_eval(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h * cut_ratio), int(w * cut_ratio)

    fcy = np.random.randint(0, h - ch + 1)
    fcx = np.random.randint(0, w - cw + 1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmixup(
        im1, im2,
        mixup_prob=1.0, mixup_alpha=1.0,
        cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch * scale, cw * scale
    hfcy, hfcx, htcy, htcx = fcy * scale, fcx * scale, tcy * scale, tcx * scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1 - v) * im2[rindex, :]
        im1_aug = v * im1 + (1 - v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy + ch, tcx:tcx + cw] = im2_aug[..., fcy:fcy + ch, fcx:fcx + cw]
        im1[..., htcy:htcy + hch, htcx:htcx + hcw] = im1_aug[..., hfcy:hfcy + hch, hfcx:hfcx + hcw]
    else:
        im2_aug[..., tcy:tcy + ch, tcx:tcx + cw] = im2[..., fcy:fcy + ch, fcx:fcx + cw]
        im1_aug[..., htcy:htcy + hch, htcx:htcx + hcw] = im1[..., hfcy:hfcy + hch, hfcx:hfcx + hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(2)
    # i1 = im1
    # i2 = im2
    # im1[:, :,:,1] = i1[:, :,:,2]
    # im1[:, :,:,2] = i1[:, :,:,1]
    # im2[:, :,:,1] = i2[:, :,:,2]
    # im2[:, :,:,2] = i2[:, :,:,1]
    im1 = im1[:, perm, :, :]
    im2 = im2[:, perm, :, :]
    # im1 = im1[:, perm]
    # im2 = im2[:, perm]

    return im1, im2


def rgb1(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    se = np.zeros(2)
    se[0] = 1
    se[1] = -1

    r = np.random.randint(2)
    phase = se[r]
    im1[:, 0, :, :] = phase * im1[:, 0, :, :]
    im2[:, 0, :, :] = phase * im2[:, 0, :, :]
    r = np.random.randint(2)
    phase = se[r]
    im1[:, 1, :, :] = phase * im1[:, 1, :, :]
    im2[:, 1, :, :] = phase * im2[:, 1, :, :]

    return im1, im2


def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch * scale, cw * scale
    hfcy, hfcx, htcy, htcx = fcy * scale, fcx * scale, tcy * scale, tcx * scale

    im2[..., tcy:tcy + ch, tcx:tcx + cw] = im2[rindex, :, fcy:fcy + ch, fcx:fcx + cw]
    im1[..., htcy:htcy + hch, htcx:htcx + hcw] = im1[rindex, :, hfcy:hfcy + hch, hfcx:hfcx + hcw]

    return im1, im2


def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1 - v) * im1[r_index, :]
    im2 = v * im2 + (1 - v) * im2[r_index, :]

    return im1, im2


# ==================================
# =======================================================================================================================
# =======================================================================================================================
# Model Defining
# Channel Estimation Class Defining

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim1, dim2, dim3, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim3, dim3, bias=qkv_bias)

        self.reweight = Mlp(dim1, dim1 // 4, dim1 *3)
        
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H*S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W*S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)
        
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PermutatorBlock(nn.Module):

    def __init__(self, dim1, dim2, dim3, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn = WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.attn = mlp_fn(dim1, dim2, dim3, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) / self.skip_lam
        x = x + self.mlp(self.norm2(x)) / self.skip_lam
        return x

class channel_est(nn.Module):
    def __init__(self, input_length):
        super(channel_est, self).__init__()
        pilot_num = int(input_length / 8)
        emd_dim = 192
        mid_dim1 = 256
        mid_dim2 = 512
        self.out_dim1 = 8
        self.out_dim2 = 16
        self.out_dim3 = 16
        self.out_dim4 = 8

        patch_size_h = 1
        patch_size_w = 1
        image_size_h = 32
        image_size_w = 52
        seg_dim = 16
        seg = emd_dim//seg_dim
        
        num_patch = (image_size_h // patch_size_h) * (image_size_w // patch_size_w)

        self.mlp_mixer1 = PermutatorBlock(dim1 = emd_dim, dim2 = image_size_h*seg, dim3 = image_size_w*seg, segment_dim=seg_dim)

        self.mlp_mixer2 = PermutatorBlock(dim1 = emd_dim, dim2 = image_size_h*seg, dim3 = image_size_w*seg, segment_dim=seg_dim)

        self.mlp_mixer3 = PermutatorBlock(dim1 = emd_dim, dim2 = image_size_h*seg, dim3 = image_size_w*seg, segment_dim=seg_dim)

        self.mlp_mixer11 = PermutatorBlock(dim1 = emd_dim, dim2 = image_size_h*seg, dim3 = image_size_w*seg, segment_dim=seg_dim)

        self.mlp_mixer22 = PermutatorBlock(dim1 = emd_dim, dim2 = image_size_h*seg, dim3 = image_size_w*seg, segment_dim=seg_dim)
        
        self.mlp_mixer33 = MixerBlock(emd_dim, num_patch, mid_dim1, mid_dim2)
        
        self.mlp_1 = PermutatorBlock(dim1 = emd_dim, dim2 = image_size_h*seg, dim3 = image_size_w*seg, segment_dim=seg_dim)

        # self.mlp_mixer11 = MLPMixer(in_channels=self.out_dim4+self.out_dim1, dim=emd_dim, num_output=8, 
        #                         patch_size_h=1, patch_size_w=1, image_size_h=image_size_h, 
        #                         image_size_w=26, depth=3, token_dim=mid_dim, channel_dim=mid_dim)

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(8, emd_dim, (patch_size_h, patch_size_w), (patch_size_h, patch_size_w)),
            Rearrange('b c h w -> b h w c')
            # Rearrange('b c h w -> b (h w) c'),
        )

        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(8, emd_dim, 3, stride=1, padding=1),
        #     Rearrange('b c h w -> b (h w) c'),
        # )

        self.fc1 = nn.Linear(pilot_num, image_size_w)
        self.fc_1 = nn.Linear(8, emd_dim)
        self.fc2 = nn.Linear(num_patch, image_size_h * image_size_w)
        # self.fc2 = nn.Linear(52, 52)
        self.fc3 = nn.Linear(image_size_h * image_size_w, image_size_h * image_size_w)

        self.fc11 = nn.Linear(emd_dim * 2, emd_dim)
        self.fc22 = nn.Linear(emd_dim * 2, emd_dim)

        self.layer_norm = nn.LayerNorm(emd_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(emd_dim, 8)
        )

        self.prelu = nn.PReLU()

    def forward(self, x):  # (batch, 2, 4, 208 or 48, 4)
        x = rearrange(x, 'b c rx (rb subc) sym -> b (c rx) (subc sym) rb', subc=8)  # (batch,8,32,26 or 6)
        up_size = (32, 52)
        # x = nn.UpsamplingNearest2d(size=up_size)(x) #2 x 96 x 14
        x = self.fc1(x)  # (batch,8,32,52)
        # x = rearrange(x,'b c h w -> b (h w) c')
        # x = self.fc_1(x)
        
        x = self.to_patch_embedding(x)
        # x = self.mlp_1(x)
        
        # x = rearrange(x,'b c h w -> b (h w) c')
        out1 = self.mlp_mixer1(x)  # (batch, patch_num, out_dim)
        # out1 = out.reshape(out.shape[0],self.out_dim1,32,26)

        out2 = self.mlp_mixer2(out1)  # (batch, patch_num, out_dim)

        out3 = self.mlp_mixer3(out2)  # (batch, patch_num, out_dim)

        caout1 = torch.cat([out2, out3], 3)
        caout1 = self.fc11(caout1)
        out22 = self.mlp_mixer11(caout1)  # (batch, patch_num, out_dim)

        caout2 = torch.cat([out22, out1], 3)
        caout2 = self.fc22(caout2)

        out11 = self.mlp_mixer22(caout2)

        # out11 = out11 + x

        # out11 = self.mlp_mixer33(out11)

        out11 = self.layer_norm(out11)  # (batch, patch, dim)
        # x = x.mean(dim=1)
        out11 = self.mlp_head(out11)
        
        out = rearrange(out11, 'b p o c-> b c (p o)')

        # out = rearrange(out11, 'b p o (tx rb)-> b tx rb p o', tx=2)
        # out2 = rearrange(out, 'b (p1 p2) (subc rb) -> b p1 subc (p2 rb)', p1=8, subc=32)
        # deout1 = out.reshape(deout.shape[0],self.out_dim2,16,26)

        # out = self.prelu(F.max_pool2d(self.encoder3(x),2,2))
        # out = self.mlp_mixer2(out2) # (batch, patch_num, out_dim)
        # out3 = out.reshape(out.shape[0],self.out_dim2//4,16,26)
        # out4 = torch.cat

        out = self.fc2(out)  # (batch, 8, 52*32)
        # ou = out
        # out = self.fc3(out) # (batch, 8, 52*32)
        # out = out + ou

        out = rearrange(out, 'b (c rx) (tx rb) -> b c rx tx rb', rx=4, tx=32)
        # out = rearrange(out, 'b (tx rb) p o -> b tx rb p o', tx=2)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_output, patch_size_h, patch_size_w, image_size_h, image_size_w, depth,
                 token_dim, channel_dim):
        super().__init__()
        assert image_size_w % patch_size_w == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size_h % patch_size_h == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size_h // patch_size_h) * (image_size_w // patch_size_w)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, (patch_size_h, patch_size_w), (patch_size_h, patch_size_w)),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_output)
        )

    def forward(self, x):  # (batch, 8, 26, 32)
        x = self.to_patch_embedding(x)  # (batch, patch, dim)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)  # (batch, patch, dim)
        # x = x.mean(dim=1)
        return self.mlp_head(x)


# Encoder and Decoder Class Defining

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    B = 2

    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        d_model = 384
        nhead = 6
        d_hid = 512
        dropout = 0.0
        nlayers = 4
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc1 = nn.Linear(64, d_model)
        self.fc2 = nn.Linear(13 * d_model, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization

    def forward(self, x):  # (batch, 2, 32, 13)
        x = rearrange(x, 'b c eig f -> f b (c eig)')
        out = self.fc1(x)  # (13, batch, d_model)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = rearrange(out, 'f b dmodel -> b (f dmodel)')
        out = self.fc2(out)  # (batch, 512/B)
        out = self.sig(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out


class Decoder(nn.Module):
    B = 2

    def __init__(self, feedback_bits, quantization=True):
        super(Decoder, self).__init__()
        d_model = 384
        nhead = 6
        d_hid = 512
        dropout = 0.0
        nlayers = 4
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, nlayers)
        self.sig = nn.Sigmoid()
        self.quantization = quantization
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc1 = nn.Linear(int(feedback_bits / self.B), 13 * d_model)
        self.fc2 = nn.Linear(d_model, 32 * 2)

    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = self.fc1(out)  # (batch, 13*d_model)
        out = rearrange(out, 'b (f dmodel) -> f b dmodel', f=13)
        out = self.pos_encoder(out)
        out = self.transformer_decoder(out)  # (13, batch, d_model)
        out = self.fc2(out)  # (13, batch, 32*2)
        out = rearrange(out, 'f b (c eig) -> b c eig f', c=2)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


# =======================================================================================================================
# =======================================================================================================================
# Function Defining
def encFunction_1(pilot_1, encModel_p1_1_path, encModel_p1_2_path):
    """
        Description:
            CSI compression based on received pilot signal
        Input:
            pilot_1: np.array, received pilot signal,  shape [NUM_SAMPLES, 2, rx_num, pilot on different subcarrier, pilot on different symbol]
            encModel_p1_1_path: path to load the first AI model, please ignore if not needed
            encModel_p1_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, encoded bit steam, shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
    """
    num_feedback_bits = 64
    subc_num = 208
    model_ce = channel_est(subc_num).cuda()
    model_ce.load_state_dict(torch.load(encModel_p1_1_path)['state_dict'])
    model_fb = AutoEncoder(num_feedback_bits).cuda()
    model_fb.encoder.load_state_dict(torch.load(encModel_p1_2_path)['state_dict'])
    test_dataset = DatasetFolder_eval(pilot_1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4,
                                              pin_memory=True)
    model_ce.eval()
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            # step 1: channel estimation
            h = model_ce(data)  # (batch,2,4,32,52)
            # step 2: eigenvector calculation
            h_complex = h[:, 0, ...] + 1j * h[:, 1, ...]  # (batch,4,32,52)
            h_complex = h_complex.cpu().numpy()
            v = cal_eigenvector(h_complex)
            # step 3: eigenvector compression
            w_complex = torch.from_numpy(v)
            w = torch.zeros([h.shape[0], 2, 32, 13], dtype=torch.float32).cuda()  # (batch,2,32,13)
            w[:, 0, :, :] = torch.real(w_complex)
            w[:, 1, :, :] = torch.imag(w_complex)
            modelOutput = model_fb.encoder(w)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all


def decFunction_1(bits_1, decModel_p1_1_path, decModel_p1_2_path):
    """
        Description:
            CSI reconstruction based on feedbacked bit stream
        Input:
            bits_1: np.array, feedbacked bit stream,  shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
            decModel_p1_1_path: path to load the first AI model, please ignore if not needed
            decModel_p1_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, reconstructed CSI (eigenvectors), shape [NUM_SAMPLES, 2, NUM_TX, NUM_SUBBAND]
    """
    num_feedback_bits = 64
    model_fb = AutoEncoder(num_feedback_bits).cuda()
    model_fb.decoder.load_state_dict(torch.load(decModel_p1_1_path)['state_dict'])
    test_dataset = DatasetFolder_eval(bits_1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4,
                                              pin_memory=True)
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            modelOutput = model_fb.decoder(data)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all


def encFunction_2(pilot_2, encModel_p2_1_path, encModel_p2_2_path):
    """
        Description:
            CSI compression based on received pilot signal
        Input:
            pilot_2: np.array, received pilot signal,  shape [NUM_SAMPLES, 2, rx_num, pilot on different subcarrier, pilot on different symbol]
            encModel_p2_1_path: path to load the first AI model, please ignore if not needed
            encModel_p2_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, encoded bit steam, shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
    """
    num_feedback_bits = 64
    subc_num = 48
    model_ce = channel_est(subc_num).cuda()
    model_ce.load_state_dict(torch.load(encModel_p2_1_path)['state_dict'])
    model_fb = AutoEncoder(num_feedback_bits).cuda()
    model_fb.encoder.load_state_dict(torch.load(encModel_p2_2_path)['state_dict'])
    test_dataset = DatasetFolder_eval(pilot_2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4,
                                              pin_memory=True)
    model_ce.eval()
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            # step 1: channel estimation
            h = model_ce(data)  # (batch,2,4,32,52)
            # step 2: eigenvector calculation
            h_complex = h[:, 0, ...] + 1j * h[:, 1, ...]  # (batch,4,32,52)
            h_complex = h_complex.cpu().numpy()
            v = cal_eigenvector(h_complex)
            # step 3: eigenvector compression
            w_complex = torch.from_numpy(v)
            w = torch.zeros([h.shape[0], 2, 32, 13], dtype=torch.float32).cuda()  # (batch,2,32,13)
            w[:, 0, :, :] = torch.real(w_complex)
            w[:, 1, :, :] = torch.imag(w_complex)
            modelOutput = model_fb.encoder(w)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all


def decFunction_2(bits_2, decModel_p2_1_path, decModel_p2_2_path):
    """
        Description:
            CSI reconstruction based on feedbacked bit stream
        Input:
            bits_2: np.array, feedbacked bit stream,  shape [NUM_SAMPLES, NUM_FEEDBACK_BITS]
            decModel_p2_1_path: path to load the first AI model, please ignore if not needed
            decModel_p2_2_path: path to load the second AI model, please ignore if not needed
            *** Note: Participants can flexibly decide to use 0/1/2 AI model in this function, needless model path can be ignored
        Output:
            output_all:  np.array, reconstructed CSI (eigenvectors), shape [NUM_SAMPLES, 2, NUM_TX, NUM_SUBBAND]
    """
    num_feedback_bits = 64
    model_fb = AutoEncoder(num_feedback_bits).cuda()
    model_fb.decoder.load_state_dict(torch.load(decModel_p2_1_path)['state_dict'])
    test_dataset = DatasetFolder_eval(bits_2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4,
                                              pin_memory=True)
    model_fb.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data = data.cuda()
            modelOutput = model_fb.decoder(data)
            modelOutput = modelOutput.cpu().numpy()
            if idx == 0:
                output_all = modelOutput
            else:
                output_all = np.concatenate((output_all, modelOutput), axis=0)
    return output_all
