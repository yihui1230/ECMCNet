from torch.nn import init
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from TED import cd_Encoder

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        padding = (kernel_size - 1) // 2
        if dilation != 1:
            padding = dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
        nn.BatchNorm2d(in_channels // 2),
        nn.ReLU(),
        nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
    )
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Cos_ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(Cos_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.fc3 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,cos):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        att = self.sigmoid(avg_out + max_out)

        weight=self.fc3(torch.cat([att,cos],1))
        weight=self.sigmoid(weight)
        out=weight*att+(1-weight)*cos
        return out
class Cos_SpatialAttention(nn.Module):
    def __init__(self):
        super(Cos_SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.conv2 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,cos):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        att=self.sigmoid(x)
        weight=self.conv2(torch.cat([att,cos],1))
        weight = self.sigmoid(weight)
        out=weight*att+(1-weight)*cos
        return out





class Attention_C(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention_C, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=8)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=8)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=8)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b,c,h,w)

        return self.project_out(out)

class group_process(nn.Module):
    def __init__(self, group):
        super(group_process, self).__init__()
        dim=128
        channel=dim//group

        self.conv_3=nn.Sequential(ConvBNReLU(channel,64),
                                  ConvBNReLU(64, 64, groups=64),
                                  ConvBNReLU(64,channel),
                                  ConvBNReLU(channel,channel))
        self.conv_5 = nn.Sequential(ConvBNReLU(channel, 64,5),
                                    ConvBNReLU(64, 64, 5, groups=64),
                                    ConvBNReLU(64, channel,5),
                                    ConvBNReLU(channel,channel))
        self.conv_7 = nn.Sequential(ConvBNReLU(channel, 64, 7),
                                    ConvBNReLU(64, 64, 7, groups=64),
                                    ConvBNReLU(64, channel, 7),
                                    ConvBNReLU(channel,channel))
        self.conv_1=nn.Conv2d(channel*3,channel,1)
        self.sam=SpatialAttention()
    def forward(self, x):

        process_conv3_f=self.conv_3(x)
        process_conv5_f = self.conv_5(x)
        process_conv7_f = self.conv_7(x)
        out=self.conv_1(torch.cat([process_conv3_f,process_conv5_f,process_conv7_f],1))+x
        out=self.sam(out)*out

        return out
class SEM(nn.Module):
    def __init__(self, group):
        super(SEM, self).__init__()
        self.group = group
        in_dim = 128
        self.in_dim=in_dim
        self.conv_embedding = nn.Conv2d(in_dim, in_dim, stride=1, kernel_size=1, padding=0)

        self.group_process = nn.ModuleList([ group_process(group) for i in range(group)])

        self.conv = nn.Sequential(ConvBNReLU(128,512),
                                  ConvBNReLU(512, 512, groups=512),
                                  ConvBNReLU(512,128),
                                  nn.Conv2d(128, 128,3,1,1)
                                  )
        self.bn=nn.BatchNorm2d(128)
        self.relu=nn.ReLU()
        self.att=Attention_C(128,8)
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        channel=self.in_dim//self.group
        x_ori = x
        x = self.conv_embedding(x)
        x = self.channel_shuffle(x, self.group)

        x=self.att(x)
        groups_fea=[]

        for i in range(self.group):
            groups_fea.append(self.group_process[i](x[:,channel*i:channel*(i+1),:,:]))

        process_x=torch.cat(groups_fea,1)
        process_x=self.bn(self.conv(process_x))

        res=process_x+x_ori
        res=self.relu(res)

        return res
class DEM(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.in_channel=in_channel
        self.cos = nn.CosineSimilarity(dim=-1)
        self.spatial_am = SpatialAttention()
        self.channel_am = ChannelAttention(self.in_channel)
        self.spatial_diff_am = Cos_SpatialAttention()
        self.channel_diff_am = Cos_ChannelAttention(self.in_channel)
        self.conv_1=ResBlock(in_channel,in_channel)
        self.conv_2 = ResBlock(in_channel, in_channel)
    def forward(self, xA, xB):
        b, c, h, w = xA.shape
        xA = self.conv_1(self.channel_am(xA) * xA)
        xA = self.spatial_am(xA) * xA
        xB = self.conv_1(self.channel_am(xB) * xB)
        xB = self.spatial_am(xB) * xB

        channel_sim = self.cos(xA.reshape(b, c, h*w),
                               xB.reshape(b, c, h*w)).unsqueeze(dim=-1).unsqueeze(dim=-1)  # b c 1 1

        channel_diff = (torch.ones_like(channel_sim) - channel_sim)/2

        channel_attention=self.channel_diff_am(abs(xA-xB),channel_diff)

        xA=self.conv_2(xA*channel_attention)
        xB = self.conv_2(xB * channel_attention)

        spatial_sim = self.cos(xA.permute(0, 2, 3, 1).reshape(b, -1, c),
                               xB.permute(0, 2, 3, 1).reshape(b, -1, c)).unsqueeze(dim=-1)  # b n 1
        spatial_sim = spatial_sim.permute(0, 2, 1).reshape(b, 1, h, w)
        spatial_diff = (torch.ones_like(spatial_sim) - spatial_sim) / 2
        spatial_attention = self.spatial_diff_am(abs(xA - xB), spatial_diff)

        xA = xA * spatial_attention
        xB = xB * spatial_attention

        return xA,xB


class style_removal_block(nn.Module):
    def __init__(self):
        super(style_removal_block,self).__init__()
        self.weight = nn.Parameter(torch.rand(128, 2,2, dtype=torch.float32),requires_grad=True )
        self.conv=ResBlock(128,128)
        self.conv_gate=ConvBNReLU(128,128)

    def forward(self,x):
        _,_,H,W=x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        b,c,h,w=x_freq.shape
        x_freq = torch.fft.fftshift(x_freq,dim=(-2,-1))
        process_freq=x_freq.clone()

        process_freq[:,:,h//2-1:h//2+1,w//2-1:w//2+1]=torch.mul(x_freq[:,:,h//2-1:h//2+1,w//2-1:w//2+1],self.weight)
        x_freq = torch.fft.ifftshift(process_freq, dim=(-2,-1))

        res = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        res=torch.sigmoid(F.gelu(self.conv_gate(res)))*x
        return self.conv(res)

class DSRM(nn.Module):
    def __init__(self):
        super(DSRM,self).__init__()
        self.block_1 = style_removal_block()


    def forward(self,x):
        x=self.block_1(x)

        return x


class EC_MCNet(nn.Module):
    def __init__(self, num_classes=7,group=8):
        super(EC_MCNet, self).__init__()
        self.num_classes = num_classes
        self.net_cd = cd_Encoder()
        self.SEM=SEM(group)
        self.DSRM=DSRM()
        self.DEM = DEM(128)
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)

        self.semantic_pred_1 = make_prediction(128, num_classes)
        self.semantic_pred_2 = make_prediction(128, num_classes)
        self.cd_pred = make_prediction(128, 1)
        self.up2 = nn.Upsample(mode='bilinear', scale_factor=(2, 2))
        self.up4 = nn.Upsample(mode='bilinear', scale_factor=(4, 4))
        self.up8 = nn.Upsample(mode='bilinear', scale_factor=(8, 8))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, xA, xB):
        cdA_3 = self.net_cd(xA)
        cdB_3 = self.net_cd(xB)

        cdA_3=self.SEM(cdA_3)
        cdB_3 = self.SEM(cdB_3)


        cdA_31=self.DSRM(cdA_3)
        cdB_31=self.DSRM(cdB_3)

        cdA_31, cdB_31=self.DEM(cdA_31, cdB_31)

        change_feature=self.resCD(torch.cat([cdA_31 , cdB_31],1))
        cd_pred = self.cd_pred(change_feature)

        semantic_predA_1 = self.semantic_pred_1(cdA_3)
        semantic_predB_1 = self.semantic_pred_2(cdB_3)


        return (self.up4(cd_pred), self.up4(semantic_predA_1), self.up4(semantic_predB_1))

