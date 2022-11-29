import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from tensorboardX import SummaryWriter
import numpy as np
class FinalLayer(nn.Module):
    def __init__(self,input_channel):
        super(FinalLayer, self).__init__()
        ks = 3
        pw = ks//2
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=ks,padding=pw)
        self.instace_norm = nn.InstanceNorm2d(num_features=input_channel)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels=3,kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.instace_norm(x1)
        x1 = self.leaky_relu(x1)
        x = x1 + x
        x = self.conv2(x)
        return x

class FusionModule(nn.Module):
    def __init__(self,in_channels,out_channels,output_size,atten = False):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,out_channels = in_channels,kernel_size=3,padding=3//2)
        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(num_features=in_channels)
        self.atten = atten

        self.conv2 = nn.Conv2d(in_channels = in_channels,out_channels=out_channels,kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(num_features=out_channels)
        self.size = output_size

        self.sigmoid = nn.Sigmoid()

    def forward(self,x_raw,x_c,x_prior):
        x_fusion = x_raw + self.sigmoid(x_c) * x_prior
        x = F.interpolate(x_fusion,size=(self.size,self.size),mode='bilinear',align_corners=False)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        if(self.atten):
            x = self.attention(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        return x
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self,q,k,v):
        '''
        Computes
        :param q: Queries (B, C, H, W)
        :param k: Keys (B, C, H, W)
        :param v: Values (B, C, H, W)
        :return:
        '''
        B,C,H,W = q.shape
        H = int(H/2)
        W = int(W/2)
        q = F.interpolate(q,size=(H,W),mode='bilinear',align_corners=False)
        k = F.interpolate(k,size=(H,W),mode='bilinear',align_corners=False)
        v = F.interpolate(v,size=(H,W),mode='bilinear',align_corners=False)

        q = q.view(B,1,C,H,W).view(B,1,C,H*W).permute(0,1,3,2) # (B,h,HxW,C)
        k = k.view(B,1,C,H,W).view(B,1,C,H*W) # (B,h,C,HxW)
        v = v.view(B,1,C,H,W).view(B,1,C,H*W).permute(0,1,3,2) # (B,h,HxW,C)
        attn = torch.matmul(q, k) / np.sqrt(H*W) # (B,h,C,HxW)
        attn = torch.softmax(attn,dim = -1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2) # (B,h,C,HxW)
        out = out.view(B,C,H,W)
        H*=2
        W*=2
        out = F.interpolate(out,size=(H,W),mode='bilinear',align_corners=False)


        return out

class upSample(nn.Module):
    def __init__(self,in_channels,out_channels,output_size):
        super(upSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels*2,out_channels = in_channels,kernel_size=3,padding=3//2)
        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels = in_channels,out_channels=out_channels,kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(num_features=out_channels)
        self.size = output_size
    def forward(self,x,skip):
        x = torch.cat((x,skip),dim = 1)
        x = F.interpolate(x,size=(self.size,self.size),mode='bilinear',align_corners=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.up4 = upSample(in_channels=256,out_channels=128,output_size=64)
        self.up3 = upSample(in_channels=128,out_channels=64,output_size=128)
        self.up2 = upSample(in_channels=64,out_channels=32,output_size=256)
        self.up1 = upSample(in_channels=32,out_channels=16,output_size=512)
        self.finalLayer = FinalLayer(input_channel=16*2)

    def forward(self,x,x_0,x_1,x_2,x_3,x_4):
        # x_0: (B,16,512,512)
        # x_1: (B,32,256,256)
        # x_2: (B,64,128,128)
        # x_3: (B,128,64,64)
        # x_4: (B,256,32,32)

        x = self.up4(x,x_4) #(B,128,64,64)
        x = self.up3(x,x_3) #(B,64,128,128)
        x = self.up2(x,x_2) #(B,32,256,256)
        x = self.up1(x,x_1) #B(B,16,512,512)

        x = torch.cat((x,x_0),dim = 1)
        x = self.finalLayer(x) #(B,3,512,512)
        return x



class TransSfPNet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False,device= None):
        super(TransSfPNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.kernel_size = 9
        self.lamda = 1
        self.m = 0.5
        self.mean_kernel = torch.ones([1,1,self.kernel_size,self.kernel_size])/self.kernel_size**2
        self.mean_kernel = self.mean_kernel.to(device)
        self.mean_kernel = nn.Parameter(data=self.mean_kernel, requires_grad=False)
        self.sum_kernel_1 = torch.ones([1,1,self.kernel_size,self.kernel_size])
        self.sum_kernel_1 = self.sum_kernel_1.to(device)
        self.sum_kernel_1 = nn.Parameter(data=self.sum_kernel_1, requires_grad=False)

        self.sum_kernel_3 = torch.ones([3,3,self.kernel_size,self.kernel_size])
        self.sum_kernel_3 = self.sum_kernel_3.to(device)
        self.sum_kernel_3 = nn.Parameter(data=self.sum_kernel_3, requires_grad=False)



        # branch
        self.backbone_orig = build_backbone(in_channels=2 ,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm,Fusion=True)
        self.aspp_orig = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone_prior = build_backbone(in_channels=12,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm,Fusion=True)
        self.aspp_prior = build_aspp(backbone,output_stride,BatchNorm)
        self.backbone_atten = build_backbone(in_channels=1 ,backbone=backbone,output_stride=output_stride,BatchNorm=BatchNorm,Fusion=True)
        self.aspp_atten = build_aspp(backbone,output_stride,BatchNorm)

        self.decoder = decoder()

        self.fusion_module_0 = FusionModule(in_channels=64,out_channels=16,output_size=512)
        self.fusion_module_1 = FusionModule(in_channels=64,out_channels=32,output_size=256)
        self.fusion_module_2 = FusionModule(in_channels=256,out_channels=64,output_size=128)
        self.fusion_module_3 = FusionModule(in_channels=512,out_channels=128,output_size=64)
        self.fusion_module_4 = FusionModule(in_channels=1024,out_channels=256,output_size=32)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        if freeze_bn:
            self.freeze_bn()





        self.writer = SummaryWriter()

    def forward(self, orig,prior):
        img  = orig
        img_split = torch.split(img, 1, 1)
        aolp = img_split[1]
        # aolp = (torch.cos(2*aolp) + 1.0)/2.0

        # get attention map
        mean_map = nn.functional.conv2d(aolp,self.mean_kernel,padding=self.kernel_size//2)
        abs_map = torch.abs(aolp - mean_map)
        abs_map = torch.pow(abs_map,self.m)
        confidence_map = nn.functional.conv2d(abs_map,self.sum_kernel_1,padding=self.kernel_size//2)
        shape = confidence_map.shape
        confidence_map = torch.reshape(confidence_map,[shape[0],-1])
        max_values,indices = torch.max(confidence_map,dim = 1)
        max_values = torch.reshape(max_values,[shape[0],1])
        confidence_map = torch.div(confidence_map,max_values)
        confidence_map = torch.reshape(confidence_map,[shape[0],shape[1],shape[2],shape[3]])
        confidence_map = 1 - confidence_map



        # orig-polar branch
        x_orig, x_orig_0,x_orig_1,x_orig_2,x_orig_3,x_orig_4 = self.backbone_orig(orig)
        x_orig = self.aspp_orig(x_orig)

        # prior branch
        x_prior, x_prior_0,x_prior_1,x_prior_2,x_prior_3,x_prior_4 = self.backbone_prior(prior)
        x_prior = self.aspp_prior(x_prior)




        # confidence branch
        x_c,x_c_0,x_c_1,x_c_2,x_c_3,x_c_4 = self.backbone_atten(confidence_map)
        x_c = self.aspp_atten(x_c)        




        # fusion step
        x_fusion = x_orig + self.sigmoid(x_c) * x_prior
        


        x_fusion_0 = self.fusion_module_0(x_orig_0,x_c_0,x_prior_0)
        x_fusion_1 = self.fusion_module_1(x_orig_1,x_c_1,x_prior_1)
        x_fusion_2 = self.fusion_module_2(x_orig_2,x_c_2,x_prior_2)
        x_fusion_3 = self.fusion_module_3(x_orig_3,x_c_3,x_prior_3)
        x_fusion_4 = self.fusion_module_4(x_orig_4,x_c_4,x_prior_4)

        # decode
        x = self.decoder(x_fusion,x_fusion_0,x_fusion_1,x_fusion_2,x_fusion_3,x_fusion_4)



        return x,confidence_map

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
