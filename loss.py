import torch
import torch.nn as nn
# import pytorch_ssim
import odl
import numpy as np
from torchvision.models import vgg19
from odl.contrib.torch import OperatorFunction
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

class DualDomainLoss():
    def __init__(self):
        criterion_mse = nn.MSELoss()
        self.criterion_mse = criterion_mse

        criterion_vgg = VGGLoss()        
        self.criterion_vgg = criterion_vgg

        self.criterion_ssim = SSIM(window_size = 8)

        self.gloss,self.gloss_mse,self.gloss_vgg,self.gloss_ssim,\
                   self.gloss_promse,self.iter = 0,0,0,0,0,0
        
        self.initialize_proj_params()

    def clear(self):
        
        self.gloss,self.gloss_mse,self.gloss_vgg,self.gloss_ssim,\
            self.gloss_promse,self.iter = 0,0,0,0,0,0

    def cal_loss(self,outputs,labels):

        loss_mse = self.criterion_mse(outputs, labels)          * 1
        loss_vgg = self.criterion_vgg(outputs, labels)          * 0.0001
        loss_ssim = (1 - self.criterion_ssim(outputs,labels))   * 0.005

        outputs_proj = self.proj_tensor(outputs)
        labels_proj = self.proj_tensor(labels)
        loss_proj_mse = self.criterion_mse(outputs_proj, labels_proj)
        loss_promse = loss_proj_mse                             * 0.0005
        loss = loss_mse + loss_vgg + loss_ssim + loss_promse 

        self.gloss_mse += loss_mse.item()
        self.gloss_vgg += loss_vgg.item()
        self.gloss_ssim += loss_ssim.item()
        self.gloss_promse += loss_promse.item()
        self.gloss = self.gloss_mse+self.gloss_vgg+self.gloss_ssim+self.gloss_promse

        self.iter += 1

        return loss
    
    def mloss(self):
        return self.gloss/self.iter
    def mmse(self):
        return self.gloss_mse/self.iter
    def mvgg(self):
        return self.gloss_vgg/self.iter
    def mssim(self):
        return self.gloss_ssim/self.iter
    def mprom(self):
        return self.gloss_promse/self.iter

    def initialize_proj_params(self):
        IMG_HEIGHT, IMG_WIDTH = 512, 512
        self.reco_space = odl.uniform_discr(
            min_pt=[-1 * IMG_HEIGHT / 4, -1 * IMG_WIDTH / 4], 
            max_pt=[IMG_HEIGHT / 4, IMG_WIDTH / 4], 
            shape=[IMG_HEIGHT, IMG_WIDTH], 
            dtype='float32'
        )

        angle_partition = odl.uniform_partition(0, 2 * np.pi * 1, 720)
        detector_partition = odl.uniform_partition(-360, 360, 1024)
        geometry = odl.tomo.FanBeamGeometry(
            angle_partition, detector_partition, src_radius=1270, det_radius=870
        )
        self.ray_trafo = odl.tomo.RayTransform(self.reco_space, geometry)

    def proj(self,slice):
        slice_proj = OperatorFunction.apply(self.ray_trafo, slice)
        return slice_proj
    
    def proj_tensor(self,tensor):
        projections = []

        for i in range(tensor.shape[0]):
            slice_2d = tensor[i, 0, :, :]

            projection = self.proj(slice_2d) 

            projection = projection.unsqueeze(0).unsqueeze(0)  # -> (1, 1, proj_h, proj_w)

            projections.append(projection)

        projections_tensor = torch.cat(projections, dim=0)  # -> (b, 1, proj_h, proj_w)

        return projections_tensor

    
class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss() 

    def forward(self, x, y):
        self.vgg = self.vgg.to(x.device)

        global_min = torch.min(torch.min(x), torch.min(y))
        global_max = torch.max(torch.max(x), torch.max(y))

        x_img = (x - global_min) * 255  / (global_max - global_min) 
        y_img = (y - global_min) * 255  / (global_max - global_min) 

        x_img = x_img.repeat(1, 3, 1, 1)
        y_img = y_img.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x_img), self.vgg(y_img)

        loss = self.criterion(x_vgg, y_vgg)

        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
    
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)