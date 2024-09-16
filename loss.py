import torch
import torch.nn as nn
import pytorch_ssim
import odl
import numpy as np
from torchvision.models import vgg19
from odl.contrib.torch import OperatorFunction

class DualDomainLoss():
    def __init__(self):
        criterion_mse = nn.MSELoss()
        self.criterion_mse = criterion_mse

        criterion_vgg = VGGLoss()        
        self.criterion_vgg = criterion_vgg

        self.criterion_ssim = pytorch_ssim.SSIM(window_size = 8)

        self.gloss,self.gloss_mse,self.gloss_vgg,self.gloss_ssim,\
                   self.gloss_promse,self.iter = 0,0,0,0,0,0

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
    
    def proj_tensor(self,tensor):
        projections = []

        for i in range(tensor.shape[0]):
            slice_2d = tensor[i, 0, :, :]
            
            projection = self.proj(slice_2d) 

            projection = projection.unsqueeze(0).unsqueeze(0)  # -> (1, 1, proj_h, proj_w)

            projections.append(projection)

        projections_tensor = torch.cat(projections, dim=0)  # -> (b, 1, proj_h, proj_w)

        return projections_tensor

    def proj(self,slice):
        IMG_HEIGHT,IMG_WIDTH = 512,512

        reco_space = odl.uniform_discr(
            min_pt=[-1*IMG_HEIGHT/4, -1*IMG_WIDTH/4], max_pt=[IMG_HEIGHT/4, IMG_WIDTH/4]
            , shape=[IMG_HEIGHT, IMG_WIDTH], dtype='float32')

        angle_partition = odl.uniform_partition(0, 2 * np.pi *1, 720)
        detector_partition = odl.uniform_partition(-360, 360, 1024)
        geometry = odl.tomo.FanBeamGeometry(
                angle_partition, detector_partition, src_radius=1270, det_radius=870)
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)        

        slice_proj = OperatorFunction.apply(ray_trafo,slice)

        return slice_proj
    
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
