from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import lpips
from PIL import Image
import SimpleITK as sitk
import time
from calflops import calculate_flops
import tifffile
from network_MITAMP import MITNet
import glob 
import os
import json
import sys
from alive_progress import alive_bar
from collections import OrderedDict


def unstandard(standard_img):        
    mean=-556.882367
    variance=225653.408219
    nii_slice = standard_img * np.sqrt(variance) + mean
    return nii_slice        

def standard(nii_slice):
    mean=-556.882367
    variance=225653.408219
    nii_slice = nii_slice.astype(np.float32)
    nii_slice = (nii_slice - mean) / np.sqrt(variance)
    return nii_slice



if __name__ == '__main__':
    # simple_test
    path = "models/MITAMP_7242.pkl"
    state_dict = torch.load(path)    
    state_dict = torch.load(path,map_location="cpu")    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v

    model = MITNet()    
    model.load_state_dict(new_state_dict)
    model.to("cuda:0").eval()
    # model.load_state_dict(state_dict).to("cuda:0").eval()

    simple_test_data_folder = "testsets/simple_test/"
    settings = ["LDCT", "LACT", "SVCT"]
    degrees = ["Low", "Mid", "High"]
    input_paths = [
        f"{simple_test_data_folder}{setting}_{degree}_input.nii.gz"
        for setting in settings for degree in degrees
    ]


    for input_path in input_paths:
        input_image = sitk.ReadImage(input_path)
        input = sitk.GetArrayFromImage(input_image)        
        input_tensor = torch.tensor(standard(input), dtype=torch.float).unsqueeze(0).to("cuda:0")

        output_tensor = model(input_tensor)

        output = unstandard(np.array(output_tensor[0].cpu().detach())).astype('int16')
        output_path = input_path.replace("input", "output")
        output = sitk.GetImageFromArray(output)
        output.CopyInformation(input_image)
        sitk.WriteImage(output, output_path)

# def package_nii(modified_array,input_file_address,output_file_address):
#     image = sitk.ReadImage(input_file_address)
#     modified_image = sitk.GetImageFromArray(modified_array)
#     modified_image.CopyInformation(image)
#     sitk.WriteImage(modified_image, output_file_address)
