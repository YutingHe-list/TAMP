from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import lpips
from PIL import Image
import time
from calflops import calculate_flops
import tifffile
import glob 
import os
import json
import sys
from alive_progress import alive_bar
from collections import OrderedDict


from network_MITAMP import MITNet
from loss import DualDomainLoss as my_loss
from database import Database as my_database

import numpy as np
import torch
import argparse
import SimpleITK as sitk
from adan import Adan


from peft import get_peft_model_state_dict
from peft import LoraConfig
from peft import get_peft_model
from peft import set_peft_model_state_dict

def get_parser():
    parser = argparse.ArgumentParser(description='MAIN FUNCTION PARSER')
    parser.add_argument('--testing_mode', type=str, default="fine-tuning")
    parser.add_argument('--NICT_setting', type=str, default="LDCT")
    parser.add_argument('--defect_degree', type=str, default="Low")

    parser.add_argument('--LoRA_load_set', type=str, default="1")
    parser.add_argument('--nii_start_index', type=int, default=1)
    parser.add_argument('--queue_len', type=int, default=5)

# batch_size
# cuda_index str

    return parser

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

def load_model(opt):
    
    state_dict = torch.load("models/foundation_model_weight/MITAMP.pkl")
    model = MITNet()
    model.load_state_dict(state_dict)

    if opt.testing_mode in ["fine_tuning", "testing_finetuned"]:

        with open('lora_path.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        modules_list = content.strip().replace("'", "").split(',\n')
        target_modules_txt = []
        for module in modules_list:
            target_modules_txt.append(module)            
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules_txt,
            modules_to_save=["stage_2.conv_easy"],
        )
        model = get_peft_model(model, config)
        
        if opt.testing_mode == "testing_finetuned":
            lora_path = f"models/fine-tune_LoRA_weight/LoRA_{opt.NICT_setting}_{opt.defect_degree}_{opt.LoRA_load_set}.pkl"
            lora_state_dict = torch.load(lora_path,map_location="cpu")    
            set_peft_model_state_dict(model,lora_state_dict)

    model.to("cuda:0")    
    return model

def package_nii(modified_array,input_file_address,output_file_address):
    image = sitk.ReadImage(input_file_address)
    modified_image = sitk.GetImageFromArray(modified_array)
    modified_image.CopyInformation(image)
    sitk.WriteImage(modified_image, output_file_address)

def slice_testing():
    model = load_model()

    slice_test_data_folder = "samples/slice_test/input"
    settings = ["LDCT", "LACT", "SVCT"]
    degrees = ["Low", "Mid", "High"]
    input_paths = [
        f"{slice_test_data_folder}/{setting}_{degree}.nii.gz"
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


def volume_testing():
    input_folder = "samples/volume_test/input"
    settings = ["LDCT", "LACT", "SVCT"]
    degrees = ["Low", "Mid", "High"]
    nii_folders=[]
    for setting in settings:
        for degree in degrees:
            nii_folders.append(f"{input_folder}/{setting}_{degree}")

    model = load_model()

    for nii_folder in nii_folders:
        for input_nii_set in range(1,12):
            input_nii_path = f"{nii_folder}/{input_nii_set}.nii.gz"
            input_nii_image = sitk.ReadImage(input_nii_path)
            input_nii_file =  sitk.GetArrayFromImage(input_nii_image)

            input_nii_tensor = torch.tensor(standard(input_nii_file), dtype=torch.float).to("cuda:2")
            S,H,W=input_nii_file.shape
            output_nii_file = np.zeros((S,H,W),dtype=np.int16)

            for i in range(S):
                input_slice_tensor = input_nii_tensor[i].unsqueeze(0).unsqueeze(0)
                output_nii_tensor = model(input_slice_tensor)            
                output = unstandard(np.array(output_nii_tensor.cpu().detach())).astype('int16')
                output_nii_file[i] = output[0][0]

            output_nii_image = sitk.GetImageFromArray(output_nii_file)
            output_nii_image.CopyInformation(input_nii_image)
            output_nii_path=input_nii_path.replace("input", "output")
            sitk.WriteImage(output_nii_image, output_nii_path)

def show_training_global_info(nii_epoch,train_loss):
    sys.stdout.write(
        "\r"+" "*70+"\r[Epoch %d] [loss %f]\n" % (nii_epoch,train_loss.mloss())
    )

def show_training_local_info(nii_epoch,train_loss):
    sys.stdout.write(
        "\r"+" "*70+
        f"\r[{nii_epoch}] [loss:{train_loss.mloss():.4f} {train_loss.mmse():2f} {train_loss.mvgg():2f} {train_loss.mssim():2f} {train_loss.mprom():2f}]"
    )

def fine_tuning(opt):
    train_dataset = my_database(opt)
    train_dataset.load_nii_to_queue()
    train_dataset.init_train_sequence()

    model = load_model(opt)

    train_loss = my_loss()
    
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr) # need change
    # optimizer = Adan(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=opt.opt_betas, eps = opt.opt_eps, max_grad_norm=opt.max_grad_norm, no_prox=opt.no_prox)

    for nii_epoch in range(opt.model_load_set+1,opt.model_load_set+1+opt.n_epochs): #need change

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size)

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)

            inputs = inputs.to("cuda:"+opt.cuda_index)
            labels = labels.to("cuda:"+opt.cuda_index)
            
            outputs = model(inputs)

            loss = train_loss.cal_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     

            show_training_local_info(nii_epoch, train_loss)

        show_training_global_info(nii_epoch,train_loss)

        LoRA_state_dict = get_peft_model_state_dict(model)
        LoRA_state_path = f"models/fine-tune_LoRA_weight/LoRA_{opt.NICT_setting}_{opt.defect_degree}_{nii_epoch}.pkl"
        torch.save(LoRA_state_dict,LoRA_state_path)

        train_dataset.refresh_next_train()
        train_loss.clear()   

def testing_finetuned():
    ...

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    if opt.testing_mode == "slice_testing":
        slice_testing()
    elif opt.testing_mode == "volume_testing":
        volume_testing()
    elif opt.testing_mode == "fine_tuning":
        fine_tuning(opt)
    elif opt.testing_mode =="testing_finetuned":
        testing_finetuned(opt)
