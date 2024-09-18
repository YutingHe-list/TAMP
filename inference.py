import os
import sys
import numpy as np
import torch
import argparse
import SimpleITK as sitk
from adan import Adan
from peft import get_peft_model_state_dict
from peft import LoraConfig
from peft import get_peft_model
from peft import set_peft_model_state_dict

from models.network_MITNet import MITNet
from utils.MyLoss import MyLoss as my_loss
from utils.MyDataset import MyDataset as my_dataset

def get_parser():
    parser = argparse.ArgumentParser(description='MAIN FUNCTION PARSER')
    parser.add_argument('--testing_mode', type=str, default="slice_testing", help="slice_testing, volume_testing")
    parser.add_argument('--LoRA_mode', type=str, default="get", help="none, load") 

    parser.add_argument('--NICT_setting', type=str, default="LDCT", help="LDCT, LACT, SVCT")
    parser.add_argument('--defect_degree', type=str, default="Low", help="Low, Mid, High")

    parser.add_argument('--training_volumes', type=int, default=44)
    parser.add_argument('--nii_start_index', type=int, default=1)
    parser.add_argument('--LoRA_load_set', type=int, default=44)
    parser.add_argument('--queue_len', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--cuda_index', type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
    parser.add_argument('--opt_betas', default=[0.98, 0.92, 0.99], type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    parser.add_argument('--no_prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')

    return parser

def show_testing_local_info(input_nii_set,nii_folder,i,S):
    sys.stdout.write(
        f"\rinfering [{input_nii_set}.nii.gz] in {nii_folder}: {i}/{S}"
    )

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

def package_nii(modified_array,input_file_address,output_file_address):
    image = sitk.ReadImage(input_file_address)
    modified_image = sitk.GetImageFromArray(modified_array)
    modified_image.CopyInformation(image)
    sitk.WriteImage(modified_image, output_file_address)

def load_model(opt):
    state_dict = torch.load("weights/MITAMP_weight/MITAMP.pkl")
    model = MITNet()
    model.load_state_dict(state_dict)
    
    if opt.LoRA_mode == "load":
        with open('LoRA_path.txt', 'r', encoding='utf-8') as file:
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
        
        lora_path = f"models/LoRA_weight/{opt.NICT_setting}_{opt.defect_degree}/LoRA_{opt.LoRA_load_set}.pkl"
        lora_state_dict = torch.load(lora_path,map_location="cpu")
        set_peft_model_state_dict(model,lora_state_dict)

    model.to(f"cuda:{opt.cuda_index}")    
    return model

def slice_testing(opt):
    model = load_model(opt)

    slice_testing_data_folder = "samples/slice_testing/input"
    
    for input_path in input_paths:
        input_image = sitk.ReadImage(input_path)
        input = sitk.GetArrayFromImage(input_image)        
        input_tensor = torch.tensor(standard(input), dtype=torch.float).unsqueeze(0).to(f"cuda:{opt.cuda_index}")

        output_tensor = model(input_tensor)

        output = unstandard(np.array(output_tensor[0].cpu().detach())).astype('int16')
        output_path = input_path.replace("input", "output")
        output = sitk.GetImageFromArray(output)
        output.CopyInformation(input_image)
        sitk.WriteImage(output, output_path)


def volume_testing(opt):
    model = load_model(opt)

    volume_test_data_folder = "samples/volume_testing/input"

    if opt.LoRA_mode == "none": # 3.2
        settings = ["LDCT", "LACT", "SVCT"]
        degrees = ["Low", "Mid", "High"]
        nii_folders=[]
        for setting in settings:
            for degree in degrees:
                nii_folders.append(f"{volume_test_data_folder}/{setting}_{degree}")

    elif opt.LoRA_mode == "load":   # 4.3
        nii_folders = [f"{volume_test_data_folder}/{opt.NICT_setting}_{opt.defect_degree}"]

    for nii_folder in nii_folders:
        for input_nii_set in range(1,2):
            input_nii_path = f"{nii_folder}/{input_nii_set}.nii.gz"
            input_nii_image = sitk.ReadImage(input_nii_path)
            input_nii_file =  sitk.GetArrayFromImage(input_nii_image)

            input_nii_tensor = torch.tensor(standard(input_nii_file), dtype=torch.float).to(f"cuda:{opt.cuda_index}")
            S,H,W=input_nii_file.shape
            output_nii_file = np.zeros((S,H,W),dtype=np.int16)

            for i in range(S):
                input_slice_tensor = input_nii_tensor[i].unsqueeze(0).unsqueeze(0)
                output_nii_tensor = model(input_slice_tensor)            
                output = unstandard(np.array(output_nii_tensor.cpu().detach())).astype('int16')
                output_nii_file[i] = output[0][0]
                show_testing_local_info(input_nii_set,nii_folder,i,S)

            output_nii_image = sitk.GetImageFromArray(output_nii_file)
            output_nii_image.CopyInformation(input_nii_image)
            output_nii_path=input_nii_path.replace("input", "output")
            sitk.WriteImage(output_nii_image, output_nii_path)

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    
    if opt.testing_mode == "slice_testing":
        slice_testing(opt)
    elif opt.testing_mode == "volume_testing":
        volume_testing(opt)