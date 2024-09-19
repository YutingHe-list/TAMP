import sys
import os
import numpy as np
import torch
import argparse
import SimpleITK as sitk
from peft import LoraConfig
from peft import get_peft_model
from peft import set_peft_model_state_dict

from models.network_MITNet import MITNet

def get_parser():
    parser = argparse.ArgumentParser(description='MAIN FUNCTION PARSER')
    parser.add_argument('--testing_mode', type=str, default="group_slice", help="single_slice, group_slice, single_volume, group_volume")
    parser.add_argument('--LoRA_mode', type=str, default="none", help="none, load") 

    parser.add_argument('--NICT_setting', type=str, default="LDCT", help="LDCT, LACT, SVCT")
    parser.add_argument('--defect_degree', type=str, default="Low", help="Low, Mid, High")

    parser.add_argument('--input_folder', type=str, default="samples/volume_testing/input")
    parser.add_argument('--input_folder', type=str, default="samples/volume_testing/input")
    parser.add_argument('--input_path', type=str, default="samples/volume_testing/input/1.nii.gz")
    parser.add_argument('--output_path', type=str, default="samples/volume_testing/output/1.nii.gz")

    parser.add_argument('--training_volumes', type=int, default=44)
    parser.add_argument('--nii_start_index', type=int, default=1)
    parser.add_argument('--LoRA_path', type=str, default="weights/MITAMP_adaptation_weight/LoRA_1.pkl")
    parser.add_argument('--queue_len', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--cuda_index', type=int, default=3)

    parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
    parser.add_argument('--opt_betas', default=[0.98, 0.92, 0.99], type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    parser.add_argument('--no_prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')

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
    state_dict = torch.load("weights/MITAMP_pretrain_weight/MITAMP_pretrain.pkl")
    model = MITNet()
    model.load_state_dict(state_dict)
    
    if opt.LoRA_mode == "load":
        with open('utils/LoRA_path.txt', 'r', encoding='utf-8') as file:
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
        
        lora_state_dict = torch.load(opt.LoRA_path,map_location="cpu")
        set_peft_model_state_dict(model,lora_state_dict)

    model.to(f"cuda:{opt.cuda_index}")    
    model.eval()
    return model

def single_slice(opt):
    model = load_model(opt)
    
    input_image = sitk.ReadImage(opt.input_path)
    input = sitk.GetArrayFromImage(input_image)        
    input_tensor = torch.tensor(standard(input), dtype=torch.float).unsqueeze(0).to(f"cuda:{opt.cuda_index}")

    output_tensor = model(input_tensor)

    output = unstandard((output_tensor[0].cpu().detach()).numpy()).astype('int16')
    output = sitk.GetImageFromArray(output)
    output.CopyInformation(input_image)
    sitk.WriteImage(output, opt.output_path)

def single_volume(opt):
    model = load_model(opt)

    input_nii_image = sitk.ReadImage(opt.input_path)
    input_nii_file =  sitk.GetArrayFromImage(input_nii_image)

    input_nii_tensor = torch.tensor(standard(input_nii_file), dtype=torch.float).to(f"cuda:{opt.cuda_index}")
    S,H,W=input_nii_file.shape
    output_nii_file = np.zeros((S,H,W),dtype=np.int16)

    for i in range(S):
        input_slice_tensor = input_nii_tensor[i].unsqueeze(0).unsqueeze(0)
        output_nii_tensor = model(input_slice_tensor)            
        output = unstandard(np.array(output_nii_tensor.cpu().detach())).astype('int16')
        output_nii_file[i] = output[0][0]
        sys.stdout.write(f"\rinfering {opt.input_path}: {i}/{S}")

    output_nii_image = sitk.GetImageFromArray(output_nii_file)
    output_nii_image.CopyInformation(input_nii_image)
    sitk.WriteImage(output_nii_image, opt.output_path)

def group_slice(opt):
    model = load_model(opt)
    input_files = os.listdir(opt.input_folder)
    input_files = [f for f in input_files if f.endswith('.nii.gz')]

    for input_file in input_files:
        input_path = os.path.join(opt.input_folder, input_file)
        output_path = os.path.join(opt.output_folder, input_file)
        input_image = sitk.ReadImage(input_path)
        input = sitk.GetArrayFromImage(input_image)        
        input_tensor = torch.tensor(standard(input), dtype=torch.float).unsqueeze(0).to(f"cuda:{opt.cuda_index}")

        output_tensor = model(input_tensor)

        output = unstandard((output_tensor[0].cpu().detach()).numpy()).astype('int16')
        output = sitk.GetImageFromArray(output)
        output.CopyInformation(input_image)
        sitk.WriteImage(output, output_path)

def group_volume(opt):
    model = load_model(opt)
    input_files = os.listdir(opt.input_folder)
    input_files = [f for f in input_files if f.endswith('.nii.gz')]

    for input_file in input_files:
        input_path = os.path.join(opt.input_folder, input_file)
        output_path = os.path.join(opt.output_folder, input_file)
        input_nii_image = sitk.ReadImage(input_path)
        input_nii_file =  sitk.GetArrayFromImage(input_nii_image)

        input_nii_tensor = torch.tensor(standard(input_nii_file), dtype=torch.float).to(f"cuda:{opt.cuda_index}")
        S,H,W=input_nii_file.shape
        output_nii_file = np.zeros((S,H,W),dtype=np.int16)

        for i in range(S):
            input_slice_tensor = input_nii_tensor[i].unsqueeze(0).unsqueeze(0)
            output_nii_tensor = model(input_slice_tensor)            
            output = unstandard(np.array(output_nii_tensor.cpu().detach())).astype('int16')
            output_nii_file[i] = output[0][0]
            sys.stdout.write(f"\rinfering {input_file}: {i}/{S}")

        output_nii_image = sitk.GetImageFromArray(output_nii_file)
        output_nii_image.CopyInformation(input_nii_image)
        sitk.WriteImage(output_nii_image, output_path)

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    
    if opt.testing_mode == "single_slice":
        single_slice(opt)
    elif opt.testing_mode == "single_volume":
        single_volume(opt)
    elif opt.testing_mode == "group_slice":
        group_slice(opt)
    elif opt.testing_mode == "group_volume":
        group_volume(opt)