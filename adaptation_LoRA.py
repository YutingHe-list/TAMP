import sys
import numpy as np
import torch
import argparse
from adan import Adan
from peft import get_peft_model_state_dict
from peft import LoraConfig
from peft import get_peft_model
import torch.optim.lr_scheduler as lr_scheduler

from models.network_MITNet import MITNet as my_network
from utils.MyLoss import MyLoss as my_loss
from utils.MyDataset import MyDataset as my_dataset


def get_parser():
    parser = argparse.ArgumentParser(description='MAIN FUNCTION PARSER')

    parser.add_argument('--input_folder', type=str, default="samples/adaptation/input")
    parser.add_argument('--label_folder', type=str, default="samples/adaptation/label")

    parser.add_argument('--training_volumes', type=int, default=44, help="number of training volumes")
    parser.add_argument('--queue_len', type=int, default=5)
    parser.add_argument('--queue_iterate_times', type=int, default=2, help="number of training volumes")
    parser.add_argument('--nii_start_index', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--cuda_index', type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
    parser.add_argument('--opt_betas', default=[0.98, 0.92, 0.99], type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    parser.add_argument('--no_prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')

    return parser

def show_training_global_info(nii_epoch, train_loss):
    sys.stdout.write(f"\n[Epoch {nii_epoch}] [loss {train_loss.mloss():.6f}]\n")


def show_training_local_info(nii_epoch, train_loss, batch_idx, total_batches):
    sys.stdout.write(
        f"\r{' ' * 70}" 
        f"\r[epoch:{nii_epoch}/44] [batch:{batch_idx}/{total_batches}] "
        f"[loss:{train_loss.mloss():.6f} = MSE:{train_loss.mmse():.6f} + "
        f"VGG:{train_loss.mvgg():.6f} + SSIM:{train_loss.mssim():.6f} + PROJ:{train_loss.mprom():.6f}]"
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

def load_model(opt):
    model = my_network()
    state_dict = torch.load("weights/MITAMP_weight/MITAMP.pkl")
    model.load_state_dict(state_dict)
    
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

    model.to(f"cuda:{opt.cuda_index}")    
    return model

def fine_tuning(opt):
    model = load_model(opt)
    train_dataset = my_dataset(opt)
    train_loss = my_loss()
    
    optimizer = Adan(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=opt.opt_betas, eps = opt.opt_eps, max_grad_norm=opt.max_grad_norm, no_prox=opt.no_prox)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for nii_epoch in range(opt.nii_start_index, opt.training_volumes*opt.queue_iterate_times+1):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size)
        total_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)

            inputs = inputs.to(f"cuda:{opt.cuda_index}")
            labels = labels.to(f"cuda:{opt.cuda_index}")
            
            outputs = model(inputs)

            loss = train_loss.cal_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     

            show_training_local_info(nii_epoch, train_loss, batch_idx, total_batches)

        show_training_global_info(nii_epoch,train_loss)

        LoRA_state_dict = get_peft_model_state_dict(model)
        LoRA_state_path = f"weights/MITAMP_ada_zoo/LoRA_{nii_epoch}.pkl"
        torch.save(LoRA_state_dict,LoRA_state_path)

        scheduler.step()
        train_dataset.refresh_next_train()
        train_loss.clear()   

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    fine_tuning(opt)