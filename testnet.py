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
    # state_dict = torch.load(path,map_location="cpu")    
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     namekey = k[7:] if k.startswith('module.') else k
    #     new_state_dict[namekey] = v
    # model.load_state_dict(new_state_dict)

    model = MITNet()    
    model.load_state_dict(state_dict).to("cuda:0").eval()

    simple_test_data_folder = "testsets/simple_test/"
    settings = ["LDCT", "LACT", "SVCT"]
    degrees = ["Low", "Mid", "High"]
    input_paths = [
        f"{simple_test_data_folder}{setting}_{degree}_input.nii.gz"
        for setting in settings for degree in degrees
    ]

    for input_path in input_paths:
        input = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
        input = standard(input)
        input_tensor = torch.tensor(input, dtype=torch.float).unsqueeze(0).unsqueeze(0).to("cuda:0")
        output_tensor = model(input_tensor)
        output = unstandard(np.array(output_tensor[0,0].cpu().detach()).astype('float32'))



    






def test_one_image(model,input,label,trunc_min=-1024,trunc_max=3072,cuda_set="cuda:0",loss_fn_alex=None):
    # input(512,512)[-1024,3072]
    # output(512,512)[-1024,3072]

    standard_input_tensor = torch.tensor(standard(input), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(cuda_set)
   
    standard_output_tensor = model(standard_input_tensor)

    # output_tensor = unstandard(output_tensor)
    standard_output = np.array(standard_output_tensor[0,0].cpu().detach()).astype('float32')
    output = unstandard(standard_output)


    input_tensor = torch.tensor(input, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(cuda_set)
    output_tensor = torch.tensor(output, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(cuda_set)
    label_tensor = torch.tensor(label, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(cuda_set)

    data_range = trunc_max - trunc_min
    if loss_fn_alex==None:
        loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(cuda_set)
    original_result, pred_result = compute_measure(input_tensor, label_tensor, output_tensor, data_range,loss_fn_alex)

    return original_result, pred_result, output

# 将Hu值image转为png格式
def get_image_for_show(x,WINDOW_CENTER,WINDOW_WIDTH):   
    BOUND_MIN, BOUND_MAX = WINDOW_CENTER-WINDOW_WIDTH//2,WINDOW_CENTER+WINDOW_WIDTH//2
    x[x>BOUND_MAX] = BOUND_MAX
    x[x<BOUND_MIN] = BOUND_MIN
    x = (x-BOUND_MIN)/(BOUND_MAX-BOUND_MIN)

    x_slice = np.expand_dims(x, axis=-1)
    x = np.concatenate([x_slice, x_slice, x_slice], axis=-1)
    x = (x * 255).astype(np.uint8)
    return x

# 获得局部放大视图
def local_image_zooming(img,j=0.2,i=0.8,h=80,w=80):
    i = int(i*512)
    j = int(j*512)
    image_with_box = img.copy()

    patch1 = img[i-h//2:i+h//2, j-w//2:j+w//2]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
    
    pt1 = (j-w//2, i-h//2)  # 长方形框左上角坐标
    pt2 = (j+w//2, i+h//2)  # 长方形框右下角坐标 

    cv2.rectangle(image_with_box, pt1, pt2, (255, 0, 0), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
    return image_with_box,patch1

def save_fig(x, y, pred, fig_address, original_result, pred_result,local_x,local_y,local_pred,model_name=None):
    # 将x、y、pred以及评价指标、局部放大 整合在一张图像并保存
    f, ax = plt.subplots(2, 4, figsize=(30, 10))
    f.suptitle(model_name, fontsize=30)    
    ax[0][0].set_title('Input', fontsize=30)
    ax[0][0].imshow(x)
    ax[1][0].imshow(local_x)
    ax[0][1].set_title('Output', fontsize=30)
    ax[0][1].imshow(pred)    
    ax[1][1].imshow(local_pred)
    ax[0][2].set_title('Ground Truth', fontsize=30)
    ax[0][2].imshow(y)
    ax[1][2].imshow(local_y)
    
    evaluation="Input Evaluation \n    PSNR: {:.4f}\n    SSIM: {:.4f}\n    RMSE: {:.4f}\n\n \
Output Evaluation \n    PSNR: {:.4f}\n    SSIM: {:.4f}\n    RMSE: {:.4f}\n\n".format(
                    original_result[0],original_result[1],original_result[2],
                    pred_result[0],pred_result[1],pred_result[2])

    ax[0][3].text(0.35,0,evaluation, fontsize=20,horizontalalignment='left' )
    for a_list in ax:
        for a in a_list:
            a.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.69)
    f.savefig(fig_address,bbox_inches='tight')
    plt.close()

    # 裁剪图片到合适尺寸
    with Image.open(fig_address) as img:
        width, height = img.size
        crop_area = (0, 0, width - width * 0.3, height)
        cropped_img = img.crop(crop_area)
        cropped_img.save(fig_address)

def test_show_one_image(model,input,label,fig_address,WINDOW_CENTER=40,WINDOW_WIDTH=400,xper=0.5,yper=0.5,cuda_set="cuda:0"):
    # input(512,512)[-1024,3072]
    # output(512,512)[-1024,3072]

    original_result, pred_result, output = test_one_image(model,input,label,-1024,3027,cuda_set)

    input = get_image_for_show(input,WINDOW_CENTER,WINDOW_WIDTH)
    label = get_image_for_show(label,WINDOW_CENTER,WINDOW_WIDTH)
    output = get_image_for_show(output,WINDOW_CENTER,WINDOW_WIDTH)
    input,local_input=local_image_zooming(input,xper,yper)
    label,local_label=local_image_zooming(label,xper,yper)
    output,local_output=local_image_zooming(output,xper,yper)

    save_fig(input, label, output, fig_address, original_result, pred_result,
             local_input,local_label,local_output,model_name=None)

def get_model(weight_address,cuda_set="cuda:1"):    
    print("\n正在导入预训练权重" )

    # 增加旁路的LoRa
    with open('lora_path.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    modules_list = content.strip().replace("'", "").split(',\n')
    target_modules_txt = []
    for module in modules_list:
        target_modules_txt.append(module)
        
    from peft import LoraConfig
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules_txt,
        modules_to_save=["stage_2.conv_easy"],
    )

    model = MyNet()   

    from peft import get_peft_model

    path = "/public/home/grj_test/CompareExperiment/code/mynet/saved_models/mynet_14_7242.pkl"
    state_dict = torch.load(path,map_location="cpu")    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v
    model.load_state_dict(new_state_dict)
    model = get_peft_model(model, config)

    lora_path = weight_address
    lora_state_dict = torch.load(lora_path,map_location="cpu")    
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model,lora_state_dict)

    model = model.to(cuda_set).eval()  # 将模型设置为评c估模式
    return model


def test_all_specialize_weight():
    # 测试FBPConvNet各专家模型

    # 遍历所有的DATABASE,TASK,SETUP
    # task_names = ["5slice","1subject","5subject","20subject"]
    task_names = ["20subject"]
    dataset_name = "AMOS"
    distort_names = ["LD", "SpV"]
    distort_setups = [[20, 60], [60]]

    cuda_set = "cuda:4"
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(cuda_set)

    for task_name in task_names:
        for distort_name, setup_list in zip(distort_names, distort_setups):
            for distort_setup in setup_list:
	            # 	挑选最优权重，加载模型
                sys.stdout.write(f"\n{task_name},{distort_name},{distort_setup}\n")
                folder = f"/public/home/grj_test/CompareExperiment/sample/MITNet/{task_name}/{dataset_name}-{distort_name}-{distort_setup}"
                best_weight_index = get_best_weight_index(f"{folder}/log.txt")
                if best_weight_index == None:
                    continue
                weight_address = f"{folder}/saved_models/mynet_14_{best_weight_index}.pkl"
                model = get_model(weight_address,cuda_set)
                
                evaluation_txt_path = f"{folder}/evaluation.txt"
                with open(evaluation_txt_path,'w') as txt_file:
                    ...
                
	            # 	遍历改场景下所有的测试nii(后10个）：
                test_sets = [121, 434, 798, 826, 950, 1353, 1360, 1403, 1412, 1474, 1512, 1537, 1586, 1592, 1647, 1659, 1666, 1673, 1678, 1756]
                data_base_path = "/public/home/grj_test/liuyuxin/data"
                for nii_set in test_sets:
	                # 	将此nii下所有slice的测试结果记录在文件夹下txt内：input_set,slice,psnr,rmse,ssim
                    all_slice, all_inpsnr, all_inssim, all_inrmse, all_inlpips, all_outpsnr, all_outssim, all_outrmse, all_outlpips=0,0,0,0,0,0,0,0,0
                    input_nii_path = f"{data_base_path}/simulate/{distort_name}/{distort_setup}/{dataset_name}/inputs_{nii_set}.nii.gz"
                    label_nii_path = f"{data_base_path}/labels_{nii_set}.nii.gz"
                    input_nii_file = sitk.GetArrayFromImage(sitk.ReadImage(input_nii_path))
                    label_nii_file = sitk.GetArrayFromImage(sitk.ReadImage(label_nii_path))
                    S,_,_ = input_nii_file.shape
                    all_slice += S 

                    with open(evaluation_txt_path,'a') as txt_file:
                        for slice in range(S):
                            sys.stdout.write(f"\r{dataset_name},{distort_name},{distort_setup},{nii_set},{slice}/{S}")
                            input = input_nii_file[slice]
                            label = label_nii_file[slice]
                            original_result, pred_result, _ = test_one_image(model,input,label,-1024,3027,cuda_set,loss_fn_alex)
                            (in_psnr, in_ssim, in_rmse, in_lpips), (out_psnr, out_ssim, out_rmse, out_lpips) = original_result, pred_result
                            txt_file.write(f"set:{nii_set}\tslice:{slice}/{S}\toutput:[psnr:{out_psnr}\tssim:{out_ssim}\trmse:{out_rmse}\tlpips:{out_lpips}]\tinput:[psnr:{in_psnr}\tssim:{in_ssim}\trmse:{in_rmse}\tlpips:{in_lpips}]\n")
                            all_inpsnr, all_inssim, all_inrmse, all_inlpips, all_outpsnr, all_outssim, all_outrmse, all_outlpips = (all_inpsnr + in_psnr, all_inssim + in_ssim, all_inrmse + in_rmse, all_inlpips+in_lpips, all_outpsnr + out_psnr, all_outssim + out_ssim, all_outrmse + out_rmse, all_outlpips+out_lpips)


