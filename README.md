<!-- # :dragon_face: TAMP - An imaging foundation model -->
[![header](https://capsule-render.vercel.app/api?type=soft&height=90&color=auto&text=TAMP%20-%20%20An%20Imaging%20Foundation%20Model&descAlign=50&descAlignY=76&descSize=19&fontSize=37&textBg=false&fontAlign=50&section=header&reversal=false&desc=Imaging%20foundation%20model%20for%20universal%20enhancement%20of%20non-ideal%20measurement%20CT&fontAlignY=38&strokeWidth=0)](https://arxiv.org/abs/2410.01591)
--- 
[![Paper](https://img.shields.io/badge/TAMP_paper-Arxiv-purple)](https://arxiv.org/abs/2410.01591)
[![Model Zoo](https://img.shields.io/badge/Model_zoo-processing-blue)](https://github.com/YutingHe-list/TAMP/blob/main/document/Model_zoo.md)
[![SimNICT dataset](https://img.shields.io/badge/SimNICT_dataset-processing-green)](https://huggingface.co/datasets/YutingHe-list/SimNICT)

:loudspeaker: **TAMP** paper - **[Imaging foundation model for universal enhancement of non-ideal measurement CT.](https://arxiv.org/abs/2410.01591)** <br/> 
Yuxin Liu*, [Rongjun Ge*](https://scholar.google.com/citations?user=v8K8HIkAAAAJ&hl=en), [Yuting He#](https://yutinghe-list.github.io/), Zhan Wu, [Chenyu You](https://chenyuyou.me/), [Shuo Li](https://engineering.case.edu/about/school-directory/shuo-li), Yang Chen#. <br/>
_*means equal contribution, #means [corresponding](mailto:ythe1995@163.com) author._

## News
- 2024.10.03: **TAMP** has been released! Welcome to use! [[Paper](https://arxiv.org/abs/2410.01591)]
- 2024.09.25: Open a [TAMP-adapted Model Zoo](https://github.com/YutingHe-list/TAMP/blob/main/document/Model_zoo.md) to release the adapted TAMP in different downstream tasks.

## Brief introduction 
Non-ideal measurement computed tomography (NICT), which sacrifices optimal imaging standards for new advantages in CT imaging, is expanding the clinical application scope of CT images. However, with the reduction of imaging standards, the image quality has also been reduced, extremely limiting the clinical acceptability. We propose a multi-scale integrated **T**ransformer **AMP**lifier (**TAMP**), the first imaging foundation model for universal NICT enhancement. It has been pre-trained on a large-scale physical-driven simulation dataset, and is able to directly generalize to the NICT enhancement tasks with various non-ideal settings and body regions. Via the adaptation with few data, it can further achieve professional performance in real-world specific scenarios.
<p align="center"><img width="100%" src="figs/background.png" />

## [TAMP-adapted Model Zoo](https://github.com/YutingHe-list/TAMP/blob/main/document/Model_zoo.md)
We are hosting a [Model Zoo](https://github.com/YutingHe-list/TAMP/blob/main/document/Model_zoo.md) to release the adapted TAMP in different downstream tasks.

- You can try to find an adapted TAMP that meets the requirements of your target task, and it will perform better.
- If you want to contribute to the Model Zoo, please send the [EMAIL](mailto:ythe1995@163.com) to our group.

## Demo
https://github.com/user-attachments/assets/fe01975c-7956-4b4d-aa1d-813623dcee01
<!-- ## Acknowledgements
- We highly appreciate -->

## Reference
```
@misc{liu2024imagingfoundationmodeluniversal,
      title={Imaging foundation model for universal enhancement of non-ideal measurement CT}, 
      author={Yuxin Liu and Rongjun Ge and Yuting He and Zhan Wu and Chenyu You and Shuo Li and Yang Chen},
      year={2024},
      eprint={2410.01591},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.01591}, 
}
```

## Ongoing
- [ ] Release SimNICT dataset with 10.9 million NICT-ICT image pairs.
- [ ] Release pre-training code of TAMP.
- [ ] TAMP Toolbox on 3D Slicer.
- [x] Open a TAMP-adapted Model Zoo.
- [x] Release adaptation code of TAMP-S.
- [x] Release inference code and pretrained weights of TAMP.

# :running: Playground: Quick start TAMP
- [ ] You can try the TAMP via the [Jupeter Notebook](). 
- [x] You can implement the TAMP on your server via the following operations.

## 1. Clone the repository and prepare environment

### Dependencies
- Python 3.10.11
- PyTorch 2.0.1
  
**Step 1**: Clone the repository
```bash
git clone https://github.com/YutingHe-list/TAMP
cd TAMP/
pip install -r requirements.txt
```

**Step 2**: Install the [Adan](https://github.com/sail-sg/Adan) and [ODL](https://github.com/odlgroup/odl) packages by following the recommended procedures.

- **Adan**

Use `which nvcc` to locate the environment where CUDA is installed (e.g., `/usr/local/cuda/bin/nvcc`), and then modify and execute the following commands based on that path:

```bash
export CUDA_HOME=/usr/local/cuda   # Path to your environment with CUDA installed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python3 -m pip install git+https://github.com/sail-sg/Adan.git
```

- **ODL**

Install the ODL package by following the steps below:
```bash
pip install odl
```
Next, clone the ODL repository and overwrite the contents of the odl folder in your TAMP environment (.e.g, `/home/xytc/anaconda3/envs/TAMP/lib/python3.10/site-packages/odl`)with the files from the `odl/odl` folder in the cloned repository.

## 2. Download the pre-trained TAMP
<!-- need added: one in paper, one for recent -->
| Weight     | Download  | Description                           |
|------------|-----------|---------------------------------------|
| TAMP_pretrain.pkl | [link](https://seunic-my.sharepoint.cn/:u:/g/personal/220232198_seu_edu_cn/EYkIR7NFZIRPoU8sMgr9A9MBKDQyEg91-43OSGLMvL4fFQ?e=2xsa3w) |Pre-trained universal NICT enhancement model|

Download the model checkpoint and save it to `./weights/TAMP_pretrain_weight/TAMP_pretrain.pkl`.

## Option: A simple dataset for quick start
We have provided a [simple simulation-based dataset](https://seunic-my.sharepoint.cn/personal/220232198_seu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F220232198%5Fseu%5Fedu%5Fcn%2FDocuments%2FTAMP%2FSimNICT%2DAMOS%2DSimple&ga=1) form a part of [AMOS](https://amos22.grand-challenge.org/) dataset for quick start. If you want to test the TAMP on NICT volumes or try the adaptation, you can download the testing data and the training data with different NICT settings in specific folds.

## 3. Universal enhancement
For your convenience, we provide two testing modes to demonstrate the universal NICT enhancement performance of TAMP:
- [Slice testing](#31-slice-testing) enhances 2D NIFTI data with the shape [1, H, W].
- [Volume testing](#32-volume-testing) enhances 3D NIFTI data with the shape [S, H, W].

### 3.1 Slice testing

**Step 1**: We have provided testing data in the `./samples/slice_testing/input` directory with the shape [1, H, W]. You can also use your own data by placing it in this directory. 

**Step 2**: To enhance **a single NICT slice file** specified by `--input_path` using TAMP, execute the following command. The enhanced slice file will be saved at `--output_path`.

```bash
python inference.py --testing_mode "single_slice" --input_path "samples/slice_testing/input/LDCT_Low.nii.gz" --output_path "samples/slice_testing/output/LDCT_Low.nii.gz" --LoRA_mode "none"
```

To enhance **all NICT slice files** in the `--input_folder` directory using TAMP, execute the following command. The enhanced slice files will be saved in `--output_folder` with the same name.

```bash
python inference.py --testing_mode "group_slice" --input_folder "samples/slice_testing/input" --output_folder "samples/slice_testing/output" --LoRA_mode "none"
```

### 3.2 Volume testing
**Step 1**: You can put the [testing data](#option-a-simple-dataset-for-quick-start) with the shape [S, H, W] or use your own data by placing it in the `./samples/volume_testing/input` directory. 

**Step 2**: To enhance **a single NICT volume file** specified by `--input_path` using TAMP, execute the following command. The enhanced volume will be saved at `--output_path`.

```bash
python inference.py --testing_mode "single_volume" --input_path "samples/volume_testing/input/1.nii.gz" --output_path "samples/volume_testing/output/1.nii.gz" --LoRA_mode "none"
```

To enhance **all NICT volume files** in the `--input_folder` directory using TAMP, execute the following command. The enhanced volume files will be saved in `--output_folder` with the same name.

```bash
python inference.py --testing_mode "group_slice" --input_folder "samples/volume_testing/input" --output_folder "samples/volume_testing/output" --LoRA_mode "none"
```

##  4. Adaptation with LoRA
We provide the TAMP-S adaptation method for specific NICT enhancement tasks in the [4.1 Model adaptation](#41-model-adaptation) section, followed by the corresponding [4.2 Slice testing](#42-slice-testing) and [4.3 Volume testing](#43-volume-testing) sections to demonstrate the performance of the adapted TAMP-S.


### 4.1 Model adaptation
**Step 1**: We provide [training data](#option-a-simple-dataset-for-quick-start) for TAMP adaptation. Download the NICT volumes from a specific NICT type folder and place them in the `./samples/adaptation/input` directory. Then, download the corresponding label volumes with the same name and place them in the `./samples/adaptation/label` directory.

**Step 2**: Execute the following command to fine-tune TAMP-S to adapt to the specific training data located in the `"input_folder"` and `"label_folder"`, with the number of volumes set by `"training_volumes"`. The parameters `"queue_len"` and `"queue_iterate_times"` control the sampling method for the training data, consistent with the method described in the paper. The LoRA weights of TAMP-S will be stored in the `./weights/LoRA_weight` directory.

```bash
python adaptation_LoRA.py --input_folder "samples/adaptation/input" --label_folder "samples/adaptation/label" --training_volumes 44 --queue_len 5 --queue_iterate_times 2
```

### 4.2 Slice testing
**Step 1**: We have provided testing data in the `./samples/slice_testing/input` directory, which is the same data used in the [3.1 Slice testing](#31-slice-testing) section. You can also use your own data by placing it in this directory. 

**Step 2**: To enhance **a single NICT slice file** specified by `--input_path` using TAMP-S, execute the following command. The LoRA weight file is specified by `"LoRA_path"`, and the enhanced slice file will be saved at `--output_path`.

```bash
python inference.py --testing_mode "single_slice" --input_path "samples/slice_testing/input/LDCT_Low.nii.gz" --output_path "samples/slice_testing/output/LDCT_Low.nii.gz" --LoRA_mode "load" --LoRA_path "weights/TAMP_adaptation_weight/LoRA_88.pkl"
```

To enhance **all NICT slice files** in the `--input_folder` directory using TAMP-S, execute the following command. The enhanced slice files will be saved in `--output_folder` with the same name.

```bash
python inference.py --testing_mode "group_slice" --input_folder "samples/slice_testing/input" --output_folder "samples/slice_testing/output" --LoRA_mode "load" --LoRA_path "weights/TAMP_adaptation_weight/LoRA_88.pkl"
```

### 4.3 Volume testing

**Step 1**: The [testing data](#option-a-simple-dataset-for-quick-start) with the shape [S, H, W] is the same as that used in the [3.2 Volume testing](#32-volume-testing) section. Download and place them in the `./samples/volume_testing/input` directory, or use your own data by placing it in the same directory.

**Step 2**: To enhance **a NICT volume file** specified by `--input_path` using TAMP-S, execute the following command. The LoRA weight file is specified by `"LoRA_path"`, and the enhanced volume file will be saved at `--output_path`.

```bash
python inference.py --testing_mode "single_volume" --input_path "samples/volume_testing/input/1.nii.gz" --output_path "samples/volume_testing/output/1.nii.gz" --LoRA_mode "load" --LoRA_path "weights/TAMP_adaptation_weight/LoRA_88.pkl"
```

To enhance **all NICT volume files** in the `--input_folder` directory using TAMP-S, execute the following command. The enhanced volume files will be saved in `--output_folder` with the same name.

```bash
python inference.py --testing_mode "group_volume" --input_folder "samples/volume_testing/input" --output_folder "samples/volume_testing/output" --LoRA_mode "load" --LoRA_path "weights/TAMP_adaptation_weight/LoRA_88.pkl"
```

