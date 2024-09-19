# MITAMP
The official code of MITAMP paper - "[Imaging foundation model for universal enhancement of non-ideal measurement CT](***)". 

Non-ideal measurement computed tomography (NICT), which sacrifices optimal imaging standards for new advantages in CT imaging, is expanding the clinical application scope of CT images. However, with the reduction of imaging standards, the image quality has also been reduced, extremely limiting the clinical acceptability. We propose a **M**ulti-scale **I**ntegrated **T**ransformer **AMP**lifier (**MITAMP**), the first imaging foundation model for universal NICT enhancement. It has been pre-trained on a large-scale physical-driven simulation dataset, and is able to directly generalize to the NICT enhancement tasks with various non-ideal settings and body regions. Via the adaptation with few data, it can further achieve professional performance in real-world specific scenarios.

<p align="center"><img width="100%" src="figs/background.png" /></p>

## News
- 2024.09.18: Release the adaptation code of MITAMP.
- 2024.09.15: Release the **MITAMP** official code for universal NICT enhancement. Welcome to use and evaluate! [[Paper](***)] 

## Dependencies
- Python 3.10.11
- PyTorch 2.0.1

## Usage of the pre-trained MITAMP
## 1. Clone the repository and prepare environment
**Step 1**: clone the repository
```bash
$ git clone https://github.com/YutingHe-list/MITAMP
$ cd MITAMP/
```

**Step 2**: build a new environment
``` bash
$ conda create -n MITAMP python==3.10.13
$ conda activate MITAMP
```

**Step 3**: set up basic pytorch
``` bash
$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Step 4**: install related packages
``` bash
$ pip install "numpy<2"
$ pip install einops timm SimpleITK peft
```

**Step 5**: Install [Adan](https://github.com/sail-sg/Adan) packages by following the recommended steps.

Use `which nvcc` to locate the environment where CUDA is installed (e.g., `/usr/local/cuda/bin/nvcc`), and then modify and execute the following commands based on that path:

```bash
export CUDA_HOME=/usr/local/cuda   # Path to your environment with CUDA installed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python3 -m pip install git+https://github.com/sail-sg/Adan.git
```

**Step 6**: Install [odl]() packages by following the recommended steps.
odl

## 2. Download the pre-trained MITAMP
<!-- need added: one in paper, one for recent -->
| Weight     | Download  | Description                           |
|------------|-----------|---------------------------------------|
| MITAMP.pkl | [link](https://seunic-my.sharepoint.cn/:u:/g/personal/220232198_seu_edu_cn/EYkIR7NFZIRPoU8sMgr9A9MBKDQyEg91-43OSGLMvL4fFQ?e=HGP8ZV) | Paper used weight                     |
| MITAMP.pkl | [link](https://seunic-my.sharepoint.cn/:u:/g/personal/220232198_seu_edu_cn/EUnCvYFcAkBGlw2VnXHNrm0B4lwvpEJBNEGbfdNu5oetsg?e=hCie8L) | Latest weight (better performence)   |

Download the model checkpoint and save it to `./weights/MITAMP_weight/MITAMP.pkl`.

## 3. Universal enhancement
For your convenience, we provide two testing modes to demonstrate the universal NICT enhancement performance of MITAMP:
- [Slice testing](#31-slice-testing) enhances 2D NIFTI data with the shape [1, H, W].
- [Volume testing](#32-volume-testing) enhances 3D NIFTI data with the shape [S, H, W].

### 3.1 Slice testing

**Step 1**: We have provided testing data in the `./samples/slice_testing/input` directory, which includes NIFTI files for nine different NICT types with the shape [1, H, W]. You can also use your own data by placing it in this directory.


**Step 2**: Execute the following command to enhance the NICT slice stored in `--input_path` using MITAMP, and the enhanced slice will be saved in `--output_path`.
```bash
python inference.py --testing_mode "slice_testing" --input_path "samples/slice_testing/input/LDCT_Low.nii.gz" --output_path "samples/slice_testing/output/LDCT_Low.nii.gz" --LoRA_mode "none"
```

### 3.2 Volume testing

**Step 1**: We provide [testing data](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EoXbDCJ9XYBKhzx72KVfWWQBGeFWqbIzT0MJWUXYOSB1Ag?e=udKtLl), which includes 11 volumes for each of the nine types of NICT data with the shape [S, H, W]. You can download them or use your own data by placing it in the `./samples/volume_testing/input` directory.

**Step 2**: Execute the following command to enhance the NICT volume stored in `--input_path` using MITAMP, and the enhanced volume will be saved in `--output_path`.

```bash
python inference.py --testing_mode "volume_testing" --input_path "samples/volume_testing/input/1.nii.gz" --output_path "samples/volume_testing/output/1.nii.gz" --LoRA_mode "none"
```

##  4. Adaptation with LoRA
We provide the MITAMP-S adaptation method for specific NICT enhancement tasks in the [4.1 Model adaptation](#41-model-adaptation) section, followed by the corresponding [4.2 Slice testing](#42-slice-testing) and [4.3 Volume testing](#43-volume-testing) sections to demonstrate the performance of the adapted MITAMP-S.


### 4.1 Model adaptation
We provide training data for MITAMP adaptation, which includes forty-four volumes for each of the nine types of NICT data along with their corresponding label volumes.

**Step 1**: Download the NICT volumes from a specific NICT type folder in the [download link](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EumSyhDHuC9Fp-34pdAhmQMBFIXsXJldhbH6wfo1A40XAA?e=oJatfZ) and place them in the `./samples/adaptation/input` directory.

**Step 2**: Download the corresponding label volumes with the same name from the [download link](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EtbZHbkTlhZOtWCHcAA4YOYBCTF6InkyqDvGRtOwLpSqwg?e=mSLGcI) and place them in the `./samples/adaptation/label` directory.

**Step 3**: Execute the following command to fine-tune MITAMP-S to adapt to the specific training data located in the `"input_folder"` and `"label_folder"`, with the number of volumes set by `"training_volumes"`. The parameters `"queue_len"` and `"queue_iterate_times"` control the sampling method for the training data, consistent with the method described in the paper. The LoRA weights of MITAMP-S will be stored in the `./weights/LoRA_weight` directory.

```bash
python adaptation_LoRA.py --input_folder "samples/adaptation/input" --label_folder "samples/adaptation/label" --training_volumes 44 --queue_len 5 --queue_iterate_times 2
```

### 4.2 Slice testing
**Step 1**: We have provided testing data in the `./samples/slice_testing/input` directory, which is the same data used in the [3.1 Slice testing](#31-slice-testing) section. You can also use your own data by placing it in this directory.

**Step 2**: Execute the following command to enhance the NICT slice stored in `--input_path` using MITAMP-S, with the LoRA weight file from the epoch specified by `"LoRA_load_set"`. The enhanced slice will be saved in `--output_path`.

```bash
python inference.py --testing_mode "slice_testing" --input_path "samples/slice_testing/input/LDCT_Low.nii.gz" --output_path "samples/slice_testing/output/LDCT_Low.nii.gz" --LoRA_mode "load" --LoRA_load_set 88
```

### 4.3 Volume testing

**Step 1**: The [testing data](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EoXbDCJ9XYBKhzx72KVfWWQBGeFWqbIzT0MJWUXYOSB1Ag?e=udKtLl) is the same as that used in the [3.2 Volume testing](#32-volume-testing) section, including 11 volumes for each of the nine types of NICT data with the shape [S, H, W]. Download and place them in the `./samples/volume_testing/input` directory, or use your own data by placing it in the same directory.


**Step 2**: Execute the following command to enhance the NICT volume stored in `--input_path` using MITAMP-S, with the LoRA weight file from the epoch specified by `"LoRA_load_set"`. The enhanced volume will be saved in `--output_path`.

```bash
python inference.py --testing_mode "volume_testing" --input_path "samples/volume_testing/input/1.nii.gz" --output_path "samples/volume_testing/output/1.nii.gz" --LoRA_mode "load" --LoRA_load_set 88
```


## Acknowledgements
- We highly appreciate

## Reference
```
Waitting
```

## Ongoing
- [ ] Release SimNICT dataset with 10.6 million NICT-ICT image pairs.
- [ ] Release pre-training code of MITAMP.
- [x] Release adaptation code of MITAMP-S.
- [x] Release inference code and pretrained weights of MITAMP.
