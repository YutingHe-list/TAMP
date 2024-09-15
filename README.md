# MITAMP
The official code of MITAMP paper - "[Imaging foundation model for universal enhancement of non-ideal measurement CT](***)". 

Non-ideal measurement computed tomography (NICT), which sacrifices optimal imaging standards for new advantages in CT imaging, is expanding the clinical application scope of CT images. However, with the reduction of imaging standards, the image quality has also been reduced, extremely limiting the clinical acceptability. We propose a **M**ulti-scale **I**ntegrated **T**ransformer **AMP**lifier (**MITAMP**), the first imaging foundation model for universal NICT enhancement. It has been pre-trained on a large-scale physical-driven simulation dataset, and is able to directly generalize to the NICT enhancement tasks with various non-ideal settings and body regions. Via the adaptation with few data, it can further achieve professional performance in real-world specific scenarios.

<p align="center"><img width="100%" src="figs/background.png" /></p>

## News
- 2024.09.15: Release a large-scale simulated NICT dataset, **SimNICT**, with 3.6 million image pairs. [[Dataset](https://huggingface.co/datasets/YutingHe-list/SimNICT)]
- 2024.09.15: Release the **MITAMP** official code for universal NICT enhancement. Welcome to use and evaluate! [[Paper](***)] 

## Dependecies
- Python 3.10.11
- PyTorch 2.0.1

## Usage of the pre-trained Models MITAMP
## 1. Clone the repository
<!-- 这里还要准备个最简单的环境 -->
```bash
$ git clone https://github.com/YutingHe-list/MITAMP
$ cd 
$ pip install -r requirements.txt
```

## 2. Download the pre-trained Models MITAMP
Download the [model checkpoint](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EoVQNIe_E0RAnfDCJXsplHoBDe9kbq-3eAbRBMlliMjxTA?e=ccZe8r) and save it to `./models/foundation_model_weight/MITAMP.pkl`.

## 3. Foundation Model Testing
For your convenience, we provide two testing modes to show the universal NICT enhancement performance of MITAMP. 
- To gain a quick understanding, the [Slice testing](#31-slice-testing) is recommended. 
- More test data is provided in the [Volume testing](#32-volume-testing). 

### 3.1 Slice testing
Nine types of NICT images are provided in the `./samples/slice_test/input` directory for a simple test of MITAMP's universal enhancement capability. Execute the following command. The MITAMP-enhanced output will be stored in `./samples/slice_test/output`
```bash
python main.py --testing_mode "slice_testing" --LoRA_mode "none"
```

### 3.2 Volume testing
We have provided additional NICT data for testing. 

**Step 1**: Download the [testing data](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EoXbDCJ9XYBKhzx72KVfWWQBGeFWqbIzT0MJWUXYOSB1Ag?e=QXqJ66), which includes 11 volumes for each of the 9 types of NICT data, and place them in the `./samples/volume_test/input` directory.

**Step 2**: Execute the following command. The MITAMP-enhanced output will be stored in `./samples/volume_test/output`
```bash
python main.py --testing_mode "volume_test" --LoRA_mode "none"
```

##  4. Model Fine-tuning and testing
We also provide the MITAMP-S adaptation code for specific NICT enhancement tasks using fine-tuning data in the [Model fine-tuning](#41-model-fine-tuning) section, followed with the corresponding [Slice testing](#42-slice-testing) and [Volume testing](#43-volume-testing).


### 4.1 Model fine-tuning
**Step 1**: Download the [fine-tuning data](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EuhW8PS-H2ZApQdw9odb-5MB96Q-XZw4N3JGhK3q7ZIc2A?e=k4rlON), which includes 44 volumes for each of the 9 types of NICT data , and place them in the `./samples/fine-tuning_test` directory. 

**Step 2**: Execute the following command to fine-tune MITAMP-S on a specific category of NICT data, where the parameter 'NICT_setting' can be set to 'LDCT', 'LACT', or 'SVCT', and the parameter 'defect_degree' can be set to 'Low', 'Mid', or 'High'. The fine-tuned LoRA weights will be stored in the corresponding `./models/fine-tune_LoRA_weight` directory. 
```bash
python main.py --testing_mode "fine_tuning" --NICT_setting "LDCT" --defect_degree "Low" --LoRA_mode "get"
```

### 4.2 Slice testing

Load and test the performance of MITAMP-S on its corresponding task with the following command. The output will be stored in `./samples/volume_test/output`. The [testing data](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EoXbDCJ9XYBKhzx72KVfWWQBGeFWqbIzT0MJWUXYOSB1Ag?e=QXqJ66) is the same as in Volume testing, so there is no need to download it again. 
```bash
python main.py --testing_mode "slice_testing" --NICT_setting "LDCT" --defect_degree "Low" --LoRA_mode "load"
```

### 4.3 Volume testing
```bash
python main.py --testing_mode "volume_testing" --NICT_setting "LDCT" --defect_degree "Low" --LoRA_mode "load"
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
