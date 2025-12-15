# :monkey: TAMP-adapted Model Zoo

Here we provide a model zoo that releases the adapted weights of our TAMP in different downstream tasks. If you want to contribute to the Model Zoo, please send the [EMAIL](mailto:ythe1995@163.com) to our group.

## News
- **2024.12.15** We have released the **TAMP-S (Real-world)** models adapted on real-world clinical NICT data from diverse sources.
- **2024.09.25** We have released the TAMP-S models that were evaluated in our [paper](https://arxiv.org/abs/2410.01591).

## Model Record

### 1. TAMP-S (Simulation-based)

[![Download](https://img.shields.io/badge/Download-HuggingFace-yellow)](https://huggingface.co/datasets/YutingHe-list/SimNICT/tree/main/weights/ModelZoo/TAMP-S(Simulation_based))

- **Adaptation method:** Low-rank adaptation (LoRA)
- **NICT setting:** Low-dose CT (LDCT), Sparse-view CT (SVCT), and Limited-angle CT (LACT)
- **Target:** Whole human body, abdomen, and chest
- **Dataset:** AutoPET, AMOS22, and COVID-19
- **Description:** **27** simulation-based adaptation models evaluated in our paper. These models are adapted to three NICT settings with different defect degrees (High, Mid, Low).

| NICT Setting | Defect Degree | Parameter | Dataset |
|--------------|---------------|-----------|---------|
| LDCT | High / Mid / Low | N_LD = 20 / 40 / 60 | AutoPET, AMOS22, COVID-19 |
| SVCT | High / Mid / Low | N_SV = 60 / 120 / 300 | AutoPET, AMOS22, COVID-19 |
| LACT | High / Mid / Low | N_LA = 90 / 120 / 150 | AutoPET, AMOS22, COVID-19 |

---

### 2. TAMP-S (Real-world)

[![Download](https://img.shields.io/badge/Download-HuggingFace-yellow)](https://huggingface.co/datasets/YutingHe-list/SimNICT/tree/main/weights/ModelZoo/TAMP-S(Real_world))

- **Adaptation method:** Low-rank adaptation (LoRA)
- **NICT setting:** Low-dose CT (LDCT), Sparse-view CT (SVCT), and Limited-angle CT (LACT)
- **Target:** Chest, cardiac, and abdomen
- **Dataset:** NJDTH (Nanjing Drum Tower Hospital Jiangbei), Mayo Clinic Low Dose CT Grand Challenge
- **Description:** **6** real-world adaptation models validated in our paper. These models are adapted to real-world clinical NICT data from diverse anatomical regions and imaging protocols.

#### Model Details

| Model | NICT Type | Body Region | Clinical Application | Training Data | Imaging Protocol |
|-------|-----------|-------------|---------------------|---------------|------------------|
| TAMP_S_NJDTH_A_Chest_LDCT | LDCT | Chest | Routine Chest CT | 5 cases (1,234 slices) | Low: 80 kVp, 30 mAs / High: 120 kVp, 90 mAs |
| TAMP_S_NJDTH_B_Cardiac_LDCT | LDCT | Cardiac | Coronary CTA | 5 cases (1,527 slices) | Low: 120 kVp, 50 mAs / High: 120 kVp, 200 mAs |
| TAMP_S_NJDTH_C_Cardiac_SVCT | SVCT | Cardiac | Full-cycle Cardiac Imaging | 5 cases (625 slices) | N_SV = 240 |
| TAMP_S_NJDTH_C_Cardiac_LACT | LACT | Cardiac | Full-cycle Cardiac Imaging | 5 cases (625 slices) | N_LA = 120 |
| TAMP_S_Mayo_Abdomen_SVCT | SVCT | Abdomen | Abdominal CT | 5 cases (990 slices) | N_SV = 180 |
| TAMP_S_Mayo_Abdomen_LACT | LACT | Abdomen | Abdominal CT | 5 cases (990 slices) | N_LA = 140 |

---

## Usage

To use the adapted TAMP-S models, please refer to the [Adaptation with LoRA](../README.md#4-adaptation-with-lora) section in the main README.

**Example for inference with TAMP-S:**

```bash
# Single slice testing
python inference.py --testing_mode "single_slice" \
    --input_path "your_input.nii.gz" \
    --output_path "your_output.nii.gz" \
    --LoRA_mode "load" \
    --LoRA_path "weights/TAMP_adaptation_weight/YOUR_MODEL.pkl"

# Volume testing
python inference.py --testing_mode "single_volume" \
    --input_path "your_input.nii.gz" \
    --output_path "your_output.nii.gz" \
    --LoRA_mode "load" \
    --LoRA_path "weights/TAMP_adaptation_weight/YOUR_MODEL.pkl"
```

---

## Acknowledgement

We thank the following institutions and datasets for their contributions:
- **Nanjing Drum Tower Hospital Jiangbei** for providing real-world clinical NICT data
- **Mayo Clinic Low Dose CT Grand Challenge** for publicly available projection data
- **AutoPET**, **AMOS22**, and **COVID-19** datasets for simulation-based validation

---

## Citation

If you use the TAMP-adapted models in your research, please cite our paper:

```bibtex
@misc{liu2024imagingfoundationmodeluniversal,
      title={Imaging foundation model for universal enhancement of non-ideal measurement CT}, 
      author={Yuxin Liu and Rongjun Ge and Yuting He and Zhan Wu and Chenyu You and Shuo Li and Yang Chen},
      year={2024},
      eprint={2410.01591},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.01591}, 
}
```
