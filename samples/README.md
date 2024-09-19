For your convenience, we have provided [data](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EjK8gleWI4JCiVfYuyLHeBkBJYqQaCq3SzfVF8M2qmwfOg?e=6oiehh) for model testing and adaptation. Download and place the files into the corresponding directories as needed. You can also provide your own data and place it in the appropriate folders.


# Slice testing
The 2D NIFTI files with the shape [1, H, W] for testing have been placed in the `slice_testing/input` directory for direct testing. The output enhanced by [MITAMP](../README.md#31-slice-testing) or [MITAMP-S](../README.md#42-slice-testing) will be saved in the `slice_testing/output` directory. The expected directory structure is as follows:

```bash
__slice_testing
  |__input
  |  |__LDCT_Low.nii.gz
  |  |__LDCT_Mid.nii.gz
  |  |__...
  |__output
     |__LDCT_Low.nii.gz
     |__LDCT_Mid.nii.gz
     |__...
```

# Volume testing
The 3D NIFTI files with the shape [S, H, W] for testing have been uploaded to [this link](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EtcfNaDy40lLt-tckD_lfJQBDWixEfxpYpUU-76f93jt7Q?e=qh42Yo). Download and place them in the `volume_testing/input` directory as needed. The output enhanced by [MITAMP](../README.md#32-volume-testing) or [MITAMP-S](../README.md#43-volume-testing) will be saved in the `volume_testing/output` directory. The expected directory structure is as follows:

```bash
__volume_testing
  |__input
  |  |__1.nii.gz
  |  |__2.nii.gz
  |  |__...
  |__label
  |  |__1.nii.gz
  |  |__2.nii.gz
  |  |__...
  |__output
     |__1.nii.gz
     |__2.nii.gz
     |__...
```

# Adaptation
The NIFTI files for adaptation have been uploaded to [this link](https://seunic-my.sharepoint.cn/:f:/g/personal/220232198_seu_edu_cn/EuhW8PS-H2ZApQdw9odb-5MB96Q-XZw4N3JGhK3q7ZIc2A?e=v2FirK). Download and place them in the `adaptation/input` and `adaptation/label` directory as needed. The expected directory structure is as follows:
```bash
__adaptation
  |__input
  |  |__1.nii.gz
  |  |__2.nii.gz
  |  |__...
  |__label
     |__1.nii.gz
     |__2.nii.gz
     |__...
```