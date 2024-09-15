Place the downloaded data here, and the folder structure is as follows:
```bash
__samples
  |__slice_testing
  |  |__input
  |     |__LDCT_Low.nii.gz
  |     |__LDCT_Mid.nii.gz
  |     |__...
  |
  |__volume_testing
  |  |__input
  |  |  |__LDCT_Low
  |  |  |  |__1.nii.gz
  |  |  |  |__2.nii.gz
  |  |  |  |__...
  |  |  |
  |  |  |__LDCT_Mid
  |  |  |__...
  |  |
  |  |__label
  |     |__LDCT_Low
  |     |  |__1.nii.gz
  |     |  |__2.nii.gz
  |     |  |__...
  |     |
  |     |__LDCT_Mid
  |     |__...
  |
  |__fine_tuning
     |__input
     |  |__LDCT_Low
     |  |  |__1.nii.gz
     |  |  |__2.nii.gz
     |  |  |...
     |  |
     |  |__LDCT_Mid
     |  |__...
     |
     |__label
        |__LDCT_Low
        |  |__1.nii.gz
        |  |__2.nii.gz
        |  |__...
        |
        |__LDCT_Mid
        |__...
```