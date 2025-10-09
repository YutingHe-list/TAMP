# NICT Simulator Documentation

## Overview

The NICT Simulator provides physics-based simulation functions to generate three types of NICT from standard CT images. This simulator was used to create the **SimNICT dataset** (10.9 million NICT-ICT image pairs) mentioned in our paper.

## Simulation Methods

### 1. Sparse-View CT (SVCT)

**Physical Principle**: Reduces the number of projection views during CT acquisition to decrease scan time and radiation dose.

**Implementation**: Uses ODL's fan-beam geometry with limited angular samples (15-360 views instead of standard 720 views). Filtered back-projection (FBP) is used for reconstruction, which introduces characteristic streak artifacts with fewer views.

**Parameters**:
- `num_views`: Number of projection angles (15-360)
- Typical values: 30, 60, 90, 120 views
- Lower views → more severe artifacts

### 2. Limited-Angle CT (LACT)

**Physical Principle**: Restricts the angular scanning range due to geometric constraints or obstructions.

**Implementation**: Uses ODL's fan-beam geometry with partial angular coverage (75°-270° instead of full 360°). The incomplete angular data causes anisotropic resolution and shadow artifacts in the direction perpendicular to the missing angles.

**Parameters**:
- `angle_range`: Scanning angular range in degrees (75-270)
- Typical values: 120°, 150°, 180°
- Smaller angle → more severe directional artifacts

### 3. Low-Dose CT (LDCT)

**Physical Principle**: Reduces X-ray photon flux to minimize radiation exposure, resulting in increased quantum noise.

**Implementation**: Uses ASTRA Toolbox:
1. Converts HU values to linear attenuation coefficients
2. Performs forward projection to create sinogram
3. Adds Poisson noise proportional to dose reduction
4. Reconstructs using FBP with noisy sinogram

**Parameters**:
- `dose_percentage`: Dose level as % of standard dose (5-75%)
- Typical values: 10%, 25%, 50%
- Lower dose → more severe noise

## Detailed API Reference

### Function: `create_sparse_view_ct()`

```python
def create_sparse_view_ct(ict_slice, height, width, num_views=60)
```

**Parameters**:
- `ict_slice` (np.ndarray): Input ICT slice in range [0, 4096]
- `height` (int): Image height in pixels
- `width` (int): Image width in pixels
- `num_views` (int): Number of projection views (default: 60)

**Returns**:
- `np.ndarray`: Reconstructed SVCT slice in range [0, 4096]

**Geometry Details**:
- Fan-beam geometry
- Source-to-isocenter: 1270 mm
- Detector-to-isocenter: 870 mm
- Detector pixels: 1024
- Detector size: 720 mm (±360 mm)

---

### Function: `create_limited_angle_ct()`

```python
def create_limited_angle_ct(ict_slice, height, width, angle_range=120)
```

**Parameters**:
- `ict_slice` (np.ndarray): Input ICT slice in range [0, 4096]
- `height` (int): Image height in pixels
- `width` (int): Image width in pixels
- `angle_range` (int): Angular scanning range in degrees (default: 120)

**Returns**:
- `np.ndarray`: Reconstructed LACT slice in range [0, 4096]

**Geometry Details**:
- Same fan-beam geometry as SVCT
- Number of angles adjusted proportionally to angle range
- Standard full scan: 720 angles over 360°
- Example: 120° scan uses 240 angles

---

### Function: `create_low_dose_ct()`

```python
def create_low_dose_ct(ict_slice, height, width, dose_percentage=25)
```

**Parameters**:
- `ict_slice` (np.ndarray): Input ICT slice in HU range [-1024, 3072]
- `height` (int): Image height in pixels
- `width` (int): Image width in pixels
- `dose_percentage` (int): Dose level as % of normal dose (default: 25)

**Returns**:
- `np.ndarray`: Reconstructed LDCT slice in HU range [-1024, 3072]

**Physics Model**:
- Linear attenuation coefficient: μ = 0.0192 mm⁻¹ (water at 70 keV)
- Noise model: Poisson distribution in projection domain
- Noise scaling: σ ∝ √[(1-d)/d · exp(sinogram)/10⁶]
  where d is dose fraction

**Geometry Details**:
- Fan-beam flat detector
- Detector pixel spacing: 1.686 mm
- Source-origin distance: 600.45 mm
- Origin-detector distance: 485.15 mm
- 720 projection angles

---

### Function: `batch_create_nict_volume()`

```python
def batch_create_nict_volume(ict_volume, nict_type='LDCT', **kwargs)
```

**Parameters**:
- `ict_volume` (np.ndarray): Input ICT volume with shape [H, W, S]
- `nict_type` (str): Type of NICT: 'SVCT', 'LACT', or 'LDCT'
- `**kwargs`: Type-specific parameters
  - SVCT: `num_views` (default: 60)
  - LACT: `angle_range` (default: 120)
  - LDCT: `dose_percentage` (default: 25)

**Returns**:
- `np.ndarray`: NICT volume with same shape as input

**Example**:
```python
# Process a 512×512×200 volume
ldct_volume = batch_create_nict_volume(
    ict_volume, 
    nict_type='LDCT', 
    dose_percentage=25
)
```

## Complete Usage Examples

### Example 1: Generate Training Data for TAMP Adaptation

```python
import nibabel as nib
from utils.nict_simulator import batch_create_nict_volume

# Load standard CT volume
ict_path = 'samples/adaptation/label/case001.nii.gz'
ict_image = nib.load(ict_path)
ict_volume = ict_image.get_fdata()

# Convert to [0, 4096] range for SVCT/LACT
ict_shifted = ict_volume + 1024

# Generate SVCT with 60 views
svct_volume = batch_create_nict_volume(
    ict_shifted, 
    nict_type='SVCT', 
    num_views=60
)

# Convert back to HU range and save
svct_hu = svct_volume - 1024
svct_image = nib.Nifti1Image(svct_hu, ict_image.affine)
nib.save(svct_image, 'samples/adaptation/input/case001.nii.gz')
```

### Example 2: Create Multi-Level LDCT Dataset

```python
import nibabel as nib
from utils.nict_simulator import batch_create_nict_volume

# Load ICT
ict_image = nib.load('data/standard_ct.nii.gz')
ict_volume = ict_image.get_fdata()

# Generate different dose levels
dose_levels = [10, 25, 50, 75]
for dose in dose_levels:
    ldct_volume = batch_create_nict_volume(
        ict_volume,
        nict_type='LDCT',
        dose_percentage=dose
    )
    
    # Save with dose level in filename
    output_path = f'data/ldct_dose{dose}.nii.gz'
    ldct_image = nib.Nifti1Image(ldct_volume, ict_image.affine)
    nib.save(ldct_image, output_path)
    print(f'Saved {output_path}')
```

### Example 3: Compare Different SVCT View Numbers

```python
import nibabel as nib
import matplotlib.pyplot as plt
from utils.nict_simulator import create_sparse_view_ct

# Load a single slice
ict_image = nib.load('data/slice.nii.gz')
ict_slice = ict_image.get_fdata()[0] + 1024  # Convert to [0, 4096]
height, width = ict_slice.shape

# Generate SVCT with different views
view_counts = [15, 30, 60, 120, 240]
results = {}

for views in view_counts:
    svct = create_sparse_view_ct(ict_slice, height, width, num_views=views)
    results[views] = svct
    print(f'Generated SVCT with {views} views')

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Original
axes[0].imshow(ict_slice, cmap='gray', vmin=0, vmax=4096)
axes[0].set_title('Original ICT')
axes[0].axis('off')

# SVCT results
for idx, views in enumerate(view_counts):
    axes[idx+1].imshow(results[views], cmap='gray', vmin=0, vmax=4096)
    axes[idx+1].set_title(f'SVCT - {views} views')
    axes[idx+1].axis('off')

plt.tight_layout()
plt.savefig('svct_comparison.png', dpi=300)
print('Comparison saved to svct_comparison.png')
```

### Example 4: Batch Processing Multiple Cases

```python
import os
import nibabel as nib
from utils.nict_simulator import batch_create_nict_volume

# Configuration
input_dir = 'data/standard_ct/'
output_dir = 'data/ldct_dose25/'
dose_percentage = 25

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process all NIfTI files
nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

for idx, filename in enumerate(nii_files):
    print(f'Processing {idx+1}/{len(nii_files)}: {filename}')
    
    # Load ICT volume
    ict_path = os.path.join(input_dir, filename)
    ict_image = nib.load(ict_path)
    ict_volume = ict_image.get_fdata()
    
    # Generate LDCT
    ldct_volume = batch_create_nict_volume(
        ict_volume,
        nict_type='LDCT',
        dose_percentage=dose_percentage
    )
    
    # Save result
    output_path = os.path.join(output_dir, filename)
    ldct_image = nib.Nifti1Image(ldct_volume, ict_image.affine)
    nib.save(ldct_image, output_path)
    
print(f'Batch processing complete! Processed {len(nii_files)} files.')
```

## Technical Notes

### Data Range Conventions

The toolkit uses the following data range conventions:

| Operation | Input Range | Output Range | Note |
|-----------|-------------|--------------|------|
| SVCT | [0, 4096] | [0, 4096] | Shifted HU scale |
| LACT | [0, 4096] | [0, 4096] | Shifted HU scale |
| LDCT | [-1024, 3072] | [-1024, 3072] | Standard HU scale |

**Conversion between ranges**:
```python
# Standard HU to shifted scale
shifted = hu_values + 1024

# Shifted scale to standard HU
hu_values = shifted - 1024
```

## Citation

If you use the NICT Simulator in your research, please cite our paper:

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

## Contact

For questions or issues related to the NICT Simulator:
- Open an issue on [GitHub](https://github.com/YutingHe-list/TAMP)
- Email: ythe1995@163.com

## License

This simulator is released under the same license as the TAMP project.

