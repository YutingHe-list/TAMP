# -*- coding: utf-8 -*-

"""
NICT Simulator Demo Script
Demonstrates basic usage of the NICT Simulator for NICT simulation

This script shows how to:
1. Generate single slice NICT simulations
2. Process complete 3D volumes
3. Save results in NIfTI format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk
from utils.nict_simulator import create_sparse_view_ct, create_limited_angle_ct, create_low_dose_ct, batch_create_nict_volume


def demo_single_slice():
    """Demo: Generate NICT from a single CT slice"""
    print("\n" + "="*60)
    print("Demo 1: Single Slice NICT Simulation")
    print("="*60)
    
    # Load a test slice
    input_path = "samples/slice_testing/input/LDCT_Low.nii.gz"
    
    if not os.path.exists(input_path):
        print(f"Warning: Test file not found: {input_path}")
        print("Please ensure sample data is available.")
        return
    
    # Load slice
    input_image = sitk.ReadImage(input_path)
    ict_slice = sitk.GetArrayFromImage(input_image)[0]  # Get first slice
    height, width = ict_slice.shape
    
    print(f"Loaded slice: {height}x{width}")
    print(f"Value range: [{ict_slice.min():.1f}, {ict_slice.max():.1f}] HU")
    
    # Convert to [0, 4096] range for SVCT/LACT
    ict_shifted = ict_slice + 1024
    
    # 1. Generate Sparse-View CT
    print("\nGenerating SVCT (60 views)...")
    svct_slice = create_sparse_view_ct(ict_shifted, height, width, num_views=60)
    svct_hu = svct_slice - 1024  # Convert back to HU
    print(f"SVCT output range: [{svct_hu.min():.1f}, {svct_hu.max():.1f}] HU")
    
    # 2. Generate Limited-Angle CT
    print("\nGenerating LACT (120° angle)...")
    lact_slice = create_limited_angle_ct(ict_shifted, height, width, angle_range=120)
    lact_hu = lact_slice - 1024  # Convert back to HU
    print(f"LACT output range: [{lact_hu.min():.1f}, {lact_hu.max():.1f}] HU")
    
    # 3. Generate Low-Dose CT
    print("\nGenerating LDCT (25% dose)...")
    ldct_slice = create_low_dose_ct(ict_slice, height, width, dose_percentage=25)
    print(f"LDCT output range: [{ldct_slice.min():.1f}, {ldct_slice.max():.1f}] HU")
    
    # Save results
    output_dir = "samples/slice_testing/output/simnict_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in [('svct', svct_hu), ('lact', lact_hu), ('ldct', ldct_slice)]:
        output_path = os.path.join(output_dir, f"{name}_demo.nii.gz")
        output_image = sitk.GetImageFromArray(data[np.newaxis, :, :])
        output_image.CopyInformation(input_image)
        sitk.WriteImage(output_image, output_path)
        print(f"Saved: {output_path}")
    
    print("\n✓ Single slice demo completed!")


def demo_volume_processing():
    """Demo: Process complete 3D volume"""
    print("\n" + "="*60)
    print("Demo 2: Volume NICT Simulation")
    print("="*60)
    
    # Check for volume test data
    input_path = "samples/volume_testing/input/1.nii.gz"
    
    if not os.path.exists(input_path):
        print(f"Warning: Test volume not found: {input_path}")
        print("Please download test data following README instructions.")
        return
    
    # Load volume
    print(f"Loading volume from: {input_path}")
    input_image = sitk.ReadImage(input_path)
    ict_volume = sitk.GetArrayFromImage(input_image)
    S, H, W = ict_volume.shape
    
    print(f"Volume shape: {S} slices × {H} × {W}")
    print(f"Value range: [{ict_volume.min():.1f}, {ict_volume.max():.1f}] HU")
    
    # Generate LDCT volume (fastest option for demo)
    print("\nGenerating LDCT volume (25% dose)...")
    ldct_volume = batch_create_nict_volume(
        ict_volume, 
        nict_type='LDCT', 
        dose_percentage=25
    )
    
    print(f"\nLDCT output range: [{ldct_volume.min():.1f}, {ldct_volume.max():.1f}] HU")
    
    # Save result
    output_dir = "samples/volume_testing/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ldct_demo.nii.gz")
    
    output_image = sitk.GetImageFromArray(ldct_volume.astype(np.int16))
    output_image.CopyInformation(input_image)
    sitk.WriteImage(output_image, output_path)
    
    print(f"Saved: {output_path}")
    print("\n✓ Volume processing demo completed!")


def demo_parameter_comparison():
    """Demo: Compare different simulation parameters"""
    print("\n" + "="*60)
    print("Demo 3: Parameter Comparison")
    print("="*60)
    
    # Create synthetic test slice
    print("Creating synthetic test data (512×512)...")
    height, width = 512, 512
    
    # Create a simple phantom with structures
    test_slice = np.zeros((height, width), dtype=np.float32)
    
    # Add circular structures with different densities
    center = height // 2
    for radius, value in [(150, 2048), (100, 1500), (50, 1000)]:
        y, x = np.ogrid[:height, :width]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        test_slice[mask] = value
    
    print(f"Test slice range: [{test_slice.min():.1f}, {test_slice.max():.1f}]")
    
    # Compare SVCT with different view numbers
    print("\nComparing SVCT with different view numbers...")
    view_counts = [30, 60, 120, 240]
    
    output_dir = "samples/slice_testing/output/simnict_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    for views in view_counts:
        print(f"  Generating SVCT with {views} views...")
        svct = create_sparse_view_ct(test_slice, height, width, num_views=views)
        
        # Save
        output_path = os.path.join(output_dir, f"svct_{views}views.nii.gz")
        output_image = sitk.GetImageFromArray(svct[np.newaxis, :, :])
        sitk.WriteImage(output_image, output_path)
    
    print(f"\nSaved comparison results to: {output_dir}")
    print("✓ Parameter comparison demo completed!")


def main():
    """Run all demos"""
    print("\n" + "🔬 " * 20)
    print("NICT Simulator - Interactive Demo")
    print("🔬 " * 20)
    
    try:
        # Demo 1: Single slice
        demo_single_slice()
        
        # Demo 2: Volume processing
        demo_volume_processing()
        
        # Demo 3: Parameter comparison
        demo_parameter_comparison()
        
        print("\n" + "="*60)
        print("All demos completed successfully! 🎉")
        print("="*60)
        print("\nNext steps:")
        print("1. Check output folders for generated NICT images")
        print("2. Use TAMP for enhancement: python inference.py --input_path <nict_file>")
        print("3. Explore NICT_Simulator.md for advanced usage")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("- Required packages are installed (odl, astra-toolbox)")
        print("- CUDA is available for LDCT simulation")
        print("- Sample data is downloaded")


if __name__ == "__main__":
    main()

