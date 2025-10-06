# -*- coding = utf-8 -*-

"""
NICT Simulator - Non-Ideal measurement CT Simulation Tool
Provides core simulation functions for three types of NICT:
1. Sparse-View CT (SVCT): Limited projection views
2. Limited-Angle CT (LACT): Restricted angular range  
3. Low-Dose CT (LDCT): Reduced photon dose with Poisson noise

This simulator was used to create the SimNICT dataset (10.9M NICT-ICT pairs).

Usage:
    from utils.nict_simulator import create_sparse_view_ct, create_limited_angle_ct, create_low_dose_ct
    
    # Generate SVCT with 60 views
    svct_slice = create_sparse_view_ct(ict_slice, height, width, num_views=60)
    
    # Generate LACT with 120° angular range
    lact_slice = create_limited_angle_ct(ict_slice, height, width, angle_range=120)
    
    # Generate LDCT with 25% dose
    ldct_slice = create_low_dose_ct(ict_slice, height, width, dose_percentage=25)

Dependencies: 
    - odl (for SVCT and LACT reconstruction)
    - astra-toolbox (for LDCT reconstruction)
    - numpy
"""

import numpy as np
import odl
import astra


def create_sparse_view_ct(ict_slice, height, width, num_views=60):
    """
    Generate Sparse-View CT simulation using ODL fan-beam geometry
    
    Args:
        ict_slice (np.ndarray): Input ICT slice in HU range [0, 4096]
        height (int): Image height in pixels
        width (int): Image width in pixels
        num_views (int): Number of projection views (default: 60)
                        Range: 15-360 views
    
    Returns:
        np.ndarray: Reconstructed sparse-view CT slice in [0, 4096] range
        
    Example:
        >>> ict_slice = np.array(...)  # ICT image [0, 4096]
        >>> svct_slice = create_sparse_view_ct(ict_slice, 512, 512, num_views=60)
    """
    # Create reconstruction space
    reco_space = odl.uniform_discr(
        min_pt=[-height/4, -width/4], 
        max_pt=[height/4, width/4], 
        shape=[height, width],
        dtype='float32'
    )

    # Define fan-beam geometry with limited views
    angle_partition = odl.uniform_partition(0, 2 * np.pi, num_views)
    detector_partition = odl.uniform_partition(-360, 360, 1024)
    geometry = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition, 
        src_radius=1270,  # Source to isocenter distance (mm)
        det_radius=870    # Detector to isocenter distance (mm)
    )
    
    # Create ray transform operator
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    
    # Forward projection to create sinogram
    projection = ray_trafo(ict_slice.astype('float32'))
    
    # Filtered back-projection reconstruction
    fbp = odl.tomo.fbp_op(ray_trafo) 
    reconstruction = fbp(projection)
    
    return np.array(reconstruction)


def create_limited_angle_ct(ict_slice, height, width, angle_range=120):
    """
    Generate Limited-Angle CT simulation using ODL fan-beam geometry
    
    Args:
        ict_slice (np.ndarray): Input ICT slice in HU range [0, 4096]
        height (int): Image height in pixels
        width (int): Image width in pixels
        angle_range (int): Angular scanning range in degrees (default: 120)
                          Range: 75-270 degrees
    
    Returns:
        np.ndarray: Reconstructed limited-angle CT slice in [0, 4096] range
        
    Example:
        >>> ict_slice = np.array(...)  # ICT image [0, 4096]
        >>> lact_slice = create_limited_angle_ct(ict_slice, 512, 512, angle_range=120)
    """
    # Create reconstruction space
    reco_space = odl.uniform_discr(
        min_pt=[-height/4, -width/4],
        max_pt=[height/4, width/4], 
        shape=[height, width],
        dtype='float32'
    )

    # Define geometry with limited angular range
    angle_fraction = angle_range / 360
    num_angles = int(720 * angle_fraction)
    angle_partition = odl.uniform_partition(0, 2 * np.pi * angle_fraction, num_angles)
    detector_partition = odl.uniform_partition(-360, 360, 1024)
    geometry = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition,
        src_radius=1270,  # Source to isocenter distance (mm)
        det_radius=870    # Detector to isocenter distance (mm)
    )
    
    # Create ray transform operator
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    
    # Forward projection to create limited-angle sinogram
    projection = ray_trafo(ict_slice.astype('float32'))
    
    # Filtered back-projection reconstruction
    fbp = odl.tomo.fbp_op(ray_trafo)
    reconstruction = fbp(projection)
    
    return np.array(reconstruction)


def create_low_dose_ct(ict_slice, height, width, dose_percentage=25):
    """
    Generate Low-Dose CT simulation with Poisson noise using ASTRA
    
    Args:
        ict_slice (np.ndarray): Input ICT slice in HU range [-1024, 3072]
        height (int): Image height in pixels
        width (int): Image width in pixels
        dose_percentage (int): Dose level as percentage of normal dose (default: 25)
                              Range: 5-75%
    
    Returns:
        np.ndarray: Reconstructed low-dose CT slice in [-1024, 3072] range
        
    Example:
        >>> ict_slice = np.array(...)  # ICT image [-1024, 3072]
        >>> ldct_slice = create_low_dose_ct(ict_slice, 512, 512, dose_percentage=25)
        
    Note:
        The input slice should be in standard HU range [-1024, 3072].
        Poisson noise is added to simulate photon count reduction.
    """
    dose_fraction = dose_percentage / 100.0
    
    # Convert HU values to linear attenuation coefficients
    u = 0.0192  # Linear attenuation coefficient (water at 70keV)
    attenuation_map = ict_slice * u / 1000.0 + u
    
    # ASTRA volume geometry setup
    vol_geom = astra.create_vol_geom([height, width])
    
    # Define fan-beam projection geometry
    angles = np.linspace(np.pi, -np.pi, 720)
    proj_geom = astra.create_proj_geom(
        'fanflat',
        1.685839319229126,    # Detector pixel spacing (mm)
        1024,                  # Number of detector pixels
        angles,                # Projection angles
        600.4500331878662,     # Source-origin distance (mm)
        485.1499423980713      # Origin-detector distance (mm)
    )
    
    # Create projector and forward projection operator
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    operator = astra.OpTomo(proj_id)
    
    # Forward projection to create sinogram
    sinogram = operator * np.mat(attenuation_map) / 2
    
    # Add Poisson noise based on dose level
    # Lower dose → higher noise
    noise = np.random.normal(0, 1, 720 * 1024)
    noise_scaling = np.sqrt((1 - dose_fraction) / dose_fraction * (np.exp(sinogram) / 1e6))
    noisy_sinogram = sinogram + noise * noise_scaling
    
    # Filtered back-projection reconstruction
    noisy_sinogram_2d = np.reshape(noisy_sinogram, [720, -1])
    reconstruction = operator.reconstruct('FBP_CUDA', noisy_sinogram_2d)
    
    # Convert back to HU values
    reconstruction = reconstruction.reshape((height, width))
    hu_reconstruction = (reconstruction * 2 - u) / u * 1000
    
    return hu_reconstruction


def batch_create_nict_volume(ict_volume, nict_type='LDCT', **kwargs):
    """
    Batch process a complete 3D CT volume to generate NICT
    
    Args:
        ict_volume (np.ndarray): Input ICT volume with shape [Height, Width, Slices]
        nict_type (str): Type of NICT to generate: 'SVCT', 'LACT', or 'LDCT'
        **kwargs: Parameters for specific NICT type
                 - For SVCT: num_views (default: 60)
                 - For LACT: angle_range (default: 120)
                 - For LDCT: dose_percentage (default: 25)
    
    Returns:
        np.ndarray: NICT volume with same shape as input
        
    Example:
        >>> ict_volume = np.array(...)  # Shape: [512, 512, 200]
        >>> ldct_volume = batch_create_nict_volume(ict_volume, 'LDCT', dose_percentage=25)
        >>> svct_volume = batch_create_nict_volume(ict_volume, 'SVCT', num_views=60)
    """
    L, W, S = ict_volume.shape
    nict_volume = np.zeros((L, W, S), dtype=np.float32)
    
    # Select appropriate simulation function
    if nict_type.upper() == 'SVCT':
        num_views = kwargs.get('num_views', 60)
        for slice_idx in range(S):
            ict_slice = ict_volume[:, :, slice_idx]
            nict_volume[:, :, slice_idx] = create_sparse_view_ct(
                ict_slice, L, W, num_views
            )
            print(f"\rProcessing SVCT slice {slice_idx+1}/{S}", end='')
            
    elif nict_type.upper() == 'LACT':
        angle_range = kwargs.get('angle_range', 120)
        for slice_idx in range(S):
            ict_slice = ict_volume[:, :, slice_idx]
            nict_volume[:, :, slice_idx] = create_limited_angle_ct(
                ict_slice, L, W, angle_range
            )
            print(f"\rProcessing LACT slice {slice_idx+1}/{S}", end='')
            
    elif nict_type.upper() == 'LDCT':
        dose_percentage = kwargs.get('dose_percentage', 25)
        for slice_idx in range(S):
            ict_slice = ict_volume[:, :, slice_idx]
            nict_volume[:, :, slice_idx] = create_low_dose_ct(
                ict_slice, L, W, dose_percentage
            )
            print(f"\rProcessing LDCT slice {slice_idx+1}/{S}", end='')
    else:
        raise ValueError(f"Unknown NICT type: {nict_type}. Must be 'SVCT', 'LACT', or 'LDCT'")
    
    print()  # New line after progress
    return nict_volume


if __name__ == "__main__":
    # Example usage
    print("NICT Simulator - Example Usage")
    print("=" * 50)
    
    # Create a sample ICT slice (512x512)
    height, width = 512, 512
    sample_ict = np.random.rand(height, width) * 2048 + 1024  # Random data [1024, 3072]
    
    print("\n1. Generating Sparse-View CT (60 views)...")
    svct = create_sparse_view_ct(sample_ict, height, width, num_views=60)
    print(f"   Output shape: {svct.shape}, Range: [{svct.min():.1f}, {svct.max():.1f}]")
    
    print("\n2. Generating Limited-Angle CT (120°)...")
    lact = create_limited_angle_ct(sample_ict, height, width, angle_range=120)
    print(f"   Output shape: {lact.shape}, Range: [{lact.min():.1f}, {lact.max():.1f}]")
    
    print("\n3. Generating Low-Dose CT (25% dose)...")
    ldct = create_low_dose_ct(sample_ict - 1024, height, width, dose_percentage=25)
    print(f"   Output shape: {ldct.shape}, Range: [{ldct.min():.1f}, {ldct.max():.1f}]")
    
    print("\n" + "=" * 50)
    print("All simulations completed successfully!")

