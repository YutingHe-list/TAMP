# TAMP Examples

This directory contains example scripts demonstrating various TAMP functionalities.

## Available Examples

### 1. NICT Simulator Demo (`simnict_demo.py`)

Interactive demonstration of the NICT Simulator for NICT simulation.

**Features**:
- Single slice NICT generation (SVCT, LACT, LDCT)
- Complete 3D volume processing
- Parameter comparison across different settings

**Usage**:
```bash
python examples/simnict_demo.py
```

**Requirements**:
- ODL package for SVCT/LACT
- ASTRA Toolbox with CUDA for LDCT
- Sample data in `samples/` directory

**Expected Output**:
- NICT simulated slices in `samples/slice_testing/output/simnict_demo/`
- NICT simulated volume in `samples/volume_testing/output/`
- Parameter comparison results in `samples/slice_testing/output/simnict_comparison/`

## Coming Soon

- [ ] Jupyter notebook for interactive TAMP usage
- [ ] End-to-end workflow: simulation → enhancement → evaluation
- [ ] Visualization tools for NICT artifacts
- [ ] Batch processing examples for large datasets

## Quick Start

1. Ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```

2. Download sample data (if not already available):
   - Follow instructions in main [README.md](../README.md)

3. Run the demo:
```bash
python simnict_demo.py
```

## Need Help?

- Check the main [README.md](../README.md) for setup instructions
- Review [SimNICT documentation](../document/SimNICT_toolkit.md) for API details
- Open an issue on GitHub if you encounter problems

## Contributing

We welcome contributions of new examples! Please:
1. Follow the existing code style
2. Add clear documentation and comments
3. Test with sample data before submitting
4. Update this README with your new example

