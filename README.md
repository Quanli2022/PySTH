
# PySTH Toolkit

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)

PySTH is a computational toolkit for analyzing solar-to-hydrogen (STH) conversion efficiency in photocatalytic materials, supporting four major photocatalytic systems.

## Features

- **Four Supported Systems**:
  - Conventional Photocatalysts
  - Janus Materials
  - Z-Scheme Heterojunctions
  - Janus Z-Scheme Heterojunctions
- **Core Functionalities**:
  - STH efficiency calculation
  - Automated contour map generation
  - Interactive CLI interface
  - Data visualization and export
- **Output Management**:
  - Automatic .dat file generation
  - High-resolution PNG plots
  - Organized output directories

## Installation

Clone the repository:

```bash
git clone https://github.com/Quanli2022/PySTH.git
tar -xJvf PySTH.tar
cd PySTH
pip install -r requirements.txt
```

### Usage

Launch the Toolkit:

```bash
python main.py
```

## Main Menu Interface

```
=========================== PySTH Toolkit ===========================
 1) Conventional photocatalysts    
 2) Janus materials                
 3) Z-scheme systems               
 4) Janus Z-scheme heterojunctions 
 0) Quit
---------------------------------------------------------------------
```

### Workflow Example 1: Conventional Photocatalysts

1. Select material type: `1`
2. Choose sub-function: `21`
   - Calculate STH efficiency
   - Generate STH efficiency map
3. Input parameters:
   - Conduction Band Minimum (CBM) in eV: `-4.2`
   - Valence Band Maximum (VBM) in eV: `-6.5`

View results and generated STH efficiency map:
- `X(H2) (eV)`: Hydrogen Evolution Reaction (HER) Overpotential
- `X(O2) (eV)`: Oxygen Evolution Reaction (OER) Overpotential
- `nabs (%)`: Light Absorption Efficiency
- `ncu (%)`: Charge Utilization Efficiency
- `nSTH (%)`: Solar-to-Hydrogen Conversion Efficiency (Core Metric)

Sample Output:

| pH  | X(H2) (eV) | X(O2) (eV) | nabs (%) | ncu (%) | nSTH (%) |
|-----|------------|------------|----------|---------|----------|
| 0   | 0.24       | 0.83       | 12.34    | 45.67   | 5.63     |
| 1   | 0.18       | 0.89       | 11.92    | 44.15   | 5.26     |
| ... | ...        | ...        | ...      | ...     | ...      |

---

### Workflow Example 2: Janus Materials

1. Select material type: `2`
2. Choose sub-function: `21`
   - Calculate STH efficiency
3. Choose the direction of the vacuum level difference: `1`
4. Select type: `211` (monolayer)
5. Input parameters:
   - Conduction Band Minimum (CBM) in eV: `-4.94`
   - Valence Band Maximum (VBM) in eV: `-6.08`
   - Vacuum Level Difference in eV: `1`

In the following pH range, photocatalytic materials can split water:

| pH  | X(H2) (eV) | X(O2) (eV) | nabs (%) | ncu (%) | nSTH (%) | nSTH_Error (%) |
|-----|------------|------------|----------|---------|----------|----------------|
| 0   | 0.50       | 0.41       | 79.75    | 54.46   | 43.43    | 30.39          |
| 1   | 0.44       | 0.47       | 79.75    | 56.54   | 45.10    | 31.56          |
| 2   | 0.38       | 0.53       | 79.75    | 60.74   | 48.44    | 33.90          |
| 3   | 0.32       | 0.59       | 79.75    | 65.20   | 52.00    | 36.39          |
| ... | ...        | ...        | ...      | ...     | ...      | ...            |

---

## Architecture

### Core Modules

| Module    | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| `load.py` | Contains data processing and calculation logic:                              |
|           | - `load_data()`                                                              |
|           | - `calculate_STH()`                                                          |
|           | - Material classes (General, Heterojunction_Z, etc.)                         |
| `main.py` | Handles user interaction:                                                    |
|           | - CLI interface                                                              |
|           | - Input validation                                                           |
|           | - Result display                                                             |


### Dependencies

Key Dependencies:
- `numpy >=1.25.1`
- `matplotlib >=3.9.2`
- `pandas >=2.1.4`
- `xlrd >=2.0.1`
- `rich >=13.8.1`

Full dependency list: see `setup.py`

### Output Files

Generated in system-specific folders:

- `Conventional photocatalysts/`
  - `STH Efficiency vs pH.png`
  - `STH Efficiency vs HER and OER.dat`
  - `BandGap_Map.png`

- `Janus materials/`
  - `STH Efficiency vs CBM and VBM.png`
  - `STH Efficiency vs Eg and Vacuum_Level_Difference.dat`

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See LICENSE for more information.

---

## Contact

- **Tao**: 1713050146@qq.com
- **Project Link**: [PySTH on GitHub](https://github.com/Quanli2022/PySTH.git)

---

This README features:
- Standard open-source project structure
- Clear installation/usage instructions
- Module architecture overview
- Contribution guidelines
- Responsive badges and tables
- Concise technical documentation

The document uses:
- Standard Markdown formatting
- Emoji-free professional style
- Consistent terminology
- Logical information hierarchy
- Cross-references to key files
