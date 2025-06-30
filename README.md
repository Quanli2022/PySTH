# PySTH Toolkit 

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue)](https://www.python.org/)

PySTH is a computational toolkit for analyzing solar-to-hydrogen (STH) 
conversion efficiency in photocatalytic materials, supporting four major 
photocatalytic systems.

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


linux system
Clone repository:
git clone --branch v1.0.0-alpha https://github.com/Quanli2022/PySTH.git
cd PySTH-1.0.0-alpha
tar -xJvf main.tar.gz
cd main
pip install .

Usage
Launch the Toolkit:
PySTH
-----------------------------------------------------------------
Windows system
git clone --branch v1.0.0-alpha https://github.com/Quanli2022/PySTH.git
cd PySTH-1.0.0-alpha
copy main.tar.gz to local
tar -xf .\main.tar.gz
cd main

Usage
Launch the Toolkit:
Run main.exe

Main Menu Interface

=========================== PySTH Toolkit ===========================
 1) Conventional photocatalysts    
 2) Janus materials                
 3) Z-scheme systems               
 4) Janus Z-scheme heterojunctions 
 0) Quit
---------------------------------------------------------------------

Workflow Example (Conventional Photocatalysts)(1)

Select material type : 1)

Choose sub-function: 21)

Calculate STH efficiency

Generate STH efficiency map

Input parameters when prompted:
- Conduction Band Minimum (CBM) in eV: -4.2
- Valence Band Maximum (VBM) in eV: -6.5

View results and generated STH efficiency map
- X(H2) (eV):Hydrogen Evolution Reaction (HER) Overpotential
- X(O2) (eV):Oxygen Evolution Reaction (OER) Overpotential
- nabs (%):Light Absorption Efficiency
- ncu (%):Charge Utilization Efficiency
- nSTH (%):Solar-to-Hydrogen Conversion Efficiency (Core Metric)

Sample Output

pH | X(H2) (eV) | X(O2) (eV) | nabs (%) | ncu (%) | nSTH (%)

0  | 0.24       | 0.83       | 12.34    | 45.67   | 5.63

1  | 0.18       | 0.89       | 11.92    | 44.15   | 5.26

... 

Workflow Example (Janus materials)(2)

Select material type : 2)

Calculate STH efficiency :21)

The direction of the vacuum level difference : 1)

Select type : 211) (monolayer)

Input parameters when prompted:
- Conduction Band Minimum (CBM) in eV: -4.94
- Valence Band Maximum (VBM) in eV: -6.08
- Vacuum Level Difference (ΔΦ) in eV: 1

In the following pH range, photocatalytic materials can split water

pH: 0 - 8

  pH  |  χ(H2) (eV) |   χ(O2) (eV)   | ηabs (%)  |  ηcu (%)  |  ηSTH (%) |   η′STH (%)

----  ------------  ------------  ----------  ---------  ----------  -----------

   0  |     0.50    |   0.41  |   79.75   |   54.46   |    43.43   |     30.39

   1  |     0.44    |   0.47  |   79.75   |   56.54   |    45.10   |     31.56

   2  |     0.38    |   0.53  |   79.75   |   60.74   |   48.44    |    33.90

   3  |     0.32    |   0.59  |   79.75   |   65.20   |   52.00    |    36.39

...



Architecture

Core Modules
Module	Description
load.py	Contains data processing and calculation logic:
- load_data()
- calculate_STH()
- Material classes (General, Heterojunction_Z, etc.)

main.py	Handles user interaction:
-
- CLI interface
- Input validation
- Result display
setup.py	Package configuration:
- Dependency management
- Entry points
- Metadata
Dependencies
Key Dependencies

numpy >=1.25.1
matplotlib >=3.9.2
pandas >=2.1.4
xlrd >=2.0.1
rich >=13.8.1
Full dependency list: setup.py

Output Files
Generated in system-specific folders:


Conventional photocatalysts/
STH Efficiency vs pH.png
STH Efficiency vs HER and OER.dat
BandGap_Map.png

Janus materials/
STH Efficiency vs CBM and VBM.png
STH Efficiency vs Eg and ΔΦ.dat
Contributing
We welcome contributions! Please follow these steps:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
Distributed under the MIT License. See LICENSE for more information.

Contact
Tao - 1713050146@qq.com
Project Link: https://github.com/Quanli2022/PySTH/tags


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
