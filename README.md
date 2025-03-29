## Under Construction

[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/AERCA/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
# Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery
A Pytorch implementation of [AERCA](https://openreview.net/forum?id=k38Th3x4d9).

## Configuration
- Ubuntu 20.04
- NVIDIA driver 535.216.03
- CUDA 12.2
- Python 3.11.11
- PyTorch 2.6.0+cu118

## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Usage
Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/AERCA.git my-project
        cd my-project

Run and test:

        python3 main.py --dataset_name linear


## Citation
```

```