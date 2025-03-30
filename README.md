## Under Construction

[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/AERCA/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
# AERCA: Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery
<div align="center">
  <img src="https://github.com/hanxiao0607/AERCA/blob/main/other/poster.png" alt="AERCA Poster" />
</div>

---

## Table of Contents

1. [Overview](#overview)
2. [System Configuration](#system-configuration)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Cloning the Repository](#cloning-the-repository)
    - [Running the Code](#running-the-code)
    - [Command-Line Options](#command-line-options)
5. [Datasets](#datasets)
6. [Citation](#citation)
7. [Contact & Support](#contact--support)

---

## Overview

The AERCA algorithm performs robust root cause analysis in multivariate time series data by leveraging Granger causal discovery methods. This implementation in PyTorch facilitates experimentation on both synthetic and real-world datasets, allowing researchers to:
- Identify causality relationships among variables.
- Detect anomalies and trace them to their root causes.
- Compare performance across various types of datasets.

For further details, refer to the paper [AERCA](https://openreview.net/forum?id=k38Th3x4d9).

---

## System Configuration

The code has been developed and tested using the following system setup:

- **Operating System:** Ubuntu 20.04
- **GPU Driver:** NVIDIA driver 535.216.03
- **CUDA Version:** 12.2
- **Python Version:** 3.11.11
- **PyTorch Version:** 2.6.0+cu118

---

## Installation

### Prerequisites

- **Python 3.11:** Ensure that Python 3.11 is installed.
- **Virtual Environment (Recommended):** It is advisable to use a virtual environment to manage dependencies.

### Steps

1. **Install `virtualenv` (if not already installed):**

   ```bash
   python3 -m pip install --user virtualenv
    ```
   
2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   ```
3. **Activate the virtual environment:**

   ```bash
    source venv/bin/activate
    ```
4. **Install the required packages:**
    
    ```bash
    pip install -r requirements.txt
    ```
   
5. **Deactivate the virtual environment (when done):**

   ```bash
   deactivate
   ```
   
For more details on setting up a virtual environment, refer to the [Python documentation](https://docs.python.org/3/tutorial/venv.html).

---

## Usage
### Cloning the Repository
Clone the repository from GitHub and navigate into the project directory. Replace `my-project` with your preferred folder name:


     git clone https://github.com/hanxiao0607/AERCA.git my-project
     cd my-project

### Running the Code
Execute the main script main.py to run the AERCA algorithm. You need to specify the dataset name using the `--dataset_name` argument. The supported dataset names are:

   - `linear`
   - `nonlinear`
   - `lorenz96`
   - `lotka_volterra`
   - `swat`
   - `msds`

For example, to run the algorithm on the `linear` dataset:

```bash
    python main.py --dataset_name linear
```

### Command-Line Options
Additional command-line options are available to customize the run-time behavior. To see all available options, execute:

```bash
    python main.py --help
```

This will display details on parameters such as hyperparameter settings, logging options, and more.

---

## Datasets
The repository includes support for multiple datasets, each designed to evaluate the algorithm under different conditions:

- Linear Dataset: Ideal for evaluating causal relationships in linear systems.
- Nonlinear Dataset: Tests the model’s performance in systems with nonlinear interactions.
- Lorenz96: A synthetic dataset based on the Lorenz96 model, often used for studying chaotic dynamics.
- Lotka-Volterra: Derived from predator-prey dynamics, useful for ecological and biological time series analysis.
- [MSDS](https://github.com/hanxiao0607/AERCA/tree/main/datasets/msds): A real-world dataset from the MSDS collection.
- [SWaT](https://github.com/hanxiao0607/AERCA/tree/main/datasets/swat): A dataset for anomaly detection in water treatment systems.

Ensure that the dataset you choose is formatted as expected by the code. Additional preprocessing scripts or instructions may be provided within the repository as needed.

---

## Citation
```

```

---

## Contact & Support
For any questions, issues, or contributions, please open an issue on GitHub or contact the repository maintainer.