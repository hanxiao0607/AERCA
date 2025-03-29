# SWaT Dataset Usage Guide

This document provides a overview of how to integrate and utilize the Secure Water Treatment (SWaT) dataset within this project. The SWaT dataset is essential for various analyses and computations performed by the provided source code. Please follow the instructions below to ensure proper configuration and usage.


---

## 1. Requesting the Dataset

1. **Visit the iTrust Website**  
   Navigate to the [iTrust website](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) to find detailed instructions on requesting the SWaT dataset.

2. **Submit the Request**  
   Follow the instructions provided by iTrust to submit a request for the dataset. You may be asked to provide information regarding your research or project to gain approval.

3. **Obtain Approval and Download**  
   Once your request is approved, you will receive instructions or links to download the dataset files. 

---

## 2. Required Files

After approval, you will have access to the following two Excel files from `SWat.A1 & A2_Dec_2015/Physical` folder:
1. **`SWaT_Dataset_Attack_v0.xlsx`**  
   Contains the attack data, capturing various sensor and actuator readings during simulated attack scenarios.

2. **`SWaT_Dataset_Normal_v1.xlsx`**  
   Contains the normal (non-attack) operational data of the SWaT system.


---

## 3. Using the Dataset

- **Download**: Once approved, download the two Excel files from the links provided by iTrust.  

- Place the downloaded Excel files in the following directory structure under `datasets/swat/` within your project:

```plaintext
project_root/
├── datasets/
│   └── swat/
│       ├── SWaT_Dataset_Attack_v0.xlsx
│       ├── SWaT_Dataset_Normal_v1.xlsx
│       ├── List_of_attacks_Final.xlsx
│       └── README.md
├── models/
│   └── aerca.py
├── requirements.txt
└── README.md
```

## Citation

If you use the SWaT dataset in your research, please cite the following paper:

> Goh, J., Adepu, S., Junejo, K. N., & Mathur, A. “A Dataset to Support Research in the Design of Secure Water Treatment Systems.” In *The 11th International Conference on Critical Information Infrastructures Security*.

Alternatively, you can use the following BibTeX entry:

```bibtex
@inproceedings{goh2016dataset,
  title={A Dataset to Support Research in the Design of Secure Water Treatment Systems},
  author={Goh, J. and Adepu, S. and Junejo, K. N. and Mathur, A.},
  booktitle={The 11th International Conference on Critical Information Infrastructures Security},
  year={2016}
}