# HCL-DTI

> **Hypergraph-Enhanced Contrastive Learning with Dynamic Negative Sampling for Robust Drug-Target Interaction Prediction**

HCL-DTI is a framework for DTI prediction that combines hypergraph-enhanced representation learning with contrastive learning and dynamic negative sampling.
---

## System Requirements

Ensure the following dependencies are installed:

- `torch~=2.3.1+cu121`
- `numpy~=1.24.4`
- `scipy~=1.10.1`
- `scikit-learn~=1.3.2`
- `pandas~=2.0.3`
- `tqdm~=4.66.4`

You can install dependencies via:
```bash
pip install -r requirements.txt
```
## Datasets

This work uses two benchmark datasets for drug–target interaction (DTI) prediction:

- **Dataset 1**: [DOI: 10.1038/s41467-017-00680-8](https://doi.org/10.1038/s41467-017-00680-8)  
- **Dataset 2**: [DOI: 10.1109/IJCNN.2018.8489028](https://doi.org/10.1109/IJCNN.2018.8489028)