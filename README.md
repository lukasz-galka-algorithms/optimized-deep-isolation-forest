# Optimized Deep Isolation Forest

This repository contains the implementation of the **Optimized Deep Isolation Forest (ODIF)** algorithm.

##  About
- The implementation includes both:
  - Optimized Deep Isolation Forest
  - Deep Isolation Forest
- The variant to be used (base or optimized) is controlled by a **optimization** hiperparameter, allowing seamless switching between versions.
- The default representation mechanism is **CERE (Computationally-Efficient deep Representation Ensemble)**, implemented using a **Multilayer Perceptron (MLP)** network.
- The repository also includes **datasets** used in the numerical experiments, sourced from the Anomaly Detection Benchmark (ADBench) framework:
```bibtex
@inproceedings{han2022adbench,  
      title={ADBench: Anomaly Detection Benchmark},   
      author={Songqiao Han and Xiyang Hu and Hailiang Huang and Mingqi Jiang and Yue Zhao},  
      booktitle={Neural Information Processing Systems (NeurIPS)}
      year={2022},  
}
