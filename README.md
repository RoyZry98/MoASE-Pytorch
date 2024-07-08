# Decomposing the Neurons: Activation Sparsity via Mixture of Experts for Continual Test Time Adaptation
![Python 3.9](https://img.shields.io/badge/Python-3.9-red)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2405.16486)

<img src="MoASE.png"/>

## Installation

Please create and activate the following conda envrionment. 
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate moase 
```

## Classification Experiments

* **ViT** as the backbone

Our source model is from timm, you can directly donwload it from the code.


### Cifar10-to-Cifar10C task 
Please load the source model from [here](https://drive.google.com/file/d/1pAoz4Wwos74DjWPQ5d-6ntyjQkmp9FPE/view?usp=sharing)

```bash
bash run_cifar10.sh # MoASE
```

### Cifar100-to-Cifar100C task 
Please load the source model from [here](https://drive.google.com/file/d/1yRekkpkIdwX_LFsOh4Ba9ndaECnY-UC-/view?usp=sharing)

```bash
cd cifar
bash run_cifar100.sh # MoASE
```

For segmentation code, you can refer to [cotta](https://github.com/qinenergy/cotta) and [SVDP](https://github.com/Anonymous-012/SVDP). As for the source model, you can directly use Segformer trained on Cityscapes.

## Citation
Please cite our work if you find it useful.
```bibtex
@article{zhang2024decomposing,
  title={Decomposing the Neurons: Activation Sparsity via Mixture of Experts for Continual Test Time Adaptation},
  author={Zhang, Rongyu and Cheng, Aosong and Luo, Yulin and Dai, Gaole and Yang, Huanrui and Liu, Jiaming and Xu, Ran and Du, Li and Du, Yuan and Jiang, Yanbing and others},
  journal={arXiv preprint arXiv:2405.16486},
  year={2024}
}
```

## Acknowledgement 
+ CoTTA code is heavily used. [official](https://github.com/qinenergy/cotta) 
+ KATANA code is used for augmentation. [official](https://github.com/giladcohen/KATANA) 
+ Robustbench [official](https://github.com/RobustBench/robustbench) 

## Data links
+ ImageNet-C [Download](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

