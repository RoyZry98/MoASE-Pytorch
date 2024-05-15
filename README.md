# Decomposing the Neurons: Activation Sparsity via Mixture of Experts for Continual Test Time Adaptation

## Installation

Please create and activate the following conda envrionment. 
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate vida 
```

## Classification Experiments

* **ViT** as the backbone

Our source model is from timm, you can directly donwload it from the code.


### Cifar10-to-Cifar10C task 
Please load the source model from [here](https://drive.google.com/file/d/1pAoz4Wwos74DjWPQ5d-6ntyjQkmp9FPE/view?usp=sharing)

```bash
cd cifar
bash ./bash/cifar10/source_vit.sh # Source model directly test on target domain
bash ./bash/cifar10/tent.sh # Tent 
bash ./bash/cifar10/cotta.sh # CoTTA
bash ./bash/cifar10/vit.sh # MoASE
```

### Cifar100-to-Cifar100C task 
Please load the source model from [here](https://drive.google.com/file/d/1yRekkpkIdwX_LFsOh4Ba9ndaECnY-UC-/view?usp=sharing)

```bash
cd cifar
bash ./bash/cifar100/source_vit.sh # Source model directly test on target domain
bash ./bash/cifar100/tent.sh # Tent 
bash ./bash/cifar100/cotta.sh # CoTTA
bash ./bash/cifar100/vit.sh # MoASE
```

For segmentation code, you can refer to [cotta](https://github.com/qinenergy/cotta) and [SVDP](https://github.com/Anonymous-012/SVDP). As for the source model, you can directly use Segformer trained on Cityscapes.
## Citation
Please cite our work if you find it useful.
```bibtex

```

## Acknowledgement 
+ CoTTA code is heavily used. [official](https://github.com/qinenergy/cotta) 
+ KATANA code is used for augmentation. [official](https://github.com/giladcohen/KATANA) 
+ Robustbench [official](https://github.com/RobustBench/robustbench) 

## Data links
+ ImageNet-C [Download](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

