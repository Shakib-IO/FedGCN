# FedGCN

This work includes a reproduction of the research presented in the [FedGCN](https://arxiv.org/abs/2201.12433) 2022 paper.<br>
<br>
![Screenshot from 2022-11-01 14-49-55](https://user-images.githubusercontent.com/65369990/208219481-7655de5e-7cb7-49b9-b84b-1c0e9776728e.png)


## Specification of dependencies
This code requires Python 3.8.0 and Pytorch 1.12.1. Run the following to install the required packages.
```
conda update conda
conda env create -f environment.yml
conda activate FedGCN 
```

## Get dataset
First, open a folder named ```dataset``` in the root folder (mkdir dataset). Then, download Amazon dataset as well as the sribbles from [GitHub Releases](https://github.com/Shakib-IO/FedGCN/releases/tag/v0.1). Finally, unzip and set your path as ```prefix="dataset path"``` in the ```data_process.py``` file.

## Train & Evaluation code
To train and evaluate FedGCN on Amazon dataset, run:
```
python Central.py
```

All experiments are conducted on a single NVIDIA 3080 GPU.

### Acknowledgements
This code base is built on top of the following repository:
- https://github.com/yh-yao/FedGCN
