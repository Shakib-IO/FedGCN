# FedGCN

The FedGCN 2022 paper's work is reproduced in this work.<br>
<br>
![Screenshot from 2022-11-01 14-49-55](https://user-images.githubusercontent.com/65369990/208219481-7655de5e-7cb7-49b9-b84b-1c0e9776728e.png)



## Specification of dependencies
This code requires Python 3.8.0. Run the following to install the required packages.
```
conda update conda
conda env create -f environment.yml
conda activate FedGCN 
```

## Get datasets
First, open a folder named ``dataset``` in the root folder (mkdir datasets). Then, download Amazon dataset as well as the sribbles from [GitHub Releases](https://github.com/Shakib-IO/FedGCN/releases/tag/v0.1). Finally, unzip and move the four folder to datasets.


### Acknowledgements
This code base is built on top of the following repositorie:
- https://github.com/yh-yao/FedGCN
