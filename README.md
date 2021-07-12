# Vision Transformer in Federated Learning 
* **Pytorch implementation for paper:** ["Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning"](https://arxiv.org/abs/2106.06047)
* Note that we simulate either CWT or FedAVG in one local machine for research usage, donot involve real communication between different clients. 

## Usage
### 0. Installation

- Run `cd ViT-FL-main`
- Install the libraries listed in requirements.txt 


### 1. Prepare Dataset 

We provide the data partitions for Cifar-10 and CelebA datasets 

- Cifar-10 dataset 
    * Download the three sets of simulated data partitions from https://drive.google.com/drive/folders/1ZErR7RMSVImkzYzz0hLl25f9agJwp0Zx?usp=sharing
    * Put the downloaded cifar10.npy at sub-folder data 
    
- For CelebA dataset (refer to https://leaf.cmu.edu/ for more usage of CelebA dataset)
    * Get the raw images at https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
    * Put the extracted the raw image folder at sub-folder data
    * Download the pre-processed data distributions at https://drive.google.com/drive/folders/1ZErR7RMSVImkzYzz0hLl25f9agJwp0Zx?usp=sharing
    * Put the downloaded CelebA.npy at sub-folder data 
   
- Retina dataset (Coming soon)

### 2. Download the Pretrained Models
- We use imagenet21k pre-train in our paper, download model from the following link and put it at ```--pretrained_dir``` folder

https://console.cloud.google.com/storage/browser/vit_models;tab=objects?prefix=&forceOnObjectsSortingFiltering=false


### 3. Train Model (refer to additional notes for usage of more tags)
- ViT-CWT train on Split-2, KS-0.65 of Cifar-10 and real world federated dataset CelebA  

```
python train_CWT.py --FL_platform ViT-CWT --dataset cifar10 --E_epoch 1 --max_communication_rounds 100 --split_type split_2 --save_model_flag
python train_CWT.py --FL_platform ViT-CWT --dataset CelebA --E_epoch 1 --max_communication_rounds 30 --split_type real

```

- ViT-FedAVG train on Split-2, KS-0.65 of Cifar-10 and real world federated dataset CelebA  


```
python train_FedAVG.py --FL_platform ViT-FedAVG --dataset cifar10 --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --save_model_flag
python train_FedAVG.py --FL_platform ViT-FedAVG --dataset CelebA --E_epoch 1 --max_communication_rounds 30 --num_local_clients 10 --split_type real

```

- All the checkpoints, results, log files will be saved to the ```--output_dir``` folder, with the final performance saved at log_file.txt 

## Additional Notes
- Some important tags for both ```train_CWT.py``` and ```train_FedAVG.py```:
    - ```--FL_platform```: selection of FL platforms, ViT-CWT or ResNet-CWT for ```train_CWT.py```, ViT-FedAVG or ResNet-FedAVG for ```train_FedAVG.py```  
    - ```--dataset```: choose of the following three datasets ["cifar10", "Retina" ,"CelebA"]
    - ```--save_model_flag```: set to True if need to save the checkpoints 
    - ```--output_dir```: the output directory where checkpoints/results/logs will be written 
    - ```--decay_type```: learning rate decay schedulers with the following three options ["cosine", "linear", "step"]
    - ```--E_epoch```: local training epoch E in FL train
    - ```--max_communication_rounds```: total communication rounds, 100 for Retina and Cifar-10, 30 for CelebA
    - ```--split_type```: type of data partitions, supports ["split_1", "split_2", "split_3"] for Cifar-10 and Retina, ["real"] for CelebA

- Additional tag for paralle FedAVG
    - ```--num_local_clients```: Num of local clients joined in each FL train. -1 (usage of all local clients) for Retina and Cifar-10, 10 for CelebA.  

- Also refer to the ```train_CWT.py``` and ```train_FedAVG.py``` for more tags


## Citations

```bibtex
@article{qu2021rethinking,
  title={Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning},
  author={Qu, Liangqiong and Zhou, Yuyin and Liang, Paul Pu and Xia, Yingda and Wang, Feifei and Fei-Fei, Li and Adeli, Ehsan and Rubin, Daniel},
  journal={arXiv preprint arXiv:2106.06047},
  year={2021}
}
```

## Acknowledgments
- Our ViT implementation is based on [Pytorch ViT implementation](https://github.com/jeonsworld/ViT-pytorch)
- Original ViT implementation at [Google ViT](https://github.com/google-research/vision_transformer)






