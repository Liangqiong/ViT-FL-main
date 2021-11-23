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
    * Put the extracted raw image folder at sub-folder data
    * The pre-processed data distributions is provided CelebA.npy at sub-folder data

- Retina dataset (Coming soon)

### 2. Set (download) the Pretrained Models
- We use ImageNet1k pre-train in our paper 
- For ViTs: To use ImageNet1K pretrained models for ViTs, please modify the loading link of pretrained models in timm mannually (modify the link setting of default_cfgs = { } in the timm/models/vision_transformer.py file): 
    * For ViT(T),   
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
        'Ti_16-i1k-300ep-lr_0.001-aug_light0-wd_0.1-do_0.0-sd_0.0.npz'),
    * For ViT(S), 
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz'), 
    * For ViT(B), 
        'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1.npz'),
- For Swin-Transformer: Please refer to https://github.com/microsoft/Swin-Transformer for the pretrained models, download the  model and put it at ```--pretrained_dir``` folder


### 3. Train Model (refer to additional notes for usage of more tags)
- ViT-CWT train on Split-2, KS-0.65 of Cifar-10 and real world federated dataset CelebA  

```
python train_CWT.py --FL_platform ViT-CWT --net_name ViT-small --dataset cifar10 --E_epoch 1 --max_communication_rounds 100 --split_type split_2 --save_model_flag
python train_CWT.py --FL_platform ViT-CWT --net_name ViT-small --dataset CelebA --E_epoch 1 --max_communication_rounds 30 --split_type real

```

- ViT-FedAVG train on Split-2, KS-0.65 of Cifar-10 and real world federated dataset CelebA  


```
python train_FedAVG.py --FL_platform ViT-FedAVG --net_name ViT-small --dataset cifar10 --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1 --split_type split_2 --save_model_flag
python train_FedAVG.py --FL_platform ViT-FedAVG --net_name ViT-small --dataset CelebA --E_epoch 1 --max_communication_rounds 30 --num_local_clients 10 --split_type real

```

- All the checkpoints, results, log files will be saved to the ```--output_dir``` folder, with the final performance saved at log_file.txt 

## Additional Notes
- Some important tags for both ```train_CWT.py``` and ```train_FedAVG.py```:
    - ```--FL_platform```: selection of FL platforms, ViT-CWT, ResNet-CWT, EfficientNet-CWT, or Swin-CWT for ```train_CWT.py```, ViT-FedAVG, ResNet-FedAVG, EfficientNet-FedAVG, or Swin-FedAVG for ```train_FedAVG.py```  
    - ```--net_name```: basic Name of this run, also providing detailed network-architecture for ViT/ResNet/EfficientNet. For ViT: ViT-small, ViT-tiny, ViT-base(default), For EfficientNet: efficientnet-b1, efficientnet-b5(default), efficientnet-b7 see sstart_config.py for more details 
    - ```--dataset```: choose of the following three datasets ["cifar10", "Retina" ,"CelebA"]
    - ```--save_model_flag```: set to True if need to save the checkpoints 
    - ```--output_dir```: the output directory where checkpoints/results/logs will be written 
    - ```--decay_type```: learning rate decay schedulers with the following three options ["cosine", "linear", "step"]
    - ```--E_epoch```: local training epoch E in FL train
    - ```--max_communication_rounds```: total communication rounds, 100 for Retina and Cifar-10, 30 for CelebA
    - ```--split_type```: type of data partitions, supports ["split_1", "split_2", "split_3"] for Cifar-10 and Retina, ["real"] for CelebA
    - ```--cfg```: configuration document for Swin-transformers if use Swin-FL, otherwise ignored it

- Additional tag for paralle FedAVG
    - ```--num_local_clients```: Num of local clients joined in each FL train. -1 (usage of all local clients) for Retina and Cifar-10, 10 for CelebA.  



- Also refer to the ```train_CWT.py``` and ```train_FedAVG.py``` for more tags

## Acknowledgments
- Our Swin-Transformer implementation is based on [Pytorch Swin implementation](https://github.com/microsoft/Swin-Transformer)
- ViT implementation is based on https://github.com/rwightman/pytorch-image-models
- Original ViT implementation at [Google ViT](https://github.com/google-research/vision_transformer)






