# MoMambas
Source code of "MoMambas"

## Download datasets:
* Download [datasets](https://pan.baidu.com/s/1S3-8fCwoMeaRs0u4tCPYFg) from BaiduNetdisk, extract with code: **aigf**.
* Extract each dataset according to the following paths:
```shell
# root path of the datasets 
datasets/
├── BF5_FCC 
├── BF3.2_FCC
├── BF5_Amor_48
├── BF5_Amor
├── Pt_Extreme/
|        ├── BF5FCC 
|        ├── BF3.2FCC
|        └── BF5Amor
└── Fe/
    ├── Fe214
    ├── Fe256
    └── Fe300
```

## Environment
```shell
% NVIDIA RTX A6000

% causal-conv1d             1.1.1                    
% mamba-ssm                 1.1.1                    
% matplotlib                3.8.4                    
% monai                     1.1.0                    
% numpy                     1.26.4                   
% opencv-python             4.10.0.84                
% python                    3.10.14              
% scipy                     1.13.0                   
% torch                     2.0.1                    
% torchvision               0.15.2 
```

## Train

### Config file path in source code folder:
```shell
source
   ├──config/
   |     └── config.json
```
### Configs:
```json
{
    "resume_state":"",
    "model_name":"MoMamba_EC", 
    "dataset_name":"BF5_FCC,BF5_Amor,BF3.2_FCC",
    "info":"[GPU:0]",
    "gpu_id":"0",
    "mom":{"num_experts":[3,3,3,3],"top_k":2,"emb_type":"PE","head":2,"use_aux_loss":false},
    "datasets_path":"/path of datasets",
    "batch_size":3,
    "total_batch":3,
    "N_epoch":100, 
    "output_folder":"/folder of user test tomograms",
    "test":false,
    "save_folder":"./experiments/"
}
```
### Config of Training Dataset:
* Train **single** datasets:
```json
"dataset_name":"BF5_FCC",
```
* Train **Pt Mixed**:
```json
"dataset_name":"BF5_FCC,BF5_Amor,BF3.2_FCC",
```
* Train **Pt Extreme**:
```json
"dataset_name":"Pt_Extreme/BF3.2FCC,Pt_Extreme/BF5FCC,Pt_Extreme/BF5Amor",
```
* Train **Fe MultiScale**:
```json
"dataset_name":"Fe/Fe214,Fe/Fe256,Fe/Fe300",
```
* Train **All Mixed**:
```json
"dataset_name":"BF5_FCC,BF5_Amor,BF3.2_FCC,Pt_Extreme/BF3.2FCC,Pt_Extreme/BF5FCC,Pt_Extreme/BF5Amor,Fe/Fe214,Fe/Fe256,Fe/Fe300",
```
### Config of MoM:
```json
"mom":{"num_experts":[3,3,3,3],"top_k":2,"emb_type":"PE","head":2,"use_aux_loss":false},
```
* "num_experts" : expert number of MoM on each stage
* "top_k" : top-k of MoM
* "emb_type" : "PE","LPE","RPE" (Corresponding to "SPE", "LPE", and "RoPE" in the paper) 
* "head" : head number of MHR
* "use_aux_loss" : use aux loss or not
### Other Variants:
```json
"model_name":"MoMamba_EC",
```
* "MoMamba_EC" : MoMambas with EC-backbone
* "MoMamba_Dn" : DnCNN-MoM in paper
### Run:
```python
    python main.py
```
A new folder will be created in experiments folder. 

## Test:

### Download Pre-trained Weight:
* Download [pre-trained](https://pan.baidu.com/s/1gYVsiJb7pCVjpWQOgw6wdA) from BaiduNetdisk, extract with code: **91s2**.
* Put the downloaded weights folder in "experiments" folder:
```shell
source
   ├──experiments/
   |     └── MoMamba_EC-AllMixed-05160128
```
### Set Test COnfig:
Set resume weights in config:
```json
"resume_state":"MoMamba_EC-AllMixed-05160128",
"output_folder":"/folder of user test tomograms",
"test":true,
```
* "resume_state" : folder of pre-trained weights, use relative path
* "output_folder" : folder of user test tomograms, use full path
### Run:
```python
    python main.py
```
A new folder will be created in experiments folder. 
