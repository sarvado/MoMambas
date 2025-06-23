# MoMambas
Source code of "MoMambas"

## Datasets

### From Baidu net disk:
    link:  https://pan.baidu.com/s/1yjDck2m8ss31ifwhIVSyUw 
    key: kikx 
### From Goolgle disk


### Download datasets:
```shell
# root path of the datasets 
datasets/
├── BF5_FCC 
├── BF3.2_FCC
├── BF5_Amor_48
├── BF5_Amor
├── Pt_Extreme
|        ├── BF5FCC 
|        ├── BF3.2FCC
|        └── BF5Amor
└── Fe
    ├── Fe214
    ├── Fe256
    └── Fe300
```



## Train

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
## Training config

### Config file path in source code:
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
    "total_batch":3,//batch_size*n_batch_size
    "N_epoch":100,    // epoch
    "output_folder":"/path of test tomograms",
    "test":false,
    "save_folder":"./experiments/"
}
```
### Training Config of Dataset:
Train single datasets:
```json
"dataset_name":"BF5_FCC",
```
Train "Pt Mixed":
```json
"dataset_name":"BF5_FCC,BF5_Amor,BF3.2_FCC",
```
Train "Pt Extreme":
```json
"dataset_name":"Pt_Extreme/BF3.2FCC,Pt_Extreme/BF5FCC,Pt_Extreme/BF5Amor",
```
Train "Fe MultiScale":
```json
"dataset_name":"Fe/Fe214,Fe/Fe256,Fe/Fe300",
```
Train "All Mixed":
```json
"dataset_name":"BF5_FCC,BF5_Amor,BF3.2_FCC,Pt_Extreme/BF3.2FCC,Pt_Extreme/BF5FCC,Pt_Extreme/BF5Amor,Fe/Fe214,Fe/Fe256,Fe/Fe300",
```
### Resume Config of Dataset:
1.Put the resume folder in experiments folder:
```shell
source
   ├──experiments/
   |     └── MoMamba_EC-BF5_FCC(3)-06201419
```
2.Set config:
```json
"resume_state":"MoMamba_EC-BF5_FCC(3)-06201419",
```
### MoM Config:
```json
"mom":{"num_experts":[3,3,3,3],"top_k":2,"emb_type":"PE","head":2,"use_aux_loss":false},

% num_experts: expert number of MoM on each stage
% top_k : top-k of MoM
% emb_type : "PE","LPE","RPE" (Corresponding to "SPE", "LPE", and "RoPE" in the paper) 
% head : head number of MHR
% use_aux_loss : use aux loss or not
```
### Other Variants of MoMambas:
```json
"model_name":"MoMamba_EC",

% MoMamba_EC : MoMambas with EC-backbone
% MoMamba_Dn : DnCNN-MoM in paper
% MoMamba_MLP':MoM with MLP as experts
``` 
### Run:
```python
    python main.py
```

### Pre-trained weights:
