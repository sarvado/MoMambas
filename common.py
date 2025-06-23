import torch

import os
import logging
import numpy as np
from torch import nn
import ast

#models
from models.momamba.MoMamba_EC import MoMamba_EC
from models.momamba.MoMamba_Dn import MoMamba_Dn
from models.momamba.MoMamba_MLP import MoMamba_MLP

from monai import metrics
from monai.metrics.regression import SSIMMetric,PSNRMetric
from fvcore.nn import FlopCountAnalysis

 

#get model by name
def get_model(name):
    return {
        'MoMamba_EC':MoMamba_EC,
        'MoMamba_Dn':MoMamba_Dn,
        'MoMamba_MLP':MoMamba_MLP
    }[name]

def get_logger(log_dir, name):
    logger = logging.getLogger(name)
    file_path = "{}/{}.log".format(log_dir, name)
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def metric(outputs,targets):
        x = targets # ground truth
        y = outputs # prediction
        data_range = x.max().unsqueeze(0)

        ssim = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(x,y)
        psnr = PSNRMetric(max_val=data_range,reduction="mean")._compute_metric(x,y)
        return psnr,ssim

#output folder
def output_folder(model,data_controller, save_folder,folder):
    model.eval()
    with torch.no_grad():
        volNames = ['ESTvol','vol','GF_Vol']
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            save_path = os.path.join(save_folder, "output_"+file_name)
            if os.path.isfile(file_path):                
                print(file_name)

                inputs=None
                for volName in volNames:          
                    inputs=data_controller.load_file(file_path, volName)
                    if inputs!=None:
                        break

                outputs = model(inputs)

                if type(outputs) == list:
                    outputs =outputs[0]

                data_controller.output_file(save_path, outputs)

#resume
def resume(resume_state, save_folder_):
    content =""
    start_epoch =0
    model_name=""
    dataset_name=""
    mom=""
    
    #load logger
    if resume_state !="":
        resume_state = os.path.join(save_folder_,resume_state)
        logger_file = os.path.join(resume_state,"logging.log")
        if os.path.exists(logger_file):
            with open(logger_file, "r") as file:
                content = file.read()

                #model
                eid = content.find("model contructing: ")
                if eid!=-1:
                    model_name = content[eid+len("model contructing: "):]
                    model_name = model_name[:model_name.find("-")]

                #dataset_name
                eid = content.find("Dataset:")
                if eid!=-1:
                    dataset_name = content[eid+len("Dataset:"):]
                    dataset_name = dataset_name[:dataset_name.find("\n")]


                #start epoch
                eid = content.rfind("epoch: ")
                if eid!=-1:
                    start_epoch=content[eid+len("epoch: "):]
                    start_epoch=start_epoch[:start_epoch.find(",")]
                    start_epoch = int(start_epoch)

                #mom
                eid = content.find("MoM settings:")
                if eid!=-1:
                    mom = content[eid+len("MoM settings:"):]
                    mom = mom[:mom.find('\n')].strip()
                    mom = ""

    return content, model_name,dataset_name,start_epoch,mom

#eval model
def eval(model,data_controller,criterion,generator,logger=None):
    model.eval()
    with torch.no_grad():
        loss_sum_test=0
        loss_all=np.array((0,0,0,0,0))
        sum_psnr=0
        sum_ssim=0
        for j, (inputs, targets, index) in enumerate(generator):
            inputs,targets=data_controller.get_data(inputs,targets)
            outputs = model(inputs)

            if type(outputs) == list:
                 outputs =outputs[0]

            loss_test = criterion(outputs, targets)
            loss_test = (loss_test.item())**0.5 # MSE -> RMSE
            loss_sum_test += loss_test            
            if logger!=None:
                logger.info("{}:{}".format(index[0],loss_test))

            if True:
                psnr,ssim=metric(outputs,targets)
                sum_psnr+=psnr.item()
                sum_ssim+=ssim.item()
        loss_all=loss_all/(j+1)
        #print(loss_all)
        return loss_sum_test/(j+1),sum_psnr/(j+1),sum_ssim/(j+1)

#train model
def train(model,data_controller,criterion,optimizer):
    running_loss = 0.0
    model.train()

    accumulation_step=0
    n_batch_size=int(data_controller.config["total_batch"]/data_controller.config["batch_size"])

    for i, (inputs, targets,index) in enumerate(data_controller.train_generator):
        # input & target data
        inputs,targets=data_controller.get_data(inputs,targets)

        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        aux_loss = 0
        if type(outputs) == list:
             outputs, aux_loss=outputs[0],outputs[1]
             
        loss = criterion(outputs, targets)
        loss += aux_loss

        #accumulation loss
        if n_batch_size>1:
            accumulation_train_loss=loss/n_batch_size
            accumulation_train_loss=accumulation_train_loss.mean()
            loss=accumulation_train_loss

        loss.backward()

        update_optimizer=False
        if n_batch_size==1:
            update_optimizer=True
        else:
            accumulation_step=accumulation_step+1
            if accumulation_step % n_batch_size == 0 or accumulation_step == len(data_controller.train_generator):
                update_optimizer=True

        if update_optimizer==True:
            optimizer.step()
            optimizer.zero_grad()

        # train imformation
        running_loss += (loss.item())**0.5

def save(model, save_path,logger,parallel_model):
        # save the result #
        if parallel_model:
             torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)
        logger.info('saving model: %s' %(save_path))

def checkFLOPs(model,data_size,logger):
    #params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_res = (1, data_size, data_size, data_size)
    input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
                                        device=next(model.parameters()).device)
    flops = FlopCountAnalysis(model, input)
    model_flops = flops.total()
    logger.info(f"Total trainable parameters: {round(n_parameters * 1e-6, 2)} M")
    logger.info(f"MAdds: {round(model_flops * 1e-9, 2)} G")  


