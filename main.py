import os
import argparse
from config.config import *

if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser(description="Volume denoising network")
    parser.add_argument('-c', '--config', type=str, default='config/config.json',
                            help='JSON file for configuration')
    args = parser.parse_args()
    
    #config
    config = get_config(args.config)

    paraller_model=False
    if config["gpu_id"] !='':
        os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu_id"] 

        if len(config["gpu_id"].split(','))>1:
            paraller_model=True

    import torch
    import torch.nn as nn
    import torch.optim as optim 
    import time
    from datetime import datetime

    from fvcore.nn import FlopCountAnalysis

    #
    from common import *
    from dataset.data_controller import Data_Controller

    #init
    N_epoch=config["N_epoch"]     # epoch
    save_folder_=config['save_folder']

    # resume state
    resume_state = config['resume_state']
    resume_content, model_name,dataset_name,start_epoch,mom = resume(resume_state, save_folder_)
    if resume_content=="":
        model_name=config["model_name"]
        dataset_name = config["dataset_name"]
    else:
        config["model_name"]=model_name
        config["dataset_name"]=dataset_name 
        if mom !="":
            config["mom"]=mom      

    #save folder
    d_name = dataset_name.replace('/','_').split(",")
    save_name='{}-{}({})-{}'.format(model_name, d_name[0],int(len(d_name)),datetime.now().strftime('%m%d%H%M'))    
    save_folder=os.path.join(save_folder_, save_name)
    if os.path.exists(save_folder) ==False:
        os.makedirs(save_folder)
    save_path=save_folder+"/model.pth"
    #log
    logger=get_logger(save_folder,"logging")
    
    # checking that cuda is available or not # 
    USE_CUDA=torch.cuda.is_available()
    DEVICE=torch.device("cuda" if USE_CUDA else "cpu")
    logger.info("CUDA: {}".format(USE_CUDA))
    config["DEVICE"]=DEVICE

    #data controller
    data_controller = Data_Controller(config)

    # get model 
    logger.info("model contructing: %s" %(save_name))
    model=get_model(model_name)

        # MoM settings
    if model_name[:3]=="MoM":
        logger.info("MoM settings: {}".format(json.dumps(config["mom"])))
        model=model(in_channels=1,out_channels=1,config=config).to(DEVICE)
    else:
        model=model(in_channels=1,out_channels=1,data_size=data_controller.data_size).to(DEVICE)
    
    logger.info("Training Info: {}".format(config['info']))

    # resume model
    resume_state_path = os.path.join(save_folder_, resume_state, "model.pth")
    if os.path.exists(resume_state_path):
        logger.info("resume state: %s" %resume_state)        
        model.load_state_dict(torch.load(resume_state_path))
        model.to(DEVICE)  


 
    #check flops
    if True:
        model.return_aux_loss = False
        checkFLOPs(model,data_controller.data_size,logger)
        model.return_aux_loss = True

    if paraller_model:
        model=nn.DataParallel(model)

    # define loss function & optimizer #
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    #train
    if start_epoch < N_epoch:
        #resume logger
        if resume_content!="":
            logger.info("resume logger: ")
            logger.info(resume_content)
        else:
            logger.info("Dataset:{}".format(dataset_name))

        logger.info("Training starts: start epoch {}".format(start_epoch))
        total_startTime = time.time()
        for epoch in range(start_epoch, N_epoch):  # loop over the dataset multiple times   
            startTime = time.time()            
            # train model
            train(model,data_controller,criterion,optimizer)
            
            endTime = time.time() - startTime

            # calculating loss of training set & validation set
            #loss_mean_test,_,_=eval(model,data_controller,criterion,data_controller.train_generator)
            #logger.info('[epoch: %d, %3d %%] training set loss: %.10f '  %(epoch + 1, (epoch + 1)/N_epoch*100 , loss_mean_test))
            
            #loss_mean_test,psnr_mean,ssim_mean=eval(model,data_controller,criterion,data_controller.validation_generator)
            loss_mean_test,psnr_mean,ssim_mean=eval(model,data_controller,criterion,data_controller.test_generator)
            logger.info('[epoch: %d, %3d %%] test set time: %.3f '  %(epoch + 1, (epoch + 1)/N_epoch*100  ,endTime))
            logger.info('test set rmse: %.10f,  psnr:%.10f, ssim:%.10f'  %(loss_mean_test, psnr_mean, ssim_mean))

            save(model, save_path,logger, paraller_model)

        total_endTime = time.time() - total_startTime
        logger.info('Training has been finished')
        logger.info('Total time: %.3f'  %(total_endTime))

    #test
    if config["test"]==True: 
        loss_mean_test,psnr_mean,ssim_mean=eval(model,data_controller,criterion,data_controller.test_generator,logger)
        logger.info('test set rmse: %.10f,  psnr:%.10f, ssim:%.10f'  %(loss_mean_test, psnr_mean, ssim_mean))

    if config["output_folder"]!="":
        output_folder(model,data_controller,save_folder,config["output_folder"])