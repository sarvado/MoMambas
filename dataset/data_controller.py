#data controller
import torch
from torch.utils import data
from dataset.dataset import *

class Data_Controller():

    def __init__(self, config):
        super(Data_Controller, self).__init__()

        self.config = config
        
        self.DEVICE=config["DEVICE"]

        #folder of all datasets
        folder=config["datasets_path"]

        dataset_name=config["dataset_name"]
        self.data_size=144
        if dataset_name[-2:]=='48':
            self.data_size=48
        
        #########################################################
        ### input parameter for training ###
        dataset_config={
            "BF3.2_FCC":1000,            
            "BF5_FCC":1000,
            "BF5_Amor":1000,
            "BF5_Amor_48":1000,
            "Pt_Extreme/BF3.2FCC":200,
            "Pt_Extreme/BF5FCC":200,
            "Pt_Extreme/BF5Amor":200,
            "Fe/Fe214":200,
            "Fe/Fe256":200,
            "Fe/Fe300":200
        }

        N_of_data=0;  # training set. number
        dataset_name=dataset_name.split(",")

        data_config={"name":[],"num":[],"start":[]}
        for d_name in dataset_name:
            data_config["name"].append(d_name)
            data_config["num"].append(dataset_config[d_name])
            data_config["start"].append(0)
            N_of_data += dataset_config[d_name]
        
        N_of_vdata=int(N_of_data/10);  # validation set, number
        N_of_tdata=int(N_of_data/10);  # test set, number          
        ###

        ### data loading setting ###
        batch_size=config["batch_size"]
        params1 = {'INPUT_INSIDE_NAME':'GF_Vol', 'TARGET_INSIDE_NAME':'target'}
        params2 = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 12}
        params3 = {'batch_size': 1, 'shuffle': True, 'num_workers': 12}

        N_of_start=1
        train_ID_set = range(N_of_start,N_of_start+N_of_data)
        validation_ID_set = range(N_of_start+N_of_data,N_of_start+N_of_data+N_of_vdata)
        test_ID_set = range(N_of_start+N_of_data+N_of_vdata,N_of_start+N_of_data+N_of_vdata+N_of_tdata)

        #dataset
        train_dataset = Dataset(train_ID_set,folder, data_config, 0, image_shift(4), **params1)
        validation_dataset = Dataset(validation_ID_set,folder, data_config, 1, image_shift(4), **params1)
        test_dataset = Dataset(test_ID_set,folder, data_config, 2, None, **params1)

        #generator
        self.train_generator = data.DataLoader(train_dataset, **params2)
        self.validation_generator = data.DataLoader(validation_dataset, **params3)
        self.test_generator = data.DataLoader(test_dataset, **params3)
        ###

    def get_data(self, inputs,target):
        size=inputs.shape[-1]
        inputs= inputs.view(-1,1,size,size,size).float().to(self.DEVICE);
        target= target.view(-1,1,size,size,size).float().to(self.DEVICE);
        return inputs,target
    
    def load_file(self, file, tag="GF_Vol"):
        input_data = scipy.io.loadmat(file)
        input_data = input_data.get(tag)
        if input_data is None:
            return None
        size=input_data.shape[-1]
        input_data= torch.tensor(input_data).view(-1,1,size,size,size).float().to(self.DEVICE);
        return input_data

    def output_file(self, file_path, outputs):
        outputs = outputs.data[0][0].cpu().numpy()
        scipy.io.savemat(file_path, {'output':outputs}) # save