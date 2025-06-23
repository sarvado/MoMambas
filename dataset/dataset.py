from array import array
from random import random
from torch.utils import data
import scipy.io
import numpy as np
import copy

### define image trainslation ###
class image_shift(object):

    def __init__(self, translation_range=0):
        #assert isinstance(translation_range, (float, tuple))
        self.translation_range = translation_range

    def __call__(self, input_data, target_data):
        inputs, target = input_data, target_data
        
        dx, dy, dz = np.random.randint(self.translation_range*2+1, size=3)-self.translation_range
        
        inputs = np.roll(inputs, dz, axis=0)
        inputs = np.roll(inputs, dy, axis=1)
        inputs = np.roll(inputs, dx, axis=2)
        
        target = np.roll(target, dz, axis=0)
        target = np.roll(target, dy, axis=1)
        target = np.roll(target, dx, axis=2)
        
        if dz>0:
            inputs[:,:,:dz] = 0
            target[:,:,:dz] = 0
        elif dz<0:
            inputs[:,:,dz:] = 0
            target[:,:,dz:] = 0
        if dy>0:
            inputs[:,:dy,:] = 0
            target[:,:dy,:] = 0
        elif dy<0:
            inputs[:,dy:,:] = 0
            target[:,dy:,:] = 0
        if dx>0:
            inputs[:dx,:, :] = 0
            target[:dx,:, :] = 0
        elif dx<0:
            inputs[dx:, :,:] = 0
            target[dx:, :,:] = 0
        return inputs, target
###


### For data loading ###
class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, folder,data_config,start_stage,Transform=None,INPUT_INSIDE_NAME='GF_Vol',TARGET_INSIDE_NAME='target'):
        'Initialization'
        self.list_IDs = list_IDs
        self.folder=folder
        self.data_config=copy.deepcopy(data_config)
        self.transform = Transform
        self.input_inside_name = INPUT_INSIDE_NAME
        self.target_inside_name = TARGET_INSIDE_NAME

        if start_stage > 0:
            for i in range(len(self.data_config["num"])):
                self.data_config["num"][i] = int(self.data_config["num"][i]/10)
                self.data_config["start"][i] = int(self.data_config["num"][i]*(9+start_stage))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def getDatasetAndID(self, ID):
        if len(self.data_config["num"]) == 1:
            return self.data_config["name"][0], ID

        inner_ID= ID-self.list_IDs.start + 1
        d_name=""
        for i in range(len(self.data_config["num"])):
            d_num = self.data_config["num"][i] 
            if (inner_ID - d_num) >= 1:
                inner_ID-=d_num
            else:
                d_name=self.data_config["name"][i]
                inner_ID += self.data_config["start"][i]
                break
        return d_name, inner_ID 

    def __getitem__(self, index):
        # ID_set
        ID = self.list_IDs[index]
        'Generates one sample of data'
        # Load data and get target_data for matlab file

        dataset, ID = self.getDatasetAndID(ID)
        #print("{}:{}".format(dataset,ID))

        INPUT_PATH=self.folder+dataset+"/input"
        TARGET_PATH=self.folder+dataset+"/target"   
 
        atomType="Pt"
        if dataset[:2] == "Fe":
            atomType="Fe" 
        input_file_name= atomType+"_input"
        target_file_name=atomType+"_target"  

        input_data = scipy.io.loadmat('{}/{}_{}.mat'.format(INPUT_PATH,input_file_name,ID))[self.input_inside_name];
        target_data = scipy.io.loadmat('{}/{}_{}.mat'.format(TARGET_PATH,target_file_name,ID))[self.target_inside_name];

        if self.transform:
            input_data, target_data = self.transform(input_data, target_data)

        index = "{}_{}".format(dataset,ID)
        return input_data, target_data, index