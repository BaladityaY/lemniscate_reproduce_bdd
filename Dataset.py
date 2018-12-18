import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from random import shuffle
import os
from bdd_tools import BDD_Helper
from docutils.nodes import image
import random
import bdd_tools

def get_device(device_id = 0):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(device_id)
        torch.cuda.device(device_id)
        return device
    else:
        device = torch.device("cpu")
        device_name = "cpu"
        return device

class Data_Moment():
    
    def __init__(self, images, speeds, start_index, stop_index, filename):
        
        self.images = images        
        self.speeds = speeds
        
        self.filename = filename
        
        self.start_index = start_index
        self.stop_index = stop_index
    
    def convert_images(self, encoded_images):
        return [cv2.imdecode(np.fromstring(encoded_img, dtype=np.uint8), -1) for encoded_img in encoded_images]      
    
    def data_point(self):     
        return {'imgs':self.convert_images(self.images[self.start_index:self.stop_index][np.arange(self.stop_index - self.start_index)]),  
                'speeds':self.speeds[self.start_index:self.stop_index][np.arange(self.stop_index - self.start_index)]}
    

class Dataset(data.Dataset):
    
    def sort_folder_ft(self, s):
        '''
        Returns the last two entries, file name and last folder name, as key to sort
        '''
        return s.split('/')[-2]+'/'+s.split('/')[-1]
        
    def sort_filelist(self,data_folder_dir):
        
        file_list = []
        for path, subdirs, files in os.walk(data_folder_dir,followlinks=True):
            for file_name in files:
                if file_name.endswith('h5'):
                    filename = os.path.join(path,file_name)
                    
                    file_list.append(filename)
                            
        return sorted(file_list,key=self.sort_folder_ft)
        
        
    
    
    def __init__(self, data_folder_dir, n_frames = 1,framerate=1):
        
        self.run_files = []
        
        for filename in self.sort_filelist(data_folder_dir):

            print("Processing {} ".format(filename))
                       
            database_file = h5py.File(filename, 'r')                        
            
            images = database_file['image']['encoded']
            speeds = np.reshape(database_file['image']['speeds'][:], [-1, 2])
            
            for i in range(len(images)):
                moment = Data_Moment(images, speeds, i, i + n_frames, filename)   
                self.run_files.append(moment)
                
        
    def __len__(self):
        return len(self.run_files)

    def __getitem__(self, index):
        data_moment = self.run_files[index]
        camera_data = torch.FloatTensor().to(get_device())
        
        # In this version of the data set there is only one frame
        frame = 0
        
        img = torch.FloatTensor(data_moment.data_point()['imgs'][frame]).to(get_device())
        camera_data = torch.cat((camera_data, img), 2)
             
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)
        
        speeds = torch.FloatTensor(data_moment.data_point()['speeds'][frame]).to(get_device())
        speeds = BDD_Helper.
        
        return camera_data, speeds, index
    
    @property
    def train_labels(self):
        return np.array(range(len(self.run_files)))

if __name__ == '__main__':
    import cv2
    train_dataset = Dataset("/home/sascha/for_bdd_training/smaller_dataset/train",6)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (images, speeds, index) in enumerate(train_loader):
        img = images[0].data.cpu().numpy()
        img = img.transpose((1,2,0))+0.5
        cv2.imshow("Test", img)
        cv2.waitKey(30)
    
