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
    
    def __init__(self, images, speeds, actions, start_index, stop_index, filename):
        
        self.images = images
        self.actions = actions
        self.speeds = speeds
        
        self.filename = filename
        
        self.start_index = start_index
        self.stop_index = stop_index
        
    @property
    def data_point(self):     
        return {'imgs':self.images[self.start_index:self.stop_index], 
                'actions':self.actions[self.start_index:self.stop_index], 
                'speeds':self.speeds[self.start_index:self.stop_index]}
    
    def __len__(self):
        # This step introduced quite a delay before training. Maybe there is a better way to check
        # if the current data moment is long enough
        if self.stop_index > len(self.images):
            return len(self.images) - self.start_index 
        else:
            return self.stop_index - self.start_index 

class Dataset(data.Dataset):
    
    def sort_folder_ft(self, s):
        '''
        Returns the last two entries, file name and last folder name, as key to sort
        '''
        return s.split('/')[-2]+'/'+s.split('/')[-1]
        
    def sort_filelist(self,data_folder_dir):
        
        file_list = []
        print data_folder_dir
        for path, subdirs, files in os.walk(data_folder_dir,followlinks=True):
            for file_name in files:
                if file_name.endswith('h5'):
                    filename = os.path.join(path,file_name)
                    
                    file_list.append(filename)
                            
        return sorted(file_list,key=self.sort_folder_ft)
        
        
    def convert_images(self, encoded_images):
        return [cv2.imdecode(np.fromstring(encoded_img, dtype=np.uint8), -1) for encoded_img in encoded_images]      
    
    
    def __init__(self, data_folder_dir, n_frames):
        
        self.run_files = []
        self.n_frames = n_frames
        self.all_action_bins = np.zeros(6)
        
        for filename in self.sort_filelist(data_folder_dir):
            print "Processing {} ".format(filename)
           
            database_file = h5py.File(filename, 'r')                        
            
            images = self.convert_images(database_file['image']['encoded'])
            speeds = np.reshape(database_file['image']['speeds'][:], [-1, 2])
            actions = BDD_Helper.turn_future_smooth(speeds, 5, 0)
            
            print('actions shape: {}'.format(actions.shape))
            #print('actions: {}'.format(actions))
            
            for i in range(len(images)):
                if i + n_frames >= actions.shape[0]:
                    continue
                
                moment = Data_Moment(images, speeds, actions, i, i + n_frames, filename)
   
                action_i = actions[i+2:i+3, :][0]
                action_i[action_i > 0] = 1.
                ind_to_change = np.where(action_i == 1.)[0][0]

                #print('ind {} for array: {}'.format(ind_to_change, action_i))
                
                if self.all_action_bins[ind_to_change] > 50 + np.min(self.all_action_bins):
                    continue
                
                self.all_action_bins[ind_to_change] = self.all_action_bins[ind_to_change] + 1
   
                if len(moment) == n_frames:
                    self.run_files.append(moment)
                else:
                    pass
                        # #print "dark frame detected" # or just trying to
                
    def __len__(self):
        return len(self.run_files)

    def __getitem__(self, index):
        data_moment = self.run_files[index]
        camera_data = torch.FloatTensor().to(get_device())
        
        for frame in range(self.n_frames): 
            
            img = torch.FloatTensor(data_moment.data_point['imgs'][frame]).to(get_device())
            camera_data = torch.cat((camera_data, img), 2)
             
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)

        all_actions = torch.from_numpy(data_moment.data_point['actions'][:]).float().to(get_device()) 
        
        return camera_data, all_actions, index
    
    @property
    def train_labels(self):
        return np.array(range(len(self.run_files)))

if __name__ == '__main__':
    import cv2
    train_dataset = Dataset("/home/sascha/for_bdd_training/tiny_test_set",6)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (images, actions, index) in enumerate(train_loader):
        print actions
        img = images[0][6:9].data.cpu().numpy()
        img = img.transpose((1,2,0))+0.5
        cv2.imshow("Test", img)
        cv2.waitKey(60)
    
