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
    
    def __init__(self, images, speeds, start_index, stop_index, filename, framerate=1):
        
        self.images = images
        self.speeds = speeds
        
        self.filename = filename
        
        self.start_index = start_index
        self.stop_index = stop_index
        self.framerate = framerate        
        
    def convert_images(self, encoded_images):
        return [cv2.imdecode(np.fromstring(encoded_img, dtype=np.uint8), -1) for encoded_img in encoded_images]      
    
    def data_point(self):     
        # First reformat speeds to pairs from one original sequence which has twice the length
        # as the images
        speeds = np.reshape(self.speeds[self.start_index:self.stop_index*2], [-1, 2])
        # Then retrieve every n+th pair, depending on the framerate
        speeds = speeds[np.arange(self.stop_index-self.start_index)*self.framerate]
        # For images, just select it appropriately for the given range and framerate
        images = self.images[self.start_index:self.stop_index]
        images = images[np.arange(self.stop_index-self.start_index)*self.framerate]
        
        return {'imgs':self.convert_images(images),  
                'speeds':speeds}
    

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
    
    def __init__(self, data_folder_dir, n_frames):
        
        self.run_files = []
        self.n_frames = n_frames

        for filename in self.sort_filelist(data_folder_dir):

            print "Processing {} ".format(filename)
           
            database_file = h5py.File(filename, 'r')                        
            
            images = database_file['image']['encoded']
            # Note that speeds is twice the length of images because there are two values for each image.
            # However, if that is reformatted here, then this won't save hdf5 dataset references but instead
            # numpy arrays which is too costly in terms of speed and memory
            speeds = database_file['image']['speeds']
            
            for i in range(len(images)):
                if i + n_frames >= actions.shape[0]: # Not enough frames left for a full data moment 
                    continue

                moment = Data_Moment(images, speeds, i, i + n_frames, filename)
   
                self.run_files.append(moment)
                


        min_count = np.min(self.all_action_bins[self.all_action_bins > 0]) # take min of non-zero bin counts

        print('mean: {}, std: {}, min: {}'.format(np.mean(self.all_action_bins), np.std(self.all_action_bins), min_count))

        print(self.all_action_bins)
        
        new_run_files = []
        new_counts = np.zeros(6)
        for rf_ind, run_file in enumerate(self.run_files):
            action_ind = self.action_inds[rf_ind]

            if new_counts[action_ind] >= min_count*3/2:
                continue


            '''
            if self.all_action_bins[action_ind] > 10*min_count and random.uniform(0,1) < .1:
                print('skipped')
                continue
            '''
            
            new_run_files.append(run_file)
            new_counts[action_ind] = new_counts[action_ind] + 1

        self.run_files = new_run_files

        '''
        f = plt.figure()
        f.add_subplot(1,2,1)
        plt.plot(range(len(self.all_action_bins)), self.all_action_bins)#; plt.show()
                        
        f.add_subplot(1,2,2)
        plt.plot(range(len(new_counts)), new_counts); plt.show()
        '''
        
    def __len__(self):
        return len(self.run_files)

    def __getitem__(self, index):
        data_moment = self.run_files[index]
        camera_data = torch.FloatTensor().to(get_device())



        
        for frame in range(self.n_frames): 
            
            img = torch.FloatTensor(data_moment.data_point()['imgs'][frame]).to(get_device())
            camera_data = torch.cat((camera_data, img), 2)
             
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)

        all_actions = torch.from_numpy(data_moment.data_point()['actions'][:]).float().to(get_device()) 
        
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
    
