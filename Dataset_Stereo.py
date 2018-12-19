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
    
    def __init__(self, images, speeds, start_index, n_frames, frame_gap, filename):
        
        self.images = images
        self.speeds = speeds
        
        self.filename = filename
        
        self.start_index = start_index
        self.stop_index = self.start_index + (n_frames*frame_gap)
        self.frame_gap = frame_gap
        
    def convert_images(self, encoded_images):
        return [cv2.imdecode(np.fromstring(encoded_img, dtype=np.uint8), -1) for encoded_img in encoded_images]      
    
    def data_point(self):     
        
        indices = np.arange(self.start_index,self.stop_index,self.frame_gap)
        print indices
        # Speeds are one list, twice the size of images, because they are flattened out pairs of values.
        # They have to be reshaped to be pairs. It would be possible to select first the range on where
        # to do the reshape operation and then do it though it is assumed the operation takes about the
        # same time so we first reshape because then indexing becomes easier, as it is equal to image indexing.
        speeds = np.reshape(self.speeds, [-1, 2])
        
        speeds = speeds[indices]
        
        velocities = np.array(np.linalg.norm(speeds,axis=1),dtype=np.float32)
        course_list = np.array(BDD_Helper.to_course_list(speeds),dtype=np.float32)
        #print course_list
        #print np.diff(course_list)
        #exit()
        velocities_courses = np.array(zip(velocities,course_list))
        
        # Images have to be re-formatted into a numpy array because the special indexing does
        # not work on hdf5 files
        images = self.images[:][indices]
        
        return {'imgs':self.convert_images(images),  
                'vel_course_pairs':velocities_courses}
    

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
    
    def __init__(self, data_folder_dir, n_frames, frame_gap):
        
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
                
                start_index = i
                
                moment = Data_Moment(images, speeds, start_index, n_frames, frame_gap, filename)
                
                if moment is None:
                    # At the end of a sequence no full scene can be compiled
                    break                
                
                self.run_files.append(moment)
                

        
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

        vel_course_pairs = torch.from_numpy(data_moment.data_point()['vel_course_pairs']).float().to(get_device()) 
        
        for i, value in enumerate(vel_course_pairs[:,1]):
            if np.isnan(value):
                vel_course_pairs[i][1] = np.random.uniform(0,2*np.pi)
        #vel_course_pairs[:,1][np.isnan(vel_course_pairs[:,1])] = np.random.uniform(0,2*np.pi) 
        
        return camera_data, vel_course_pairs, index
    
    @property
    def train_labels(self):
        return np.array(range(len(self.run_files)))

if __name__ == '__main__':
    import cv2
    train_dataset = Dataset("/home/sascha/for_bdd_training/tiny_test_set",6,frame_gap=1)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, (images, vel_course, index) in enumerate(train_loader):
        
        for frame in np.split(images[0],6):
            print frame.shape
            img = frame.data.cpu().numpy()
            img = img.transpose((1,2,0))+0.5
            cv2.imshow("Test", img)
            cv2.waitKey(90)
    
