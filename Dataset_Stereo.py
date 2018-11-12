import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from random import shuffle
import os

from itertools import permutations

import cPickle as pickle
#from src.utils.device_selector import get_device # From Sascha's carla code
from util_moduls.Utils import get_device 

def get_episode_key(h5File):
    # Find key which is the episode number or break if not existent because of a big
    episode_key = None
    for key in h5File.keys():
        if key.isdigit():
            return key
            
    if episode_key is None:
        print "No episode key. Skipping file"
        h5File.close()
        return None
###### Helper methods

class Data_Moment():
    '''
    Returns a stereo image pair and the respective steering command from an hdf5 file.
    Among data moments the generators to access the underlying hdf5 files will be shared
    though the index will always point to the beginning of a set of n_frames and None
    will be returned if no full set can be returned.
    '''
    
    def __init__(self, left_images, right_images, steer_generator, motor_generator, start_index, stop_index, filename, frame_rate=1):
        self.left_generator = left_images
        self.right_generator = right_images
        self.steer_generator = steer_generator
        self.motor_generator = motor_generator
        self.filename = filename
        
        self.start_index = start_index
        self.stop_index = stop_index

        self.frame_rate = frame_rate
        
    @property
    def data_point(self):     
        return {'left':self.left_generator[self.start_index:self.stop_index*self.frame_rate:self.frame_rate], 'right':self.right_generator[self.start_index:self.stop_index*self.frame_rate:self.frame_rate], 'steer':self.steer_generator[self.start_index:self.stop_index*self.frame_rate:self.frame_rate], 'motor':self.motor_generator[self.start_index:self.stop_index*self.frame_rate:self.frame_rate]}
    
    def __str__(self):
        return str(self.left_generator) + str(self.right_generator) 
    
    def __len__(self):
        # This step introduced quite a delay before training. Maybe there is a better way to check
        # if the current data moment is long enough        
        return len(self.left_generator[self.start_index:self.stop_index])




class Dataset(data.Dataset):
    
    def sort_folder_ft(self, s):
        '''
        Returns the last two entries, file name and last folder name, as key to sort
        '''
        return s.split('/')[-2]+'/'+s.split('/')[-1]
        
    def sort_filelist(self,data_folder_dir):
        
        file_list = []
        
        for path, subdirs, files in os.walk(data_folder_dir):
            for file_name in files:
                if file_name.endswith('h5'):
                    filename = os.path.join(path,file_name)
                    
                    file_list.append(filename)

        
        return sorted(file_list,key=self.sort_folder_ft)
        

    def tmp_get_file_list(self):
        '''
        Quick fix until the new models are trained
        '''
        
        returned_list = []
        
        path = '/data/carla_training_data/' #"ascha/for_carla_training/train/"
        with open('../examples/working_list.txt') as f:
            filenames = f.read().splitlines()
        
        for filename in filenames:
            returned_list.append(os.path.join(path,filename))
            
        return returned_list

        
    def __init__(self, data_folder_dir, n_frames, prediction_steps, gpu, uniform=True, print_stuff=False, need_limit=False, frame_rate=1):
        
        self.run_files = []
        self.n_frames = n_frames
        self.prediction_steps = prediction_steps
        self.gpu = gpu; print('gpu: {}'. format(gpu))
        self.saved_train_labels = None
        self.steer_bins = np.zeros(7)
        
        #print data_folder_dir
        
        self.moment_counter = 0
        
        self.samp_limit = 64
        self.frame_rate = frame_rate

        self.num_files = 0
        
        #for filename in self.tmp_get_file_list(): # TEMPORARY FIX WITH OLD MODEL
        for filename in self.sort_filelist(data_folder_dir):
            if need_limit and self.moment_counter >= self.samp_limit:
                break                
        
            if need_limit and self.moment_counter >= self.samp_limit:
                break
            
            database_file = h5py.File(filename, 'r')
            episode_key = get_episode_key(database_file)
            if episode_key is None:
                continue
            
            if 'file invalid' in database_file.keys():
                print "Invalid file skipped " + filename
                database_file.close()
                continue

            #print('filename: {}'.format(filename))
            self.num_files = self.num_files + 1
            
            left_images = database_file[episode_key]['CameraStereoLeftRGB']
            right_images = database_file[episode_key]['CameraStereoRightRGB']
            steer = database_file[episode_key]['expert_control_steer']
            motor = database_file[episode_key]['expert_control_throttle']
            
            
            for i in range(len(left_images)):
                if need_limit and self.moment_counter >= self.samp_limit:
                    break
                
                moment = Data_Moment(left_images, right_images, steer, motor, i, i + n_frames, filename, self.frame_rate)
                
                moment_steer = moment.data_point['steer'][(self.n_frames/2) - self.prediction_steps]
                moment_steer = (np.array(moment_steer) + 1.)*6/2.
                if moment_steer < 0:
                    moment_steer = 0
                if moment_steer > 6:
                    moment_steer = 6
                moment_steer = int(np.round(moment_steer))
                
                min_steer_num = np.min([self.steer_bins[0]+self.steer_bins[1],
                                        self.steer_bins[2],
                                        self.steer_bins[3], 
                                        self.steer_bins[4],
                                        self.steer_bins[5]+self.steer_bins[6]])
                
                needs_balance = False
                
                if moment_steer > 1 and moment_steer < 5:
                    bin_count = self.steer_bins[moment_steer]
                elif moment_steer == 0 or moment_steer == 1:
                    bin_count = self.steer_bins[0] + self.steer_bins[1]
                else:
                    bin_count = self.steer_bins[5] + self.steer_bins[6]
                
                if uniform and bin_count >= min_steer_num + 150:
                    '''
                    print('moment_steer: {}, steer count: {}, min steer conut: {}'.format(moment_steer, 
                                                                                          bin_count,
                                                                                          min_steer_num))
                    '''
                    needs_balance = True

                if len(moment) == n_frames:
                    # Check if any image is dark
                    image_sequence = left_images[i:i+n_frames]
                    
                    if (not needs_balance) and (not (0 in [np.sum(left_image) for left_image in image_sequence])):
                        self.run_files.append(moment)
                        self.steer_bins[moment_steer] = self.steer_bins[moment_steer] + 1
                        self.moment_counter = self.moment_counter + 1
                    else:
                        pass
                        # print "dark frame detected" # or just trying to uniformly balance dataset
                        
                else:
                    break
                
    
                        
        
        if print_stuff:
            resized_bins = np.zeros(5)
            resized_bins[0] = self.steer_bins[0] + self.steer_bins[1]
            resized_bins[1] = self.steer_bins[2]
            resized_bins[2] = self.steer_bins[3]
            resized_bins[3] = self.steer_bins[4] 
            resized_bins[4] = self.steer_bins[5] + self.steer_bins[6]
            
            f, axes = plt.subplots(1,2)
            axes[0].plot(range(5), resized_bins) #hist(self.steer_bins, bins=7)
            axes[1].plot(range(7), self.steer_bins)
            plt.show()

        print('num files: {}'.format(self.num_files))
                

                
    def __len__(self):
        return len(self.run_files)

    def __getitem__(self, index):
        print('getting index: {}'.format(index))
        data_moment = self.run_files[index]
        
        camera_data = torch.FloatTensor().to(get_device(self.gpu))

        # Loop over all frames in between n_frames/2 - prediction_steps and n_frames/2.
        #
        for frame in range(self.n_frames): #(self.n_frames / 2) - self.prediction_steps, self.n_frames / 2):
            
            stereo_pair = torch.FloatTensor().to(get_device(self.gpu))
            for camera in ('left', 'right'):
                camera_image = data_moment.data_point[camera][frame]
                
                stereo_pair = torch.cat((stereo_pair, torch.from_numpy(camera_image).float().to(get_device(self.gpu))), 2).to(get_device(self.gpu))
        
            camera_data = torch.cat((camera_data, stereo_pair), 2)
             
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)

        # Get the past 3 steering commands before and including n_frames/2
        current_steering = torch.from_numpy(data_moment.data_point['steer'][:]).float().to(get_device(self.gpu)) #Changed to ':' from '0 : self.n_frames/2'

        return camera_data, current_steering, index
    
    def get_image(self, index):
        print index
        print len(self.run_files)
        return self.run_files[index].data_point['left']

    '''
    def get_label(self, index):
        data_moment = self.run_files[index]        
        current_steering = torch.from_numpy(data_moment.data_point['steer'][-self.prediction_steps:]).float()
        action_class = get_class_no(current_steering)
        return action_class, data_moment.data_point['steer'][-self.prediction_steps:]
    '''
    
    @property
    def train_labels(self):
        
        return np.array(range(len(self.run_files)))

if __name__ == '__main__':
    test = Dataset("/home/sascha/for_carla_training/train",6,1,'cpu')
    
