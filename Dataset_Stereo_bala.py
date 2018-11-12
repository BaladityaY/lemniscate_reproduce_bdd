import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from random import shuffle
import os
from util_moduls.Utils import get_device
from itertools import permutations

from util_moduls.steering_classes import get_class_no
import cPickle as pickle


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
    
    def __init__(self, left_images, right_images, steer_generator, motor_generator, left_img_ref, right_img_ref, steer_ref, motor_ref):
        self.left_generator = left_images
        self.right_generator = right_images
        self.steer_generator = steer_generator
        self.motor_generator = motor_generator
        
        self.left_img_ref = left_img_ref
        self.right_img_ref = right_img_ref
        self.steer_ref = steer_ref
        self.motor_ref = motor_ref
        
    @property
    def data_point(self):     
        return {'left':self.left_generator[self.left_img_ref], 'right':self.right_generator[self.right_img_ref], 'steer':self.steer_generator[self.steer_ref], 'motor':self.motor_generator[self.motor_ref]}
    
    def __str__(self):
        return str(self.left_generator) + str(self.right_generator) + " Index:" + str(self.index)

class Dataset(data.Dataset):
        
    def __init__(self, data_folder_dir, n_frames, prediction_steps, gpu):
        
        self.run_files = []
        self.n_frames = n_frames
        self.prediction_steps = prediction_steps
        self.gpu = gpu
        self.saved_train_labels = None
        
        for path, subdirs, files in os.walk(data_folder_dir):
            for file_name in files:
                file_ = os.path.join(path, file_name)
                
                if file_.endswith('.h5'):                
                    filename = os.path.join(data_folder_dir, file_)
                    
                    database_file = h5py.File(filename, 'r')
                    episode_key = get_episode_key(database_file)
                    if episode_key is None:
                        continue
                    
                    if 'file invalid' in database_file.keys():
                        print "Invalid file skipped " + file_
                        database_file.close()
                        continue
                    
                    left_images = database_file[episode_key]['CameraStereoLeftRGB']
                    right_images = database_file[episode_key]['CameraStereoRightRGB']
                    steer = database_file[episode_key]['expert_control_steer']
                    motor = database_file[episode_key]['expert_control_throttle']
                    
                    left_img_refs = database_file['moment_refs/{}/CameraStereoLeftRGB'.format(n_frames)]
                    right_img_refs = database_file['moment_refs/{}/CameraStereoRightRGB'.format(n_frames)]
                    steer_refs = database_file['moment_refs/{}/expert_control_steer'.format(n_frames)]
                    motor_refs = database_file['moment_refs/{}/expert_control_throttle'.format(n_frames)]
                    
                    for left_img_ref, right_img_ref, steer_ref, motor_ref in zip(left_img_refs,right_img_refs,steer_refs,motor_refs):
                        self.run_files.append(Data_Moment(left_images, right_images, steer, motor, left_img_ref, right_img_ref, steer_ref, motor_ref))
                
    def __len__(self):
        return len(self.run_files)

    def __getitem__(self, index):
        
        data_moment = self.run_files[index]
        
        camera_data = torch.FloatTensor().to(get_device(self.gpu))

        # Loop over all n_frames and not over the prediction steps which are at
        # the end of a data moment
        for frame in range(self.n_frames-self.prediction_steps):
            stereo_pair = torch.FloatTensor().to(get_device(self.gpu))
            for camera in ('left', 'right'):
                camera_image = data_moment.data_point[camera][frame]
                stereo_pair = torch.cat((stereo_pair, torch.from_numpy(camera_image).float().to(get_device(self.gpu))), 0).to(get_device(self.gpu))
                
                #cv2.imshow("test",camera_image)
                #cv2.waitKey(1000)
        
            camera_data = torch.cat((camera_data, stereo_pair), 2) 
            
        camera_data = camera_data.float() / 255. - 0.5
        camera_data = torch.transpose(camera_data, 0, 2)
        camera_data = torch.transpose(camera_data, 1, 2)
        
        current_steering = torch.from_numpy(data_moment.data_point['steer'][-self.prediction_steps:]).float().to(get_device(self.gpu))
        # In the current version, motor commands are not used
        #current_motor = torch.from_numpy(data_moment.data_point['motor'][-self.prediction_steps:]).float()
        
        #current_actions = torch.cat((current_steering, current_motor),0)
        # action_class = get_class_no(current_steering)
        
        return camera_data, current_steering, index
    
    def get_image(self,index):
        print index
        print len(self.run_files)
        return self.run_files[index].data_point['left']

    def get_label(self,index):
        data_moment = self.run_files[index]        
        current_steering = torch.from_numpy(data_moment.data_point['steer'][-self.prediction_steps:]).float()
        action_class = get_class_no(current_steering)
        return action_class, data_moment.data_point['steer'][-self.prediction_steps:]
    
    @property
    def train_labels(self):
        
#         if self.saved_train_labels is None:
#                 
#             try:
#                 with open("saved_labels.pkl","rb") as labelfile:
#                     labels = pickle.load(labelfile)
#                 print "Loading labels from pkl file"
#             except Exception as ex:
#                 
#                 print "Generating training labels"
# #                 labels = []
# #                 
#                 labels = range(len(self.run_files))
#                 for index in range(len(self.run_files)):
#                     
#                     if index % 100 == 0:
#                         print index
#                     
#                     # data_moment = self.run_files[index]
#                     
#                     # Camera data is the value and is skipped. Only the label is extracted
#                     # current_steering = torch.from_numpy(data_moment.data_point['steer'][-self.prediction_steps:]).float()
#                     
#                     # action_class = get_class_no(current_steering)
#         
#                     labels.append(index)
#                     
#                 self.saved_train_labels = labels
#                 
#                 with open("saved_labels.pkl","wn") as dumpfile:
#                     pickle.dump(labels,dumpfile)
#                     print "Writing labels to pkl file"
#                 
#         else:
#             labels = self.saved_train_labels

        return np.array(range(len(self.run_files)))

if __name__ == '__main__':
    pass
    #print len([entry for entry in permutations(steering_angles,3)])
    
