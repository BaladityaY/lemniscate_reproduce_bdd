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
from hdf5_wrapper import DB_manager


def get_device(device_id = 0):
    device = torch.device("cuda")
    return device
    

def get_free_mem():
    meminfo = dict((i.split()[0].rstrip(':'),int(i.split()[1])) for i in open('/proc/meminfo').readlines())
    mem_available = float(meminfo['MemAvailable'])
    mem_total = float(meminfo['MemTotal'])
    return (mem_available/mem_total)*100.

memory_data_buffer = {}

def load_to_mem(hdf_reference):
    '''
    Loads the content of the hdf_reference dataset into memory, using the dataset reference as a key.
    This is only done if that reference does not yet exist in the buffer
    '''
    
    if not hdf_reference in memory_data_buffer: 
        memory_data_buffer.update({hdf_reference:hdf_reference[:]})
    # Return a pointer to the data in the dict
    return memory_data_buffer.get(hdf_reference)

class Data_Moment():
    
    def __init__(self, sequence, start_index, n_frames, frame_gap, preload_to_mem = False):
        '''
        There can be no calculation with content of the hdf5 file or a selection of ranges here
        because that will slow down loading and put a lot of data in memory
        '''        
        
        self.images = load_to_mem(sequence.images) if preload_to_mem else sequence.images
        self.speeds = load_to_mem(sequence.speeds) if preload_to_mem else sequence.speeds
                
        # Because the change of course is calculated, we need n+1 datapoint to calculate
        # n course changes. This is done by increasing the length of a data moment and
        # then throwing the first frame away
        n_frames += 1
        
        self.start_index = start_index
        self.stop_index = self.start_index + (n_frames*frame_gap) 
        print self.stop_index
        print self.start_index
        exit()
        self.frame_gap = frame_gap
        if self.stop_index >= len(self.images):
            self.invalid = True
        else:
            self.invalid = False
        
        
    def convert_images(self, encoded_images):
        return [cv2.imdecode(np.fromstring(encoded_img, dtype=np.uint8), -1) for encoded_img in encoded_images]
    
    def data_label(self):
        '''
        
        Experimental method if the label can be loaded faster than a image,label pair
        '''
        speed_indices = np.arange(self.start_index,self.stop_index,self.frame_gap)

        # Speeds are one list, twice the size of images, because they are flattened out pairs of values.
        # They have to be reshaped to be pairs. It would be possible to select first the range on where
        # to do the reshape operation and then do it though it is assumed the operation takes about the
        # same time so we first reshape because then indexing becomes easier, as it is equal to image indexing.
        speeds = np.reshape(self.speeds, [-1, 2])
        speeds = speeds[speed_indices]
        actions = BDD_Helper.turn_future_smooth(speeds, 5, 0)
        
        return {'actions':actions}
    
    
    def data_point(self):
        
        img_indices = np.arange(self.start_index,self.stop_index,self.frame_gap)
        img_indices = np.delete(img_indices,0)
        
        speed_indices = np.arange(self.start_index,self.stop_index,self.frame_gap)

        # Speeds are one list, twice the size of images, because they are flattened out pairs of values.
        # They have to be reshaped to be pairs. It would be possible to select first the range on where
        # to do the reshape operation and then do it though it is assumed the operation takes about the
        # same time so we first reshape because then indexing becomes easier, as it is equal to image indexing.
        speeds = np.reshape(self.speeds, [-1, 2])
        speeds = speeds[speed_indices]
        actions = BDD_Helper.turn_future_smooth(speeds, 5, 0)
        
        images = self.images[:][img_indices]
        
        return {'imgs':self.convert_images(images),  
                'actions':actions}
    

class Dataset(data.Dataset):
    
    
    def __init__(self, data_file_path, n_frames=6, frame_gap=4, preload_to_mem = True, keep_memory_free=15, sliding_window=False):
        
        self.run_files = []
        self.n_frames = n_frames

        db = DB_manager(data_file_path)
        
        debug = False
        seq_limit = 50
        
        # We need to ensure one fixed not randomized order of images because the approach has to index
        # the images always in the same way and os.walk does not ensure one fixed order
        for seq_no, sequence in enumerate(db.get_sequence_list()):
                        
            if debug and seq_no >= seq_limit:
                break
            
            image_length = len(sequence.images)

            # If we are not going over the data in a sliding window fashion we jump depending on the amount 
            # of frames and the gap size
            step = 1 if sliding_window else n_frames*frame_gap
            
            for i in range(0,image_length,step):
                
                start_index = i
                if preload_to_mem and get_free_mem() > keep_memory_free:
                    moment = Data_Moment(sequence, start_index, n_frames, frame_gap, preload_to_mem)
                else:
                    if preload_to_mem:
                        print("Loading to mem stopped")
                    moment = Data_Moment(sequence, start_index, n_frames, frame_gap, False)
                
                if moment.invalid:
                    # At the end of a sequence no full scene can be compiled
                    # print("Moment too short")
                    break                
                
                self.run_files.append(moment)
            
             
        
    def __len__(self):
        return len(self.run_files)
    
    def isnan(self,x):
        '''
        numpy-free way for torch compatible is nan check
        '''
        return x != x

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
    
    def __getlabel__(self,index):
        data_moment = self.run_files[index]        
        all_actions = torch.from_numpy(data_moment.data_label()['actions'][:]).float().to(get_device())
        
        return all_actions, index
    
    @property
    def train_labels(self):
        return np.array(range(len(self.run_files)))

if __name__ == '__main__':
    pass
# 
#     #train_dataset = Dataset("/home/sascha/for_bdd_training/full_dataset/train",n_frames=6,frame_gap=4,preload_to_mem=False)
#     #train_dataset = Dataset("/home/sascha/for_bdd_training/smaller_dataset/train",n_frames=6,frame_gap=4,preload_to_mem=False)
#     train_dataset = Dataset("/data/dataset/full_train_set.hdf5",n_frames=6,frame_gap=4,preload_to_mem=False)    
#     print "Dataset has {} entries".format(len(train_dataset))
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
#     
#     for i, (images, vel_course, index) in enumerate(train_loader):
#         #print len(train_loader.dataset)
#         img = images[0][6:9].data.cpu().numpy()
#         img = img.transpose((1,2,0))+0.5
#         print(vel_course)
#         cv2.imshow("Test", img)
#         cv2.waitKey(30)


