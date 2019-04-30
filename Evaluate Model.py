import torch
import torch.nn as nn
from collections import defaultdict
import pandas as pd
import h5py
from sklearn.metrics import log_loss
from collections import OrderedDict
import cv2
import models
import os
from Dataset import Dataset
from torch.autograd.variable import Variable
import numpy as np
import sys
from util_moduls.Utils import get_device
import argparse
from lib.NCEAverage import NCEAverage
from lib.NCECriterion import NCECriterion
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import time
from test import NN, kNN
from scipy import misc
from PIL import Image


parser = argparse.ArgumentParser(description='Evaluate a trained network')
parser.add_argument('--free_mem_percent', dest='mem_free',default=10, type=int,
                    help='Memory to be kept free when loading data into memory.')
parser.add_argument('--preload_data', dest='preload', action='store_true',
                    help='Store memory in data before training')
parser.add_argument('--lower-index', dest='index_low', type=int,
                    help='Lower index of entries in the data loader to be parsed')
parser.add_argument('--upper-index', dest='index_up', type=int,
                    help='Upper index of entries in the data loader to be parsed')
parser.add_argument('--only-show-size', dest='show_val_size', action='store_true',
                    help='Show only the size of the val loader')


args = parser.parse_args()



def get_free_mem():
    meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1])) for i in open('/proc/meminfo').readlines())
    mem_available = float(meminfo['MemAvailable'])
    mem_total = float(meminfo['MemTotal'])
    return (mem_available / mem_total) * 100.

def resize2d(img, size):
    return (torch.nn.functional.adaptive_avg_pool2d(Variable(img, requires_grad=False), size)).data



low_dim = 128
checkpoint = torch.load('model_best.pth.tar')
model = models.__dict__[checkpoint['arch']](n_frames=6, low_dim=low_dim)
model = model.cuda()
model = torch.nn.DataParallel(model)

state_dict = checkpoint['state_dict']

print('Loading epoch: {}'.format(checkpoint['epoch']))

lemniscate = checkpoint['lemniscate']

model.load_state_dict(state_dict)           

training_file = "/data/dataset/full_train_set_v3.hdf5"
validation_file = "/data/dataset/full_val_set_v3.hdf5"

# Keep 25% of the memory free for writing data into the dictionary.

keep_memory_free = args.mem_free
preload_to_mem = args.preload


# In this order the whole val dataset should be loaded into memory and a part of the training set
val_dataset = Dataset(validation_file, preload_to_mem=preload_to_mem, keep_memory_free=keep_memory_free, sliding_window=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=0)

if args.show_val_size:
    print "The val loader has {} entries".format(len(val_loader))
    exit()

train_dataset = Dataset(training_file, preload_to_mem=preload_to_mem, keep_memory_free=keep_memory_free)

# In this hacky code the batch size has to be always 1 !!!!!!!!!!!!!!!!!!!
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=False,
    num_workers=0)


ndata = train_dataset.__len__()
nce_k = 4096
nce_t = .07
nce_m = .5
iter_size = 1

n_frames = 6
gpu = 0

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 50)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2


def write_text(img, text):

    img_copy = img.copy()
    cv2.putText(img_copy, str(text),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    return img_copy


model.eval()
debug = True

topk = 10  # It became evident that the top 5 NNs are sufficient for the best results


correct = 0.
correct_past = 0.
correct_future = 0.

total = 0
observed_timestep = 3

trainFeatures = lemniscate.memory.t()

trainLabels = torch.LongTensor(train_loader.dataset.train_labels).cuda()

#stat_data = []

start_time = time.time()

def draw_blue_rectangle(img, x_start,y_start,side_length,gt_straight,gt_stop,gt_left,gt_right):
    # Left
    b_side_length = side_length#15
    b_start_left = x_start#20
    b_start_top = y_start#20
    
    pt1 = (b_start_left,b_start_top)
    pt2 = (b_start_left+b_side_length,b_start_top+b_side_length)
    
    cv2.rectangle(img,pt1,pt2,(np.min((255,180+int(255*gt_left))),
                               int(255*gt_left),int(255*gt_left)),-1)
    
    # Center
    pt3 = pt2
    pt4 = (b_start_left+b_side_length*2,(b_start_top))
    
    cv2.rectangle(img,pt3,pt4,(np.min((255,180+int(255*gt_stop))),int(255*gt_stop),int(255*gt_stop)),-1)
    
    # Right
    pt5 = pt4
    pt6 = (b_start_left+b_side_length*3,(b_start_top+b_side_length))
    
    cv2.rectangle(img,pt5,pt6,(np.min((255,180+int(255*gt_right))),int(255*gt_right),int(255*gt_right)),-1)
    
    # Top
    
    pt7 = (b_start_left+b_side_length,b_start_top-b_side_length)
    pt8 = (b_start_left+b_side_length*2,b_start_top)
        
    cv2.rectangle(img,pt7,pt8,(np.min((255,180+int(255*gt_straight))),int(255*gt_straight),int(255*gt_straight)),-1)
    
def draw_green_rectangle(img, x_start,y_start,side_length,gt_straight,gt_stop,gt_left,gt_right):
    # Left
    b_side_length = side_length#15
    b_start_left = x_start#20
    b_start_top = y_start#20
    
    pt1 = (b_start_left,b_start_top)
    pt2 = (b_start_left+b_side_length,b_start_top+b_side_length)
    
    cv2.rectangle(img,pt1,pt2,(int(255*gt_left),np.min((255,180+int(255*gt_left))),
                               int(255*gt_left)),-1)
    
    # Center
    pt3 = pt2
    pt4 = (b_start_left+b_side_length*2,(b_start_top))
    
    cv2.rectangle(img,pt3,pt4,(int(255*gt_stop),np.min((255,180+int(255*gt_stop))),int(255*gt_stop)),-1)
    
    # Right
    pt5 = pt4
    pt6 = (b_start_left+b_side_length*3,(b_start_top+b_side_length))
    
    cv2.rectangle(img,pt5,pt6,(int(255*gt_right),np.min((255,180+int(255*gt_right))),int(255*gt_right)),-1)
    
    # Top
    
    pt7 = (b_start_left+b_side_length,b_start_top-b_side_length)
    pt8 = (b_start_left+b_side_length*2,b_start_top)
        
    cv2.rectangle(img,pt7,pt8,(int(255*gt_straight),np.min((255,180+int(255*gt_straight))),int(255*gt_straight)),-1)

def add_indicator(img, gt_actions, nn_actions):
    
    gt_t_3 = gt_actions
    
    gt_straight = gt_t_3[0]
    gt_stop = gt_t_3[1]
    gt_left = gt_t_3[2]
    gt_right = gt_t_3[3]
    
    if nn_actions is not None:
        
        nn_t_3 = nn_actions
    
        nn_straight = nn_t_3[0]
        nn_stop = nn_t_3[1]
        nn_left = nn_t_3[2]
        nn_right = nn_t_3[3]
        
        draw_blue_rectangle(img,52,30,15,gt_straight,gt_stop,gt_left, gt_right)
        draw_green_rectangle(img,127,30,15,nn_straight,nn_stop,nn_left, nn_right)
    else:
        draw_blue_rectangle(img,52,30,15,gt_straight,gt_stop,gt_left, gt_right)

    return img

with h5py.File('video_images.h5py', 'a') as out_file:
    with torch.no_grad():
    
        #criterion = nn.BCELoss(reduce=False).cuda()
        #bce = lambda t1, t2: criterion(t1, t2).mean().flatten()
        #pixel_loss = nn.MSELoss()
        
        file_keys = out_file.keys()
        if args.index_low:
            val_enumerator = enumerate(val_loader,args.index_low)
        else:
            val_enumerator = enumerate(val_loader)
        
        for batch_idx, (input_imgs, targets, indexes) in val_enumerator:
            batch_time = time.time()
            
            if args.index_up:
                if batch_idx >= args.index_up:
                    print "Reached upper limit {}".format(batch_idx)
                    exit()
                                    
            if batch_idx % 50 == 0:
                print "Batch {} from {}, upper lim is {}".format(batch_idx, len(val_loader),args.index_up)
                if get_free_mem() < 1:
                    print "ERROR: OUT OF MEMORY"
                    exit()
                
            if 'val_set_{}'.format(batch_idx) in file_keys:
                # Skip already added entries
                print "Skipping {}".format(batch_idx)
                continue
            
            targets = targets.cuda(async=True)
            indexes = indexes.cuda(async=True)
            input_imgs = input_imgs.cuda(async=True)
            
            batchSize = input_imgs.size(0)
            
            og_targets = targets.clone()
    
            input_imgs = input_imgs[:, 0:int((n_frames / 2) * 3), :, :]  # extract only img 1 through 3
            
            targets_orig = targets.clone().cpu().numpy()
            targets = targets[:, 0:int(n_frames / 2)]  # extract steers first 3 targets
            
            # targets = torch.from_numpy(np.array([[[0., 0., 0., 1., 0., 0.],
            # [0., 0., 0., 1., 0., 0.],
            # [0., 0., 0., 1., 0., 0.]]])).float().cuda()
    
            features = model(input_imgs, targets)
    
            dist = torch.mm(features, trainFeatures)
    
            yd, yi = dist.topk(topk, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
    
            retrieval = retrieval.narrow(1, 0, topk).clone()
            yd = yd.narrow(1, 0, topk)
            
            min_val = yd.min()
            max_val = yd.max()
            a = 1 / (max_val - min_val)
            b = 1 - a * max_val
            yd_mapped = yd.clone().mul(a).add(b)
            
            image_steering_labels = defaultdict()
            
            indexes = np.array(indexes.cpu())
            
            neighbour_stat_list = []
            img_retrievals = []
            
            show_images = True
            
            # data_keys = ['speeds','latitude','longitude','gyro_x','gyro_y','gyro_z','acc_x','acc_y','acc_z','file_key']
            data_keys = val_loader.dataset.__get_data_point__(indexes[0]).data_point().keys()
            
            try:
                current_reference_value = {key:val_loader.dataset.__get_data_point__(indexes[0]).data_point()[key][:] for key in data_keys if key is not'imgs'}
            except:
                # gyro is not always there
                current_reference_value = {key:val_loader.dataset.__get_data_point__(indexes[0]).data_point()[key][:] for key in data_keys if key is not 'imgs' and not 'gyro' in key}
            
            for top_id in range(topk):
    
                ret_ind = int(retrieval[0, top_id])           
                
                try:
                    retrieval_value = {key:train_loader.dataset.__get_data_point__(ret_ind).data_point()[key][:] for key in data_keys if key is not'imgs'}
                except:
                    retrieval_value = {key:train_loader.dataset.__get_data_point__(ret_ind).data_point()[key][:] for key in data_keys if key is not'imgs'  and not 'gyro' in key}
                
                retrieval_value.update({'action_label':train_loader.dataset.__getlabel__(ret_ind)[0]})
                retrieval_value.update({'action_target':targets_orig[0]})
                
                # action_label_past = train_loader.dataset.__getlabel__(ret_ind)[0][0:3]
                # targets_past = targets_orig[0][0:3]
                # action_correlations.append(bce(action_label_past,targets_past).cpu().numpy())
                
                # reference_retrieval_difference = np.sum(np.linalg.norm((retrieval_value,current_reference_value), axis = 0))
                neighbour_stat_list.append(retrieval_value)
                
                if show_images: img_retrievals.append(train_loader.dataset.__getitem__(ret_ind)[0])
            
            # text_on_img = train_loader.dataset.__getlabel__(retrieval[0, 0])[0]
            if show_images: ret_imgs_cv2_comp = [(img_retrievals[i][6:9] + 0.5).transpose(1, 2).transpose(0, 2) for i in range(topk)]
            
            # avg_img = torch.stack(ret_imgs_cv2_comp).mean(0)        
            if show_images: query_img = (input_imgs[0, 0:3, :, :] + 0.5).transpose(1, 2).transpose(0, 2)        
            
            #if show_images: show_img = torch.cat(ret_imgs_cv2_comp, 1)  # .cpu().numpy()
            #if show_images: show_img = torch.cat((query_img, show_img), 1).cpu().numpy()
            if show_images: nb_imgs = []
            
            if show_images:
            
                return_img = np.zeros((224,224,3),np.float)
                neighbor_image_list = [(img_retrievals[i][6:9] + 0.5).transpose(1, 2).transpose(0, 2) for i in range(topk)]
                for i, nb_img in enumerate(neighbor_image_list):
                    nn_ret = neighbour_stat_list[i]['action_label'][2] # choosing the 3rd frame
                    gt_ret = neighbour_stat_list[i]['action_target'][2]
                    nb_img = ret_imgs_cv2_comp[i].cpu().numpy()
                    nb_img = np.array(nb_img * 255.,dtype=np.uint8) 
                    #add_indicator(nb_img,gt_ret,nn_ret)
                    nb_imgs.append(nb_img)
                    
                    return_img = return_img+nb_img/len(neighbor_image_list)

            
            final_action_vec = []
            
            for neighbor in neighbour_stat_list[0:topk]:
                
                averaged_nbs = np.average(np.array(neighbor['action_label'][2:]),axis=0)
                final_action_vec.append(averaged_nbs)
            
            
            #print "before {}".format(np.array(final_action_vec),axis=0)
            #print "after {}".format(np.average(np.array(final_action_vec),axis=0))
            
            final_action_vec = np.average(np.array(final_action_vec),axis=0)
            
            if show_images:
                gt_ret = neighbour_stat_list[0]['action_target'][2]
                
                query_img = query_img.cpu().numpy()
                
                query_img = np.array(query_img * 255.,dtype=np.uint8)
                
                query_img = add_indicator(query_img,gt_ret,final_action_vec)
                # The following can be used to write directly on images
                # show_img = write_text(show_img,mse_pixel_loss.cpu().numpy())
                
                #img_to_average = np.concatenate(nb_imgs,axis=0)
                #other_images = np.mean(img_to_average,axis=0)
                #other_images = np.expand_dims(other_images,2)
                
                #print query_img.shape
                #print other_images.shape
                return_img=np.array(np.round(return_img),dtype=np.uint8)
                show_img = np.hstack((query_img,return_img))
                
                cv2.imshow('test', show_img)
                cv2.waitKey(1)
                cv2.imwrite("avg-{}.jpg".format(batch_idx),show_img)
            
            
            
            
            
            # The next lines can be used to quickly debug code
            if batch_idx % 10 == 0:
                print "Batch time {}".format(time.time() - batch_time)
                                
            val_set_id = batch_idx            
            
            for key in current_reference_value.keys():
                try:
                    out_file.create_dataset('val_set_{}/reference/{}'.format(val_set_id, key), data=current_reference_value[key])
                except Exception as ex:
                    print ex
                    print 'We tried the key val_set_{}/reference/{}'.format(val_set_id, key)
            
            for neighbor_id, neighbors_dict in enumerate(neighbour_stat_list):
                try:
                    for key in neighbors_dict.keys():
                        out_file.create_dataset('val_set_{}/neighbor_{}/{}'.format(val_set_id, neighbor_id, key), data=neighbors_dict[key])
                except Exception as e:
                    print neighbors_dict
                    print e
                
    print "End time is {}".format(time.time() - start_time)
        
    if show_images: cv2.destroyAllWindows()

# In[13]:

# 
#     for val_set_id, list_entry in enumerate(stat_data):
#         
#         reference_value_dict = reference_values[val_set_id]
#         for key in reference_value_dict.keys():
#             out_file.create_dataset('val_set_{}/reference/{}'.format(val_set_id, key), data=reference_value_dict[key])
#         
#         for neighbor_id, neighbors_dict in enumerate(list_entry):
#             try:
#                 for key in neighbors_dict.keys():
#                     out_file.create_dataset('val_set_{}/neighbor_{}/{}'.format(val_set_id, neighbor_id, key), data=neighbors_dict[key])
#             except Exception as e:
#                 print neighbors_dict
#                 print e
            
        # break
print "Finished creating stat data file"

# In[ ]:

# test_file = h5py.File('stat_data.h5py','r')

# print test_file['val_set_0']['reference']['acc_x'][:]-test_file['val_set_0']['neighbor_0']['acc_x'][:]
# 

# print test_file['val_set_0']['reference']['file_key'][...]
# 

# test_file.close()

# In[ ]:

# In[ ]:

# In[ ]:

# In[35]:

# action_correlations = [np.average(neighbours) for neighbours in stat_data['action_correlations']]

# In[37]:

#plt.plot(action_correlations)
#plt.show()

# stat_data['action_correlations']

# # Following is some code to check if the information within the hdf5 files is correct
# 

# data_root = "/data/outside_data/BDDUnzipped/structure_for_code/tfrecords/training"

# filename = '9530facd-cb0abe12.h5'

# testfile = h5py.File(os.path.join(data_root,filename),'r')

# def convert_images(encoded_images):
#     return [cv2.imdecode(np.fromstring(encoded_img, dtype=np.uint8), -1) for encoded_img in encoded_images]

# decoded_images = convert_images(testfile['image']['encoded'])

# 

# for i, img in enumerate(decoded_images):
#     position = (testfile['latitude'][i],testfile['longitude'][i])
#     #print position
#     img = write_text(img,text=position)
#     cv2.imshow('test',img)
#     cv2.waitKey(30)
#     
#     if i > 200:
#         break
#         
# cv2.destroyAllWindows()

# # So now I know the freshly added lat and long to the hdf5 file works
# 

# ### The next lines of code are old code to play around with the actions

# model.eval()
# debug = True
# 
# topk = 5 # It became evident that the top 5 NNs are sufficient for the best results
# steer_eval = {}
# 
# correct = 0.
# correct_past = 0.
# correct_future = 0.
# 
# total = 0
# observed_timestep = 3
# 
# trainFeatures = lemniscate.memory.t()
# 
# trainLabels = torch.LongTensor(train_loader.dataset.train_labels).cuda()
# 
# stat_data = {}
# stat_data['action_correlations'] = []
# 
# start_time = time.time()
# img_id = 13
# ti
# 
# input_imgs = torch.unsqueeze(train_loader.dataset.__getitem__(img_id)[0],0)
# targets = torch.unsqueeze(train_loader.dataset.__getitem__(img_id)[1],0)
# 
# for i in range(0,6)*4:
#     with torch.no_grad():
# 
#         criterion = nn.BCELoss(reduce=False).cuda()
#         bce = lambda t1, t2: criterion(t1, t2).mean().flatten()
#         pixel_loss = nn.MSELoss()
# 
#         targets = targets.cuda(async=True)
#         input_imgs = input_imgs.cuda(async=True)
# 
#         og_targets = targets.clone()
# 
#         input_imgs = input_imgs[:,0:int((n_frames/2)*3),:,:] #extract only img 1 through 3
#         targets_orig = targets.clone()
# 
#         targets = targets[:,0:int(n_frames/2)] #extract steers first 3 targets
# 
#         # The next lines are used to investigate how the images change
#         # with different actions as input
#         action_prob = np.array([int(j == 3) for j in range(6)])
# 
#         #targets = torch.from_numpy(np.array([[action_prob,
#         # action_prob,
#         # action_prob]])).float().cuda()
# 
#         features = model(input_imgs, targets)
# 
#         dist = torch.mm(features, trainFeatures)
# 
#         yd, yi = dist.topk(topk, dim=1, largest=True, sorted=True)
#         candidates = trainLabels.view(1,-1).expand(batchSize, -1)
#         retrieval = torch.gather(candidates, 1, yi)
# 
#         retrieval = retrieval.narrow(1, 0, topk).clone()
#         yd = yd.narrow(1, 0, topk)
# 
#         min_val = yd.min()
#         max_val = yd.max()
#         a = 1 / (max_val - min_val)
#         b = 1 - a * max_val
#         yd_mapped = yd.clone().mul(a).add(b)
# 
#         image_steering_labels = defaultdict()
# 
#         action_correlations = []
#         img_retrievals = []
# 
#         show_images = True
# 
#         for top_id in range(topk):
# 
#             ret_ind = int(retrieval[0, top_id])
#             action_label_past = train_loader.dataset.__getlabel__(ret_ind)[0][0:3]
# 
#             #train_test_retrieval = val_loader.dataset.run_files[top_id].data_point()
#             targets_past = targets_orig[0][0:3]
# 
#             action_correlations.append(bce(action_label_past,targets_past).cpu().numpy())
# 
#             if show_images: img_retrievals.append(train_loader.dataset.__getitem__(ret_ind)[0])
# 
# 
#         #text_on_img = train_loader.dataset.__getlabel__(retrieval[0, 0])[0]
#         if show_images: ret_imgs_cv2_comp = [(img_retrievals[i][6:9]+0.5).transpose(1,2).transpose(0,2) for i in range(topk)]
# 
#         #avg_img = torch.stack(ret_imgs_cv2_comp).mean(0)        
#         if show_images: query_img = (input_imgs[0,0:3,:,:]+0.5).transpose(1,2).transpose(0,2)
# 
# 
#         if show_images: show_img = torch.cat(ret_imgs_cv2_comp,1)#.cpu().numpy()
#         if show_images: show_img = torch.cat((query_img,show_img),1).cpu().numpy()
# 
#         #mse_pixel_loss = pixel_loss(Variable(avg_img), Variable(query_img))
# 
#         stat_data['action_correlations'].append(action_correlations)
# 
#         # The following can be used to write directly on images
#         #show_img = write_text(show_img,mse_pixel_loss.cpu().numpy())
#         if show_images: cv2.imshow('test',show_img)
#         if show_images: cv2.waitKey(3)
# 
#         input_imgs = torch.unsqueeze(train_loader.dataset.__getitem__(ret_ind)[0],0)
# 
# 
# if show_images: cv2.destroyAllWindows()
# 
