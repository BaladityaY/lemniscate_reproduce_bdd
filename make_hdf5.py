import numpy as np
import h5py

import torch
import models
import os
from Dataset_Stereo import Dataset
#from torch.autograd.variable import Variable
import sys
from util_moduls.Utils import get_device
#from lib.NCEAverage import NCEAverage
from lib.NCECriterion import NCECriterion
#import matplotlib.pyplot as plt
#from test import NN, kNN
#from scipy import misc
from collections import OrderedDict
import pickle


print "Usage: make_hdf5 checkpoint_file training_data_folder validation_data_folder filename_hdf5_file"

low_dim = 128
checkpoint = torch.load(sys.argv[1]) 
model = models.__dict__[checkpoint['arch']](low_dim=low_dim)
model = model.cuda()
state_dict = checkpoint['state_dict']

print('epoch: {}'.format(checkpoint['epoch']))

lemniscate = checkpoint['lemniscate']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict)

traindir = sys.argv[2]
n_frames = 6
gpu = 0
j = 0
seed = 232323
batch_size = 1
train_dataset = Dataset(traindir, n_frames)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=j)
torch.manual_seed(seed)

valdir = sys.argv[3]
val_dataset = Dataset(valdir, n_frames)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=j)

ndata = train_dataset.__len__()
nce_k = 4096
nce_t = .07
nce_m = .5
iter_size = 1

#lemniscate = NCEAverage(gpu, low_dim, ndata, nce_k, nce_t, nce_m).to(get_device(gpu))
criterion = NCECriterion(ndata).to(get_device(gpu))

n_neighbours = 50

hf = h5py.File(sys.argv[4]+'.h5', 'w')

print('create h5py data')
    
all_img_names = []
#all_imgs = []
all_id_nums = []
all_steer_truths = []
all_steer_preds = []
all_losses = []
all_steer_diffs = []
all_yds = []

trainFeatures = lemniscate.memory.t()
trainLabels = torch.LongTensor(train_loader.dataset.train_labels).cuda()

for i, (input_imgs, input_steerings, indices) in enumerate(val_loader):

    # DEBUG TEST WRITING
#     if i > 5:
#         break
    og_input_steerings = input_steerings.clone().cpu().numpy()

    input_imgs = input_imgs[:,0:9,:,:] #extract only img 3 out of 6
    input_steerings = input_steerings[:,0:3] #extract steers first 3 out of 6

    indices = indices.to(get_device(gpu))

    input_steerings = input_steerings.cuda(async=True)
    indices = indices.cuda(async=True)
    batchSize = input_imgs.size(0)

    features = model(input_imgs, input_steerings)
    output = lemniscate(features, indices)
    loss = criterion(output, indices) / iter_size

    dist = torch.mm(features, trainFeatures)

    yd, yi = dist.topk(n_neighbours, dim=1, largest=True, sorted=True)
    candidates = trainLabels.view(1,-1).expand(batchSize, -1)
    retrieval = torch.gather(candidates, 1, yi)

    retrieval = retrieval.narrow(1, 0, n_neighbours).clone().cpu().numpy()#.view(-1)
    
    yd = yd.narrow(1, 0, n_neighbours)
    
    batch_img_names = []
    batch_id_nums = []
    batch_steer_truths = []
    batch_steer_preds = []
    batch_yds = []
    
    for batch_id in range(len(input_imgs)):
        batch_i_steer = og_input_steerings[batch_id,:]
        batch_yds.append(yd[batch_id].data.cpu().numpy())
        
        ret_img_names = []
        ret_id_nums = []
        ret_steer_preds = []
                
        image_steering_label = []
        steer_diff = 0

        for top_n_id in range(n_neighbours):
            ret_ind = retrieval[batch_id, top_n_id]
            img_steer_lab = train_loader.dataset[ret_ind][1].cpu().numpy()
            
            ret_img_names.append(train_loader.dataset.run_files[ret_ind].filename)
            ret_id_nums.append(ret_ind)
            ret_steer_preds.append(img_steer_lab)            
            
        batch_img_names.append(ret_img_names)
        batch_id_nums.append(ret_id_nums)
        batch_steer_truths.append(batch_i_steer)
        batch_steer_preds.append(ret_steer_preds)
    

    batch_img_names = np.array(batch_img_names)
    batch_id_nums = np.array(batch_id_nums)
    batch_steer_truths = np.array(batch_steer_truths)
    batch_steer_preds = np.array(batch_steer_preds)
    batch_losses = np.array(loss.cpu().data)
    batch_yds = np.array(batch_yds)
    
    all_yds.append(batch_yds)
    all_img_names.append(batch_img_names)
    all_id_nums.append(batch_id_nums)
    all_steer_truths.append(batch_steer_truths)
    all_steer_preds.append(batch_steer_preds)
    all_losses.append(batch_losses)

all_img_names = np.array(all_img_names)
all_id_nums = np.array(all_id_nums)
all_steer_truths = np.array(all_steer_truths)
all_steer_preds = np.array(all_steer_preds)
all_losses = np.array(all_losses)

hf.create_dataset('all_img_names', data=all_img_names)
hf.create_dataset('all_id_nums', data=all_id_nums)
hf.create_dataset('all_steer_truths', data=all_steer_truths)
hf.create_dataset('all_steer_preds', data=all_steer_preds)
hf.create_dataset('all_losses', data=all_losses)

hf.create_dataset('all_yds', data=all_yds)



hf.close()


# a = {'all_img_names': all_img_names, 
#      'all_id_nums': all_id_nums,
#      'all_steer_truths': all_steer_truths, 
#      'all_steer_preds': all_steer_preds, 
#      'all_losses': all_losses,
#      'all_steer_diffs': all_steer_diffs}
# 
# with open('data.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


