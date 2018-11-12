"""Utility classes for training."""
import os

import torch

def get_device(gpu_id):
    if gpu_id == 'cpu':
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.set_device(gpu_id)
            torch.cuda.device(gpu_id)
            return device
        else:
            device = torch.device("cpu")
            device_name = "cpu"
            return device

class MomentCounter:
    """Notify after N Data Moments Passed"""

    def __init__(self, n):
        self.start = 0
        self.n = n

    def step(self, data_index):
        if data_index.ctr - self.start >= self.n:
            self.start = data_index.ctr
            return True
        return False


def csvwrite(filename, objs):
    with open(filename, 'a') as csvfile:
        csvfile.write(",".join([str(x) for x in objs]) +'\n')


class LossLog:
    """Keep Track of Loss, can be used within epoch or for per epoch."""

    def __init__(self):
        self.ctr = 0
        self.total_loss = 0

    def add(self, loss):
        self.total_loss += loss
        self.ctr += 1

    def average(self):
        return self.total_loss / (self.ctr * 1.)

def save_net(weights_file_name, net):
    torch.save(
        net.state_dict(),
        os.path.join(
            ARGS.save_path,
            weights_file_name +
            '.weights'))
    # Infer files are no longer created. Use e.g.
    #save_data = torch.load(os.path.join(ARGS.save_path, git_branch_name+"_epoch%02d.weights" % (epoch - 1,)),map_location=lambda storage, loc: storage)
    # to load weights onto the right gpu
    # Next, save for inference (creates ['net'] and moves net to GPU #0)
    #weights = {'net': net.state_dict().copy()}
    #for key in weights['net']:
    #    weights['net'][key] = weights['net'][key].cuda(device=0)
    #torch.save(weights,
    #           os.path.join(ARGS.save_path, weights_file_name + '.infer'))

##########
# Image creator from text in pillow
##########

from PIL import Image

def get_img(text, img=None):
    img = Image.new('RGB', (60, 30), color = 'red')
