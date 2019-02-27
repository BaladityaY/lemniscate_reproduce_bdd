

import argparse
import os
import sys
from future import division
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from util_moduls.Utils import get_device

import datasets
import models
import math

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN
from Dataset import Dataset
from torch.autograd.variable import Variable

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train-data', metavar='DIR',
                     help='path to the training file')
parser.add_argument('--val-data', metavar='DIR',
                     help='path to the validation file')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default=0, type=int, 
                    help='gpu id to run on')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float, 
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')

best_prec1 = -500000
best_prec1_past = -500000
best_prec1_future = -500000

n_frames = 6

def resize2d(img, size):
    return (torch.nn.functional.adaptive_avg_pool2d(Variable(img,requires_grad=False), size)).data

def add_epoch_score(filename, epoch, score):
    f = open(filename, 'a')
    f.write('Epoch {}: {}\n'.format(epoch, score))
    f.close()
    
    return

def main():
    global args, best_prec1, best_prec1_past, best_prec1_future
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](low_dim=args.low_dim)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.to('cuda')
        else:
            model = torch.nn.DataParallel(model).to('cuda')
    else:
        model.to('cuda')
        model = torch.nn.parallel.DistributedDataParallel(model)


    # Data loading code
    training_file = args.train_data
    validation_file = args.val_data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = Dataset(training_file, n_frames)
    val_dataset = Dataset(validation_file, n_frames)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, #(train_sampler is None), 
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    # define lemniscate and loss function (criterion)
    ndata = train_dataset.__len__()
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.gpu, args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m).to('cuda')
        criterion = NCECriterion(ndata).to('cuda')
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).to('cuda')
        criterion = nn.CrossEntropyLoss().to('cuda')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    if args.evaluate:
        kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)
        return

    #criterion = criterion.to('cuda')
    
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, lemniscate, criterion, optimizer, epoch)
        print "Exited because this is a debug run"
        exit()
        # evaluate on validation set
        prec1, prec1_past, prec1_future = NN(epoch, model, lemniscate, train_loader, val_loader)

        add_epoch_score('epoch_scores.txt', epoch, prec1)
        add_epoch_score('epoch_scores_past.txt', epoch, prec1_past)
        add_epoch_score('epoch_scores_future.txt', epoch, prec1_future)
  
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch)

        is_best_past = prec1_past > best_prec1_past
        best_prec1_past = max(prec1_past, best_prec1_past)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1_past': best_prec1_past,
            'optimizer' : optimizer.state_dict(),
        }, is_best_past, epoch, best_mod='_past')
        
        is_best_future = prec1_future > best_prec1_future
        best_prec1_future = max(prec1_future, best_prec1_future)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1_future': best_prec1_future,
            'optimizer' : optimizer.state_dict(),
        }, is_best_future, epoch, best_mod='_future')
        
        
    # evaluate KNN after last epoch
    kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)


def train(train_loader, model, lemniscate, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    
    # If training should be ended for debug purposes after a fixed amount of data samples
    debug = False
    debug_end_cycle = 5
    
    for i, (input_imgs,action_probabilities, indices) in enumerate(train_loader):
        
        if debug and i >= debug_end_cycle:
            exit()
            
        # measure data loading time
        data_time.update(time.time() - end)

        indices = indices.to('cuda')
        input_imgs = input_imgs.to('cuda')
        action_probabilities = action_probabilities.to('cuda') 
        # Change the image size so it fits to the network
        #input_imgs = resize2d(input_imgs, (224,224))
        # The images are now already in the right size
        input_imgs = input_imgs[:,0:9,:,:] #extract the first 3 images
        action_probabilities = action_probabilities[:,0:3] #extract steers first 3 out of 6 
                
# Code to see the images        
#         for j in range(input_imgs.size()[0]):
#             for img in input_imgs.data.numpy():
#                  
#                 #print img[0:3].transpose((1,2,0)).shape                
#                 cv2.imshow("Test",img[0:3].transpose((1,2,0))+0.5)
#                 cv2.waitKey(400)
        #print input_imgs.size()
        feature = model(input_imgs, action_probabilities)
        output = lemniscate(feature, indices)
        loss = criterion(output, indices) / args.iter_size

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input_imgs.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t').format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)
                  
                  
        


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', best_mod=''):
    filename_split = filename.split('.')
    filename = filename_split[0] + '_epoch{:02d}'.format(epoch) + '.' + filename_split[1] + '.' + filename_split[2]
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best{}.pth.tar'.format(best_mod))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr
    if epoch < 120:
        lr = args.lr
    elif epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    #lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
