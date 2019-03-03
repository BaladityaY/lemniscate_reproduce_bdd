import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import datasets
from lib.utils import AverageMeter
import torchvision.transforms as transforms
import numpy as np
from torch.autograd.variable import Variable
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import cPickle as pickle
import matplotlib.pyplot as plt
import scipy
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch.nn as nn

def resize2d(img, size):
    return (torch.nn.functional.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data


def bce(t1, t2):
    #t1 = Variable(torch.from_numpy(a1).type(torch.FloatTensor))
    #t2 = Variable(torch.from_numpy(a2).type(torch.FloatTensor))

    #print t1
    #print t2

    criterion = nn.BCELoss(reduce=False)
    loss = criterion(t1, t2)

    return np.mean(np.array(loss).flatten())

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()

    start_time = time.time()

    topk = 50
    steer_eval = {}
    
    correct = 0.
    correct_past = 0.
    correct_future = 0.
    
    total = 0
    
    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        #trainloader.dataset.transform = testloader.dataset.transform
        #temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (input_imgs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            batchSize = input_imgs.size(0)
            
            input_imgs = input_imgs[:,0:9,:,:] #extract only img 3 out of 6
            targets = targets[:,0:3] #extract steers first 3 out of 6
            
            input_imgs = resize2d(input_imgs, (224,224))
            
            features = net(input_imgs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        #trainloader.dataset.transform = transform_bak
    
    end = time.time()
    with torch.no_grad():
        
        print "Start of testing {}".format(time.time() - start_time)
        for batch_idx, (input_imgs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(async=True)
            indexes = indexes.cuda(async=True)
            input_imgs = input_imgs.cuda(async=True)
            
            print "Elements loaded onto GPU {}".format(time.time() - start_time)
            
            batchSize = input_imgs.size(0)
            print "Batch {} of {}".format(batch_idx,len(testloader))
            #og_input_imgs = input_imgs.clone().cpu().numpy()
            og_targets = targets.clone()
            
            #print('input_imgs shape: {}'.format(input_imgs.shape))
            #print('og_targets: {}'.format(og_targets))
            #print input_imgs.size()
            input_imgs = input_imgs[:,0:9,:,:] #extract only img 1 through 3
            targets = targets[:,0:3] #extract steers first 3 targets
                
            #input_imgs = resize2d(input_imgs, (224,224))
            # The new images are already in the right size
            #print input_imgs.size()
            features = net(input_imgs, targets)
            
            print "Images and targets put through network {}".format(time.time() - start_time)
            
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(topk, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, topk).clone()
            yd = yd.narrow(1, 0, topk)
            
            #print('retrieval shape: {}'.format(retrieval.shape))
            #print('retrieval numbers: {}'.format(retrieval))
            
            print "Top 50 NNs retrieved {}".format(time.time() - start_time)
            
            image_steering_labels = []
            image_steering_labels_past = []
            image_steering_labels_future = []
            indexes = np.array(indexes)
            
            for batch_id in range(len(input_imgs)):
                steer_eval[indexes[batch_id]] = {'og_steer': og_targets[batch_id, :],
                                                 'nn_steers': None,
                                                 'nn_ids': None}
                nn_steers = []
                nn_ids = []
                        
                #print('og_targets.shape: {}'.format(og_targets.shape))
                #print('batch_id: {}'.format(batch_id))
                batch_i_steer = og_targets[batch_id,:] # 6x6
                
                image_steering_label = 0
                image_steering_label_past = 0
                image_steering_label_future = 0

                #print batch_i_steer
                for top5_id in range(topk):
                    ret_ind = int(retrieval[batch_id, top5_id])
                    img_steer_lab = trainloader.dataset[ret_ind][1]
                    #img_steer_lab = trainloader.dataset.get_label(retrieval[batch_id, top5_id])[1] #old way
                    image_steering_label += bce(img_steer_lab, batch_i_steer) #np.abs((np.array(img_steer_lab) - batch_i_steer)/2.)
                    image_steering_label_past += bce(img_steer_lab[0:3], batch_i_steer[0:3]) #np.abs((np.array(img_steer_lab[0:3]) - batch_i_steer[0:3])/2.)
                    image_steering_label_future += bce(img_steer_lab[3:6], batch_i_steer[3:6]) #np.abs((np.array(img_steer_lab[3:6]) - batch_i_steer[3:6])/2.)

                    nn_steers.append(img_steer_lab.clone().data.cpu().numpy())
                    nn_ids.append(ret_ind)

                steer_eval[indexes[batch_id]]['nn_steers'] = np.array(nn_steers)
                steer_eval[indexes[batch_id]]['nn_ids'] = np.array(nn_ids)
                    
                image_steering_labels.append(image_steering_label/topk)
                image_steering_labels_past.append(image_steering_label_past/topk)
                image_steering_labels_future.append(image_steering_label_future/topk)
                

                                
            image_steering_labels = -1*np.array(image_steering_labels)
            image_steering_labels_past = -1*np.array(image_steering_labels_past)
            image_steering_labels_future = -1*np.array(image_steering_labels_future)
            
            print "Loss calculated {}".format(time.time() - start_time)
            
            #print('image_steering_labels shape: {}'.format(image_steering_labels.shape))
            #print('image_steering_labels numbers: {}'.format(image_steering_labels))
            #image_steering_labels = 1 - image_steering_labels
            #image_steering_labels_past = 1 - image_steering_labels_past
            #image_steering_labels_future = 1 - image_steering_labels_future
            
            #print('image_steering_labels numbers 2: {}'.format(image_steering_labels))
            
            '''
            batch_i_steer = np.array(targets[batch_id,:])
            
            img_steer_lab = trainloader.dataset.get_label(retrieval_index)[1]
                    
            image_steering_labels.append(np.array(img_steer_lab) - batch_i_steer)
            '''
            
            # Sascha: The next line makes the reporting inconsistent with the reporting
            # during training time. Changed the code here to have batches numbered and not
            # individual data points
            #total += indexes.shape[0]
            
            correct += np.sum(image_steering_labels)
            correct_past += np.sum(image_steering_labels_past)
            correct_future += np.sum(image_steering_labels_future)
            
            cls_time.update(time.time() - end)
            end = time.time()
            
            print "Batch finished {}".format(time.time() - start_time)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top5: {:.2f}'.format(
                  batch_idx, len(testloader), correct/total, net_time=net_time, cls_time=cls_time))

            
        with open('steer_eval_epoch{}.pkl'.format(epoch), 'wb') as handle:
            pickle.dump(steer_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)
                             
        #correct_rate = np.array(correct) # Changed this from correct_rate which was not known
        #print('correct_rate mean: {}, std: {}'.format(np.mean(correct_rate), np.std(correct_rate)))

    return correct/total, correct_past/total, correct_future/total

def img_error_bar(img, error, color):
    error = np.max((np.min((1, error)), -1))
    x_tenth = img.shape[0]/10
    y_half = img.shape[1]/2
    y_error = int(img.shape[1]*error/2)

    
    print('error: {}, x_tenth: {}, y_half: {}, y_error: {}, min: {}, max: {}'.format(error, x_tenth, y_half, y_error, 
                                                                                     np.min((y_half,y_half+y_error)), 
                                                                                     np.max((y_half,y_half+y_error))))
    
    bar = np.zeros((x_tenth, np.abs(y_error), 3))
    
    if color == 'green':
        bar[:,:,1] = 255
        
    if color == 'red':
        bar[:,:,0] = 255
    
    img[0:x_tenth, np.min((y_half,y_half+y_error)):np.max((y_half,y_half+y_error)), :] = bar        
    
        
    return img


def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, to_plot=False, 
        img_name='current_retrievals'):
    
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    sample_len = trainFeatures.shape[1]/10
    trainFeatures = trainFeatures[:, 0:sample_len]
    
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.train_labels[0:sample_len]).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (input_imgs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(async=True)
            
            input_imgs = input_imgs[:,0:9,:,:] #extract only img 3 out of 6
            targets = targets[:,0:3] #extract steers first 3 out of 6
            
            input_imgs = resize2d(input_imgs, (224,224))
            
            batchSize = input_imgs.size(0)
            features = net(input_imgs)
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
            # So a batch of image features is calculated and then saved in the tensor trainFeatures. This is done at
            # all of the dimensions at dim 0 and in increments of a batch size at dim 1 
            
        trainLabels = torch.LongTensor(temploader.dataset.train_labels).cuda()
        trainloader.dataset.transform = transform_bak
    
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()

        all_targets = None
        all_diffs = None
        
        all_diffs_mean = None
        all_bin_diffs_mean = None
        all_bin_diffs_mode = None

        
        for batch_idx, (input_imgs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(async=True)
            indexes = indexes.cuda(async=True)
            
            og_input_imgs = input_imgs.clone().cpu().numpy()
            og_targets = targets.clone().cpu().numpy()
            
            input_imgs = input_imgs[:,0:9,:,:] #extract only img 3 out of 6
            targets = targets[:,0:3] #extract steers first 3 out of 6
            
            batchSize = input_imgs.size(0)
            #print('original test img shape: {}'.format(input_imgs.shape))
            #print('original test img numbers: {}'.format(input_imgs))
            input_imgs = resize2d(input_imgs, (224,224))
            
            features = net(input_imgs,targets)
            
            net_time.update(time.time() - end)
            end = time.time()
            
            dist = torch.mm(features, trainFeatures)
            
            top_features, top_feature_indices = dist.topk(K, dim=1, largest=True, sorted=True)
            
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            # trainLabels are the labels of all images, in the correct order when the testloader was not shuffled. 
            # So candidates is now a flat list of all trainingLabels for each batch
            retrieval = torch.gather(candidates, 1, top_feature_indices)
            # So now get from the trainingLabels the labels of the images, as indexed by top_feature_indices
            # This should work because the trainFeatures are created in the order as the images
            # are in the unshuffled dataset.
            
            # The next line is a guess that it works that ways
            top5_image_indices = retrieval[:,0:5]
            
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            # C is the max trainlabel. So the trainlabels must be numeric class numbers.
            
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            # The image labels are laid flat and a 1 is written into this tensor
            # at each index, as given by the label. This assumes I guess that the
            # image labels are all iid and numbers. 
            
            yd_transform = top_features.clone().div_(sigma).exp_()
            # top_features's are the actual distances of the features to the trainFeatures in the memomry bank.
            # So this is creating a copy, divides it by sigma. Sigma is nce_t, the number of negative
            # examples for NCE. 
            # So these are weights of neighbours now
            
            
                    
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            # All right so the weights of the neighbours are multiplied per batch with the one hot vector. So the weights are only
            # multiplied if the one hot vector shows that a image at that position is among the winners. 
            # The resulting matrix should have batchSize x features x images many entries I guess. 
            # In that matrix only the discounted weights of winners should be present
            # They are summed within each line so withing each batch
            
            # This assumse I think that the training labels are ascending index numbers, starting with 0
            # and ending with the "max label" which has to be len(imgs)
            _, predictions = probs.sort(1, True)
            # They sort in the big matrix per line but DESCENDING. So 
            # the lowest predictions or the lowest still existing discounted neighbouring weights are
            # now the most left in each line????

            print "Current indexes {}".format(indexes)
            # Find which predictions match the target
            correct = predictions.eq(indexes.data.view(-1,1))
            # The predictions are only the indices, saying which indices of which labels
            # belong to the biggest weights. The targets are the indices of the images
            # At the very left in predictions we find the index of the biggest dicounted 
            # feature in the memory bank. At the same position in target is the first image label
            # which is also the index of that image
            # This works most likely because the targets are ascending, also the image indexes
            # and the predictions are only the indices for the predictions. 
            
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += indexes.size(0)

            
            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))
            
            
            ####
            input_imgs = input_imgs.cpu().data.numpy()
            input_imgs = (input_imgs + .5)*255
            og_input_imgs = (og_input_imgs + .5)*255
            '''
            input_imgs = input_imgs.view(input_imgs.size()[0], 
                                         input_imgs.size()[2], 
                                         input_imgs.size()[3], 
                                         input_imgs.size()[1])
            '''
            input_imgs = np.transpose(input_imgs, (0, 2, 3, 1))
            og_input_imgs = np.transpose(og_input_imgs, (0, 2, 3, 1))
            
            padlen = 10
            xlen = og_input_imgs[0].shape[0]
            ylen = og_input_imgs[0].shape[1]
            
            #print("new test img shape: {}".format(input_imgs.shape))

            all_imgs = None
            # Show results for 5 batches
            for batch_id in range(8):
                
                #print('og_input_imgs shape: {} \n ylen: {}'.format(og_input_imgs.shape, ylen))
                #print('shape 1: {}'.format(og_input_imgs[batch_id, :,:, 0:3].shape))
                #print('shape 2: {}'.format(np.zeros((padlen, ylen, 3)).shape))
                
                batch_imgs = None
                og_imgs = [og_input_imgs[batch_id, :,:, 0:3], 
                           og_input_imgs[batch_id, :,:, 3:6],
                           og_input_imgs[batch_id, :,:, 6:9],
                           og_input_imgs[batch_id, :,:, 9:12],
                           og_input_imgs[batch_id, :,:, 12:15],
                           og_input_imgs[batch_id, :,:, 15:18]]
                
                for og_ind, og_img in enumerate(og_imgs):
                    og_img = img_error_bar(og_img, og_targets[batch_id, og_ind], 'green')
                    #print('og_ind {} img shape: {}'.format(og_ind, og_img.shape))
                    if batch_imgs is None:
                        batch_imgs = og_img
                    else:
                        batch_imgs = np.vstack((batch_imgs, og_img))
                
                batch_imgs = np.vstack((batch_imgs, np.zeros((padlen, ylen, 3))))
                
                '''
                batch_imgs = np.vstack((og_input_imgs[batch_id, :,: 0:3], 
                                        og_input_imgs[batch_id, :,:, 6:9], 
                                        og_input_imgs[batch_id, :,:, 12:15],
                                        og_input_imgs[batch_id, :,:, 18:21], 
                                        og_input_imgs[batch_id, :,:, 24:27], 
                                        og_input_imgs[batch_id, :,:, 30:33],
                                        np.zeros((padlen, ylen, 3)))) #None
                '''
                
                #print('batch shape 1: {}'.format(batch_imgs.shape))
                    
                batch_imgs = np.hstack((batch_imgs, np.zeros((batch_imgs.shape[0], padlen, 3))))
                      
                fetched_steer_diffs = []
                fetched_steer_bin_diffs = []
                
                batch_i_steer = np.array(og_targets[batch_id,:])
                print('batch_i_steer: {}'.format(batch_i_steer))
                      
                for retrieval_index in top5_image_indices[batch_id]:
                    print("retrieval index: {}".format(retrieval_index))
                    
                    # Images come as triplets with three color channels
                    
                    fetched_images =  trainloader.dataset.get_image(retrieval_index)
                    #print('batch_imgs shape: {}'.format(batch_imgs.shape))
                    #print('fetched_imgs shape: {}'.format(fetched_images.shape))
                    
                    '''
                    fetched_images = fetched_images[0].reshape((1, 
                                                                fetched_images[0].shape[0], 
                                                                fetched_images[0].shape[1], 
                                                                fetched_images[0].shape[2]))
                    '''
                    fetched_imgs = fetched_images[0:6]
                    img_steer_lab = np.array(trainloader.dataset[retrieval_index][1])
                    #print "Steering commands: {}".format(trainloader.dataset.get_label(retrieval_index)[0:3])
                    
                    top5_images = None
                    
                    for img_ind, img in enumerate(fetched_images):
                        '''
                        if img_ind == 0:
                            print("train db img shape: {}".format(img.shape))
                        '''
                        
                        img = scipy.misc.imresize(img, (xlen, ylen))

                        
                        img = img_error_bar(img, (img_steer_lab[img_ind] - batch_i_steer[img_ind])/2, 'red')
                        
                        '''
                        if img_ind == 0:
                            print("new train db img shape: {}".format(img.shape))
                            
                            print("new train db img numbers: {}".format(img))
                        '''
                        
                        if top5_images is None:
                            top5_images = img
                        else:
                            top5_images = np.vstack((top5_images, img))
                    
                    top5_images = np.vstack((top5_images, np.zeros((padlen, ylen, 3))))
                    
                    '''
                    imgA = fetched_images[0]
                    imgB = fetched_images[1]
                    imgC = fetched_images[2]
                    
                    print("test img shape: {}".format(imgA.shape))
                    
                    top5_images = np.vstack((imgA, imgB, imgC, np.zeros((padlen, ylen, 3))))
                    '''
                    
                    # Append the steering commands
                    
                    fetched_steer_diffs.append(img_steer_lab - batch_i_steer)
                    fetched_steer_bin_diffs.append(np.round((img_steer_lab+1)*(6./2)).astype(int) - 
                                                   np.round((batch_i_steer+1)*(6./2)).astype(int))
                    #image_steering_labels.append(trainloader.dataset.get_label(retrieval_index)[1])
                    
                    
                    if batch_imgs is not None:
                        batch_imgs = np.hstack((batch_imgs, top5_images))
                    else:
                        batch_imgs = top5_images
                
                    
                #print "Inter-batch standard deviation of steering labels: {}".format(np.std(np.array(image_steering_labels)))
                
                print('fetched_steer_diffs: {}'.format(fetched_steer_diffs))
                
                fetched_steer_diffs = np.array(fetched_steer_diffs)
                fetched_steer_bin_diffs = np.array(fetched_steer_bin_diffs)
                
                fetched_modes = stats.mode(fetched_steer_bin_diffs, axis=0)[0][0]
                
                if all_diffs is None:
                    all_diffs = fetched_steer_diffs
                else:
                    all_diffs = np.vstack((all_diffs, fetched_steer_diffs))
                
                if all_diffs_mean is None:
                    all_diffs_mean = np.mean(fetched_steer_diffs, axis=0)
                else:
                    all_diffs_mean = np.vstack((all_diffs_mean, np.mean(fetched_steer_diffs, axis=0)))
                    
                if all_bin_diffs_mean is None:
                    all_bin_diffs_mean = np.mean(fetched_steer_bin_diffs, axis=0)
                else:
                    all_bin_diffs_mean = np.vstack((all_bin_diffs_mean, np.mean(fetched_steer_bin_diffs, axis=0)))
                
                if all_bin_diffs_mode is None:
                    all_bin_diffs_mode = fetched_modes
                else:
                    all_bin_diffs_mode = np.vstack((all_bin_diffs_mode, fetched_modes))
                
                if to_plot:
                    
                    f, ax = plt.subplots(1, 1)
                    for img_steer_diff in fetched_steer_diffs:
                        ax.plot(range(len(img_steer_diff)), img_steer_diff*3, 'r-')
                    for img_steer_bin_diff in fetched_steer_bin_diffs:
                        ax.plot(range(len(img_steer_bin_diff)), img_steer_bin_diff, 'g-')
                    ax.plot(range(len(img_steer_bin_diff)), fetched_modes, 'b-')
                    ax.set_xlim([-.2, 5.2])
                    ax.set_ylim([-6.2, 6.2])
                    plt.show()
                    
                    f, ax = plt.subplots(1, 1)
                    ax.errorbar(range(fetched_steer_diffs.shape[1]), 
                                np.mean(fetched_steer_diffs*3, axis=0), 
                                np.std(fetched_steer_diffs*3, axis=0)/np.sqrt(fetched_steer_diffs.shape[0]), 
                                label='fetched_steer_diffs')
                    ax.errorbar(range(fetched_steer_bin_diffs.shape[1]), 
                                np.mean(fetched_steer_bin_diffs, axis=0), 
                                np.std(fetched_steer_bin_diffs, axis=0)/np.sqrt(fetched_steer_bin_diffs.shape[0]), 
                                label='fetched_steer_bin_diffs')
                    ax.plot(range(fetched_steer_bin_diffs.shape[1]),
                            fetched_modes, 
                            label='fetched_steer_bin_diffs mode')
                    ax.legend()
                    ax.set_xlim([-.2, 5.2])
                    ax.set_ylim([-6.2, 6.2])
                    plt.show()
                
                    
                #print('batch_imgs shape:{}'.format(batch_imgs.shape))
                if all_imgs is not None:
                    all_imgs = np.vstack((all_imgs, batch_imgs, np.zeros((padlen, batch_imgs.shape[1], 3))))
                else:
                    all_imgs = np.vstack((batch_imgs, np.zeros((padlen, batch_imgs.shape[1], 3))))
            
            #print('all imgs numbers:{}'.format(batch_imgs))
            
            scipy.misc.imsave(img_name+'.jpg', all_imgs)
            '''
            im = Image.fromarray(all_imgs)
            im.save("current_retrievals.jpg")
            '''
            
            if all_targets is None:
                all_targets = targets
            else:
                all_targets = np.vstack((all_targets, targets))
                
            if batch_idx >= 50 and to_plot:
                plt.hist(all_targets[:,0], bins=11, label='t0')
                plt.hist(all_targets[:,1], bins=11, label='t1')
                plt.hist(all_targets[:,2], bins=11, label='t2')
                plt.legend(loc='upper right')
                plt.show()
                
                
                plt.hist(all_diffs[:,0], bins=11, label='t0')
                plt.hist(all_diffs[:,1], bins=11, label='t1')
                plt.hist(all_diffs[:,2], bins=11, label='t2')
                plt.legend(loc='upper right')
                plt.show()
                
                
                x = all_targets[:,1] - all_targets[:,0]
                y = all_targets[:,2] - all_targets[:,1]
                #z = all_targets[:,2] - all_targets[:,0]
                
                plt.plot(x, y, 'k.', alpha=.15)
                plt.show()
                
                '''
                f = plt.figure()
                f.set_size_inches(20, 20)
                ax1 = f.add_subplot(2, 2, 1)
                ax1.plot(x, y)
                #ax1.view_init(30, 10)
                
                ax2 = f.add_subplot(2, 2, 2, projection='3d')
                ax2.scatter(all_targets[:,0], all_targets[:,1], all_targets[:,2])
                ax2.view_init(30, 55)
                
                ax3 = f.add_subplot(2, 2, 3, projection='3d')
                ax3.scatter(all_targets[:,0], all_targets[:,1], all_targets[:,2])
                ax3.view_init(30, 100)
                
                ax3 = f.add_subplot(2, 2, 4, projection='3d')
                ax3.scatter(all_targets[:,0], all_targets[:,1], all_targets[:,2])
                ax3.view_init(30, 145)
                '''
                
                
                f, ax = plt.subplots(1,1)
                x = np.array([range(6)]*all_diffs_mean.shape[0])#.reshape((all_diffs_mean.shape[0], 6))
                
                #print('all_diffs_mean shape: {}'.format(all_diffs_mean.shape))
                #print('all_diffs_mean: {}'.format(all_diffs_mean))
                #print('all_bin_diffs_mode shape: {}'.format(all_bin_diffs_mode.shape))
                #print('x shape: {}'.format(x.shape))
                #print('x: {}'.format(x))
                
                ax.plot(x.flatten(), (all_diffs_mean*3.5).flatten(), 'r.', alpha=.15)
                ax.plot(x.flatten(), all_bin_diffs_mean.flatten(), 'g.', alpha=.15)
                ax.plot(x.flatten(), all_bin_diffs_mode.flatten(), 'b.', alpha=.15)
                ax.set_xlim([-.2, 5.2])
                ax.set_ylim([-6.2, 6.2])
                plt.show()
                
                f, ax = plt.subplots(1, 1)
                for diff_mean in all_diffs_mean:
                    ax.plot(range(all_diffs_mean.shape[1]), 
                            diff_mean, 'r-', alpha=.15)
                for diff_mean in all_bin_diffs_mean:
                    ax.plot(range(all_bin_diffs_mean.shape[1]), 
                            diff_mean, 'g-', alpha=.15)
                for diff_mode in all_bin_diffs_mode:
                    ax.plot(range(all_bin_diffs_mode.shape[1]), 
                            diff_mode, 'b-', alpha=.15)
                ax.legend()
                ax.set_xlim([-.2, 5.2])
                ax.set_ylim([-6.2, 6.2])
                plt.show()
                
                f, ax = plt.subplots(1, 1)
                ax.errorbar(range(all_diffs_mean.shape[1]),
                            np.mean(all_diffs_mean*3.5, axis=0),
                            np.std(all_diffs_mean*3.5, axis=0)/np.sqrt(all_diffs_mean.shape[0]),
                            label="all_diffs_mean")
                ax.errorbar(range(all_bin_diffs_mean.shape[1]),
                            np.mean(all_bin_diffs_mean, axis=0),
                            np.std(all_bin_diffs_mean, axis=0)/np.sqrt(all_bin_diffs_mean.shape[0]),
                            label="all_bin_diffs_mean")
                ax.errorbar(range(all_bin_diffs_mode.shape[1]),
                            np.mean(all_bin_diffs_mode, axis=0),
                            np.std(all_bin_diffs_mode, axis=0)/np.sqrt(all_bin_diffs_mode.shape[0]),
                            label="all_bin_diffs_mode")
                ax.legend()
                ax.set_xlim([-.2, 5.2])
                ax.set_ylim([-6.2, 6.2])
                plt.show()
                
                return
            
        return
                
    print(top1*100./total)

    return top1/total

