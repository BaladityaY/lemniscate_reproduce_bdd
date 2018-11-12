'''
Created on Jul 8, 2018

@author: picard
'''

import torch
import cv2
import numpy as np

def show_image(camera_tensor, grayscale=False, delay=30):
    '''
    Will show one camera image from the camera tensor. It will always take the first
    batch and the first image, even if more images and a bigger batchsize is used.
    '''
    
    if not grayscale:
        one_img = camera_tensor[0][0:3]
        one_img = one_img.data.cpu().numpy()
        one_img = one_img.transpose(1,2,0)
        one_img += 0.5
        one_img *= 255
        one_img = one_img.astype(np.uint8)
        one_img = cv2.cvtColor(one_img,cv2.COLOR_RGB2BGR)
    else:
        one_img = camera_tensor[0][0]
        one_img = one_img.data.cpu().numpy()
        one_img += 0.5
        one_img *= 255
        one_img = one_img.astype(np.uint8)
    
    #print one_img.shape
    cv2.imshow("Camera Image",one_img)
    cv2.waitKey(delay)
    #cv2.destroyAllWindows()
    #exit()

if __name__ == '__main__':
    pass