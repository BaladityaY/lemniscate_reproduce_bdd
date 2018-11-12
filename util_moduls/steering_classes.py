'''
Created on Aug 7, 2018

@author: sascha
'''
import numpy as np
from itertools import permutations
import time

'''
This module creates a lookup table of steering combinations with length 3. 

It also provides methods to calculate the most matching class for three steering values to fall in.
'''

steering_angles = np.linspace(-0.5,0.5,num=10) # This could also be changed to a finer resolution in front of the angles
steering_classes = [entry for entry in permutations(steering_angles,3)]

def get_class_no(steering_np_array):
    
    return sum_diff_method(np.array(steering_np_array))

    
def sum_diff_method(np_array):
    
    class_distance = None
    for i in range(len(steering_classes)):
        
        class_member = steering_classes[i]
        
        if class_distance is None:
            nearest_class_id = i
            class_distance = np.sum(np.abs(class_member - np_array))
        else:
            current_distance = np.sum(np.abs(class_member - np_array))
            
            if current_distance < class_distance:
                nearest_class_id = i
                class_distance = current_distance
                
    
    return nearest_class_id

def get_steering_from_class_vec(class_vector):
    
    steering_class_id = np.argmax(class_vector)
    
    return steering_classes[steering_class_id]


if __name__ == '__main__':
    
    get_class_no(np.array([0.1105,  0.0102, -0.2056]))
    
    pass



