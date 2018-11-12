'''
Created on Nov 11, 2018

@author: Sascha Hornauer

Copied helper methods from the BDD_Driving_Model repository where tensorflow dependencies were removed

Original code at: https://github.com/gy20073/BDD_Driving_Model by gy20073

'''
import math
import numpy as np

turn_str2int={'not_sure': -1, 'straight': 0, 'slow_or_stop': 1,
              'turn_left': 2, 'turn_right': 3,
              'turn_left_slight': 4, 'turn_right_slight': 5,} #'acceleration': 6, 'deceleration': 7}


ego_previous_nstep = 30
n_sub_frame = 108
release_batch = True
resize_images = "228,228"
balance_drop_prob = -1.0

decode_downsample_factor = 1
temporal_downsample_factor = 5
data_provider = "nexar_large_speed"
# ground truth maker
speed_limit_as_stop = 2.0
stop_future_frames = 1
deceleration_thres = 1
no_slight_turn = True
deceleration_thres = 1

frame_rate = 15

class BDD_Helper(object):
    
    
    turn_int2str={y: x for x, y in turn_str2int.iteritems()}
    naction = np.sum(np.less_equal(0, np.array(turn_str2int.values())))
    
    @staticmethod
    def future_smooth(actions, naction, nfuture):
        # TODO: could add weighting differently between near future and far future
        # given a list of actions, for each time step, return the distribution of future actions
        l = len(actions) # action is a list of integers, from 0 to naction-1, negative values are ignored
        out = np.zeros((l, naction), dtype=np.float32)
        for i in range(l):
            # for each output position
            total = 0
            for j in range(min(nfuture, l-i)):
                # for each future position
                # current deal with i+j action
                acti = i + j
                if actions[acti]>=0:
                    out[i, actions[acti]] += 1
                    total += 1
            if total == 0:
                out[i, BDD_Helper.turn_str2int['straight']] = 1.0
            else:
                out[i, :] = out[i, :] / total
        return out
    
    @staticmethod
    def speed_to_course(speed):
        pi = math.pi
        if speed[1] == 0:
            if speed[0] > 0:
                course = pi / 2
            elif speed[0] == 0:
                course = None
            elif speed[0] < 0:
                course = 3 * pi / 2
            return course
        course = math.atan(speed[0] / speed[1])
        if course < 0:
            course = course + 2 * pi
        if speed[1] > 0:
            course = course
        else:
            course = pi + course
            if course > 2 * pi:
                course = course - 2 * pi
        assert not math.isnan(course)
        return course
    
    
    @staticmethod
    def to_course_list(speed_list):
        l = speed_list.shape[0]
        course_list = []
        for i in range(l):
            speed = speed_list[i,:]
            course_list.append(BDD_Helper.speed_to_course(speed))
        return course_list
    
    @staticmethod
    def turning_heuristics(speed_list, speed_limit_as_stop=0):
        course_list = BDD_Helper.to_course_list(speed_list)
        speed_v = np.linalg.norm(speed_list, axis=1)
        l = len(course_list)
        action = np.zeros(l).astype(np.int32)
        course_diff = np.zeros(l).astype(np.float32)
    
        enum = turn_str2int
    
        thresh_low = (2*math.pi / 360)*1
        thresh_high = (2*math.pi / 360)*35
        thresh_slight_low = (2*math.pi / 360)*3
    
        def diff(a, b):
            # return a-b \in -pi to pi
            d = a - b
            if d > math.pi:
                d -= math.pi * 2
            if d < -math.pi:
                d += math.pi * 2
            return d
    
        for i in range(l):
            if i == 0:
                action[i] = enum['not_sure']
                continue
    
            # the speed_limit_as_stop should be small,
            # this detect strict real stop
            if speed_v[i] < speed_limit_as_stop + 1e-3:
                # take the smaller speed as stop
                action[i] = enum['slow_or_stop']
                continue
    
            course = course_list[i]
            prev = course_list[i-1]
    
            if course is None or prev is None:
                action[i] = enum['slow_or_stop']
                course_diff[i] = 9999
                continue
    
            course_diff[i] = diff(course, prev)*360/(2*math.pi)
            if thresh_high > diff(course, prev) > thresh_low:
                if diff(course, prev) > thresh_slight_low:
                    action[i] = enum['turn_right']
                else:
                    action[i] = enum['turn_right_slight']
    
            elif -thresh_high < diff(course, prev) < -thresh_low:
                if diff(course, prev) < -thresh_slight_low:
                    action[i] = enum['turn_left']
                else:
                    action[i] = enum['turn_left_slight']
            elif diff(course, prev) >= thresh_high or diff(course, prev) <= -thresh_high:
                action[i] = enum['not_sure']
            else:
                action[i] = enum['straight']
    
            if no_slight_turn:
                if action[i] == enum['turn_left_slight']:
                    action[i] = enum['turn_left']
                if action[i] == enum['turn_right_slight']:
                    action[i] = enum['turn_right']
    
            # this detect significant slow down that is not due to going to turn
            if deceleration_thres > 0 and action[i] == enum['straight']:
                hz = frame_rate / temporal_downsample_factor
                acc_now = (speed_v[i] - speed_v[i - 1]) / (1.0 / hz)
                if acc_now < - deceleration_thres:
                    action[i] = enum['slow_or_stop']
                    continue
    
        # avoid the initial uncertainty
        action[0] = action[1]
        return action
    
    @staticmethod
    def turn_future_smooth(speed, nfuture, speed_limit_as_stop):
        # this function takes in the speed and output a smooth future action map
        turn = BDD_Helper.turning_heuristics(speed, speed_limit_as_stop)
        smoothed = BDD_Helper.future_smooth(turn, BDD_Helper.naction, nfuture)
        return smoothed
    


if __name__ == '__main__':
    print "test"
    
    pass