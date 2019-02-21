'''
@author: Sascha Hornauer
'''
import h5py
import os.path

class DB_item(object):
    
    def __init__(self, images, speeds):
        self.images = images
        self.speeds = speeds   
        
        

class DB_manager(object):
    
    def __init__(self, preload_to_mem, keep_memory_free, data_file_path):        
        
        # Most likely this is not needed anymore
        self.preload_to_mem = preload_to_mem
        self.keep_memory_free = keep_memory_free
        print "Loading {}".format(data_file_path)
        print "The file exists: {}".format(os.path.isfile(data_file_path)) 
        
        '''
        Since there is only one hdf5 file left, it's pretty easy to corrupt it. Always Always Always
        make sure to close it.
        '''
        hdf5_file =  h5py.File(data_file_path,'r')
        
        # The following format, where the whole path is contained in the keys,
        # is a mistake during the file generation. File generation takes almost
        # an hour so for a quick fix this is kept and should be changed int he future.
        
        # Get if it's train or val data
        file_keys = hdf5_file.keys()
        
        self.sequences = []
        
        for file_key in file_keys:
            
            images = hdf5_file[file_key]['image']['encoded']                
            speeds = hdf5_file[file_key]['image']['speeds']
            self.sequences.append(DB_item(images,speeds))
    
    
    def get_sequence_list(self):
        return self.sequences
    

