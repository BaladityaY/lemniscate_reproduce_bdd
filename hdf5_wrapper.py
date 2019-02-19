'''
@author: Sascha Hornauer
'''
import h5py
from enum import Enum

storage_type = Enum('storage_type','file_handle string')

class DB_object(object):
    
    def __init__(self, filename, db_type, getitem_wrapper):
        
        self.getitem_wrapper = getitem_wrapper
        if db_type == storage_type.string:
            self.handle = filename
        elif db_type == storage_type.file_handle:
            self.handle = h5py.File(filename,'r')
    
    def __getitem__(self,index):
        return self.getitem_wrapper(self.handle,index)
    
    def close(self):
        self.handle.close()

class DB_manager(object):
    
    def __init__(self, preload_to_mem, keep_memory_free, max_open_files = 15000):
        
        self.preload_to_mem = preload_to_mem
        self.keep_memory_fre = keep_memory_free
        self.max_open_files = 15000
        self.open_files = []
        
    def get_free_mem(self):
        meminfo = dict((i.split()[0].rstrip(':'),int(i.split()[1])) for i in open('/proc/meminfo').readlines())
        mem_available = float(meminfo['MemAvailable'])
        mem_total = float(meminfo['MemTotal'])
        return (mem_available/mem_total)*100.

    def __getitem_wrapper(self, handle, index):
        # This method passes the hdf5 handle.
        # If the file has to be opened to pass the handle
        # it will close another file in the open file list to 
        # stay below the max open files. 
        if handle == storage_type.string:
            if len(self.open_files) > self.max_open_files:
                self.open_files[0].close()            
            return_handle = h5py.File(handle.filename,'r')
            
        elif handle == storage_type.file_handle:
            

    def add_db_object(self, filename):
        print "Add {}".format(filename)

        # If too many files are opened, add it as string containing db item
        # to open the file on demand
        if len(self.open_files) > self.max_open_files:        
            db_object = DB_object(filename,storage_type.string,__getitem_wrapper)
        else:
            db_object = DB_object(filename,storage_type.file_handle,__getitem_wrapper)
            self.open_files.append(db_object)   
        
        return db_object
        

# The following code is to add the memory holding functionality later
#         if self.preload_to_mem and self.get_free_mem() > self.keep_memory_free:
#             self.in_memory[filename] = DB_file(filename)

# def load_to_mem(hdf_reference):
#     '''
#     Loads the content of the hdf_reference dataset into memory, using the dataset reference as a key.
#     This is only done if that reference does not yet exist in the buffer
#     '''
#     if not hdf_reference in memory_data_buffer: 
#         memory_data_buffer.update({hdf_reference:hdf_reference[:]})
#     # Return a pointer to the data in the dict
#     return memory_data_buffer.get(hdf_reference)


    
if __name__ == '__main__':
    print list(storage_type)
    
        

