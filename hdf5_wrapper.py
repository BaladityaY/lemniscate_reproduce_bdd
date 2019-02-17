'''
@author: Sascha Hornauer
'''

from enum import Enum

storage_type = Enum('storage_type','mem_handle file_handle string')

class DB_item(object):
    
    def __init__(self, type):
        pass

class DB_manager(object):
    
    def __init__(self, preload_to_mem, keep_memory_free):
        
        self.preload_to_mem = preload_to_mem
        self.keep_memory_fre = keep_memory_free
        
        self.open_files = []
        
    def get_free_mem(self):
        meminfo = dict((i.split()[0].rstrip(':'),int(i.split()[1])) for i in open('/proc/meminfo').readlines())
        mem_available = float(meminfo['MemAvailable'])
        mem_total = float(meminfo['MemTotal'])
        return (mem_available/mem_total)*100.


# def load_to_mem(hdf_reference):
#     '''
#     Loads the content of the hdf_reference dataset into memory, using the dataset reference as a key.
#     This is only done if that reference does not yet exist in the buffer
#     '''
#     if not hdf_reference in memory_data_buffer: 
#         memory_data_buffer.update({hdf_reference:hdf_reference[:]})
#     # Return a pointer to the data in the dict
#     return memory_data_buffer.get(hdf_reference)

    def add_file(self, filename):
        print "Add {}".format(filename)

        pass
    
    def __getitem__(self,index):
        print index
        
#         if self.preload_to_mem and self.get_free_mem() > self.keep_memory_free:
#             self.in_memory[filename] = DB_file(filename)
        
        return self


    
if __name__ == '__main__':
    print list(storage_type)
    
        

