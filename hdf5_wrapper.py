'''
@author: Sascha Hornauer
'''

from enum import Enum
import h5py
from collections import deque

storage_type = Enum('storage_type', 'file_handle string')

class DB_item(object):
    
    def __init__(self, filename, db_type, close_open_file, stack_open_file_method):
        self.type = db_type
        self.filename = filename
        self.count_file_method = close_open_file
        self.stack_open_file_method = stack_open_file_method
        
        if db_type == storage_type.file_handle:
            self.handle = h5py.File(filename, 'r')
        else:
            self.handle = filename
    
    def __getitem__(self, index):
        
        if self.db_type == storage_type.file_handle:
            return self.handle[index]
        else:
            self.close_open_file()
            
            new_open_file = h5py.File(self.filename, 'r')
            self.handle = new_open_file
            self.stack_open_file_method(new_open_file)
            return new_open_file[index]
        
        

class DB_manager(object):
    
    def __init__(self, preload_to_mem, keep_memory_free, max_open_files=15000):
        
        self.preload_to_mem = preload_to_mem
        self.keep_memory_fre = keep_memory_free
        
        self.open_files = deque()
        self.max_open_files = max_open_files

    def close_open_file(self):
        if len(self.open_files) >= self.max_open_files:
            self.open_files[0].close()
            self.open_files[0].pop(0)
            
    def stack_open_file(self, db_object):
        self.open_files.append(db_object)

    def add_file(self, filename):
        print "Add {}".format(filename)
        
        if len(self.open_files) < self.max_open_files:
            db_object = DB_item(filename, storage_type.file_handle)
            self.stack_open_file(db_object)
        else:
            db_object = DB_item(filename, storage_type.string)
        
        return db_object
    
    
#         if self.preload_to_mem and self.get_free_mem() > self.keep_memory_free:
#             self.in_memory[filename] = DB_file(filename)
# 
#         
#     def get_free_mem(self):
#         meminfo = dict((i.split()[0].rstrip(':'),int(i.split()[1])) for i in open('/proc/meminfo').readlines())
#         mem_available = float(meminfo['MemAvailable'])
#         mem_total = float(meminfo['MemTotal'])
#         return (mem_available/mem_total)*100.


    
if __name__ == '__main__':
    print list(storage_type)
    
        

