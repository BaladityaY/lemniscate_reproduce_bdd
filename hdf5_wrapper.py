'''
@author: Sascha Hornauer
'''
import h5py
from enum import Enum
import h5py
from collections import deque

storage_type = Enum('storage_type', 'file_handle string')

class DB_item(object):
   
    def __init__(self, filename, db_type, close_file_method, stack_file_method):
        self.db_type = db_type
        self.filename = filename
        self.close_file = close_file_method
        self.stack_file = stack_file_method
        
        if db_type == storage_type.file_handle:
            self.handle = h5py.File(filename, 'r')
        elif db_type == storage_type.string:
            self.handle = filename
        else:
            raise TypeError('The DB item has an unknown type. This is fatal')
        
       
            
    
    def __getitem__(self, index):
        
        if self.db_type == storage_type.file_handle:
            return self.handle[index]
        elif self.db_type == storage_type.string:
            # If we need to open another file we first close a file from the open list. 
            # That file should be the one, added in the most distant past
            self.close_file()            
            new_open_file = h5py.File(self.filename, 'r')
            self.handle = new_open_file
            self.db_type = storage_type.file_handle
            self.stack_file(new_open_file)
            return new_open_file[index]
        else:
            raise TypeError('The DB item has an unknown type. This is fatal')
        
    def close(self):
        self.handle.close()
        
        

class DB_manager(object):
    
    def __init__(self, preload_to_mem, keep_memory_free, max_open_files=15000):
        
        self.preload_to_mem = preload_to_mem
        #self.keep_memory_free = keep_memory_free
        
        self.open_files = deque()
        self.max_open_files = max_open_files

    def close_open_file(self):
        if len(self.open_files) >= self.max_open_files:
            self.open_files[0].close()
            self.open_files.popleft()
            
    def stack_open_file(self, db_object):
        self.open_files.append(db_object)


    def add_file(self, filename):
        #print "Add {}".format(filename)
        
        if len(self.open_files) < self.max_open_files:
            db_item = DB_item(filename, storage_type.file_handle,self.close_open_file,self.stack_open_file)
            self.stack_open_file(db_item)
        else:
            db_item = DB_item(filename, storage_type.string,self.close_open_file,self.stack_open_file)
        
        return db_item
    
    
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
    
        

