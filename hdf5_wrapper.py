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


    def add_file(self, filename):
        print "Add {}".format(filename)
        pass
    
    def __getitem__(self,index):
        print index
        pass
    
if __name__ == '__main__':
    print list(storage_type)
    