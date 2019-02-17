'''
@author: Sascha Hornauer
'''

class DB_manager(object):
    
    
    
    def __init__(self, preload_to_mem, keep_memory_free):
        
        self.preload_to_mem = preload_to_mem
        self.keep_memory_fre = keep_memory_free



    def add_file(self, filename):
        print "Add {}".format(filename)
        pass
    
    def __getitem__(self,index):
        print index
        pass