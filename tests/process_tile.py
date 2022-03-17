import os
import pool
import random
import string

class TileProcessor():
    
    def __init__(self, working_dir):
        
        self.working_dir = working_dir
        self.temp_dir_path = self.working_dir + '/' + self.__randomword(25)
        
        return None
    
    def run(self):
        '''
        Run the primary processing logic.
        '''
        
        # Create the temporary directory
        self.create_temporary_dir()
        
        # Delete the temporary directory
        self.__create_temporary_dir()
        
        return None
    
    def __create_temporary_dir(self):
        '''
        Create the temporary directory.
        '''
        if not os.path.exists(self.temp_dir_path):
            os.makedirs(self.temp_dir_path)
        return None

    def __delete_temporary_dir(self):
        '''
        delete the directory
        '''
        os.rmdir(self.temp_dir_path)
        return None
    
    def __randomword(length):
        '''Generate a random length string'''
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
    
if __name__ == "__main__":
    
    # Define a path tot he dataset
    test_dir = "/home/john/datasets/forecasting_disturbance/amazon_tiles/m35p5"
    
    # Instantiate the processor
    TileProcessor()
    
    
    