import os

def mkdirs(directories):
    '''
    Makes a directory if it does not exist
    '''
    for directory in directories:
        try:
            if not os.path.exists(directory):
                print("CREATING DIR: ", directory)
                os.mkdir(directory)
            else:
                print("ALREADY EXISTS DIR: ", directory)
                
        except Exception as e: print(e)