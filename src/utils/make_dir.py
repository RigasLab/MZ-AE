import os

def mkdirs(directories):
    '''
    Makes a directory if it does not exist
    '''
    for directory in directories:
        try:
            os.mkdir(directory)
        except Exception as e: print(e)