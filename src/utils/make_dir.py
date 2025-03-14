import os

def mkdirs(directories):
    for directory in directories:
        try:
            if not os.path.exists(directory):
                print("CREATING DIR: ", directory)
                os.makedirs(directory, exist_ok=True)
            else:
                print("ALREADY EXISTS DIR: ", directory)
                
        except Exception as e:
            print(e)