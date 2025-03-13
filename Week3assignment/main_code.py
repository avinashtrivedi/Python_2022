import os.path
import shutil
import json
from Functions import read,save,edit_config,display

BASIC_CONFIG_PATH = 'text_files/basic_config.json'
OVERWRITTEN_CONFIG_PATH = 'text_files/config_override.json'
path = BASIC_CONFIG_PATH
        
def main():
    if os.path.exists(OVERWRITTEN_CONFIG_PATH):
        path = OVERWRITTEN_CONFIG_PATH   
    data = read.read(path)
    display.display(data)
    data = edit_config.edit_config(data)
    save.save(data,path)
    
if __name__ == '__main__':
    main()