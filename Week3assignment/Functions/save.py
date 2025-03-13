import os.path
import shutil
import json
from Functions import display as dp

def save(data,path):
    try:
        print('\nThe final configurations are:')
        dp.display(data)
        save_discard = input('Enter 1 to save or any other number to discard: ')
        if save_discard == '1':

            shutil.copy(path,path.split('.')[0] + '_backup.json')
            print('Backup is created.')

            with open('text_files/config_override.json', 'w') as fp:
                json.dump(data, fp)

            print('Configurations has been saved Successfully as config_override.json')
        else:
            print('Configurations has been discarded.')
    except:
        print('Unable to save.')