import os.path
import shutil
import json

def read(path):
    try:
        with open(path) as fp:
            data = json.load(fp)
        return data
    except:
        print('Wrong path',path)
        return False

def display(data): 
    i = 1
    for k,v in data.items():
        print(f'#{i}. {k} - "{v}"')
        i = i + 1
        
def edit_config(data):
    try:
        while True:
            print('\nMake your selection what you want to do')
            print('#1 Add')
            print('#2 Modify')
            print('#3 Delete')
            c = int(input('Type your choice (in numeric): '))

            if c == 2:
                c_modify = int(input('Enter the sequence no to modify: '))
                val = input('Enter the new value: ')

                if c_modify <= len(data):

                    var = list(data.items())[c_modify-1][0]
                    data[var] = val

                else:
                    print(f'Wrong sequence, last sequence is {len(data)}')
            elif c==1:
                var = input('Enter a variable name: ')
                val = input('Enter the value: ')

                data[var] = val
            elif c==3:
                c_modify = int(input('Enter the sequence no to Delete: '))

                if c_modify >= 4 and c_modify <= len(data):
                    var = list(data.items())[c_modify-1][0]
                    del data[var]
                    print(f'Entry for Sequence {c_modify} deleted')
                else:
                    print(f'Wrong sequence')

            print('\nThe final configurations are:')
            display(data)
            choice = input('Do you want to continue Add/Modify/Delete y/n: ')
            if choice.lower() == 'y':
                pass
            elif choice.lower() == 'n':
                return data
            else:
                print('Wrong input,Please enter y/n.')
    except:
        print('Wrong input')
        
def save(data,path):
    try:
        print('\nThe final configurations are:')
        display(data)
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