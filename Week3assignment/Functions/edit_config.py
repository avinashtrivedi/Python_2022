from Functions import display as dp
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
            dp.display(data)
            choice = input('Do you want to continue Add/Modify/Delete y/n: ')
            if choice.lower() == 'y':
                pass
            elif choice.lower() == 'n':
                return data
            else:
                print('Wrong input,Please enter y/n.')
    except:
        print('Wrong input')