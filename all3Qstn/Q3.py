def main():
    introduction()
    count = 0
    
    # loop to receive multiple set of data
    while True:
        
        # receive and process the data
        data = input('The three original integers are: ')
        data = [int(i) for i in data.split()]

        # find the average
        avg = findaverage(data[0],data[1],data[2])
        print(f'The average is {avg:.3f}')
        
        # compare the data with average
        comparetoavg(data[0],data[1],data[2],avg)
        
        # whether do you want to stop or continue
        c = input('Do you want to continue (y/n)? ')
        count  = count + 1 
        if c not in ['y','Y']:
            break
    print(f'{count} sets of three data values were entered and processed')

# program header
def introduction():
    print('Code Developed by YourName')
    print('Program to find average and do comparison')
    print('-------------------------------------------')

# compute the average
def findaverage(x,y,z):
    avg = (x+y+z)/3
    return avg

# compare the numbers with average
def comparetoavg(x,y,z,avg):
    count = 0
    
    # loop to iterate through each numbers
    for i in [x,y,z]:
        if i>avg:
            print(f'{i} is above the average')
        elif i == avg:
            count = count + 1
            print(f'{i} is equal to the average')
        else:
            print(f'{i} is below the average')
            
    print(f'{count} values equal to the average')

# call the main function.
main()