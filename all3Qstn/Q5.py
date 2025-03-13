# read the date
def readoriginaldate():
    date = input('Enter the date string: ')
    return date

# break the date into month,date and year
def breakoriginaldate(date):
    print(f'{date} is the original date')
    
    # split the date string using '/' as separator
    m,d,y = date.split('/')
    print(f'{m} is the month {d} is the day {y} is the year')
    return m,d,y

def printdate3ways(m,d,y):
    
    # map digit to respective month
    month_dict = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',
              8:'August',9:'September',10:'October',11:'November',12:'December'}
    
    # print date in different format
    print(f'{d}-{m}-{y} is the European way of printing')
    print(f'{month_dict[m]} {d}, 20{y} is the American way of printing')
    print(f'{m:02}-{d:02}-20{y} is the full way of printing')

def main():
    
    # loop to repeat the process
    while True:
        
        # read the date
        d = readoriginaldate()
        
        # break the date
        m,d,y = breakoriginaldate(d)
        
        # convert it to integer
        m,d,y = int(m),int(d),int(y)
        
        # print in different format
        printdate3ways(m,d,y)
        print('-----------------------------------------')
        # whether do you want to stop or continue
        c = input('Do you want to continue (y/n)? ')
        if c not in ['y','Y']:
            break

main()