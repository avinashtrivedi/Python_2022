from numpy import save


i = './contact.csv'
k='contact.csv'



with open('contact.csv', 'r') as f:
    
    read_ = f.read().split('\n')

contact = []



rows = read_[1:]

banner = read_[0].split(',')
for row in rows: 
    row = row.split(',') 
    con = {} 
    for i, header in enumerate(banner):
        con[header] = row[0]
        
    for people, header in zip(row, banner):

        con[header] =  people
    con = dict(zip(banner, row))
    contact.append(con) 
    print(contact[-1])

def main():
   menu()

def menu():
    print()
    
    choice = int(input("""
                      1: Read File
                      2: Add file
                      3: Delete
                      4: Quit

                      Please choose: """))

    if choice == 1:
        name = input('enter name: ')
        for con in contact:
            if con['Name'] == name:
                for header in banner:
                    print(f'{header}:{con[header]}', )
        read_file(name)
    elif choice == 2:
        Add_File(contact)
    elif choice == 3:
                    
        delete()
        
    # elif choice == 4:

    #     Quit()
    else:
        print("goodbye")
        menu()

def read_file(name):
    
    with open(k, 'r',encoding='utf-8') as file:
        lines = file.read().split('\n')
    headers = lines[0].split(',')
    
    con = [] # return con
   
    rows = read_[1:]

    banner = read_[0].split(',')
    for row in rows: 
        row = row.split(',') 
        con = {} 
        for i, header in enumerate(banner):
            con[header] = row[0]
            
        for people, header in zip(row, banner):

            con[header] =  people
        con = dict(zip(banner, row))
        contact.append(con) 
             
    
def Add_File(contact):

    
     

    # contact = []
      
    # Fruits=input('Please enter fruit: ')
    # contact.append(Fruits)    
    # colors=input('Please enter color: ')
    # contact.append(colors)


    # with open('contact.csv', 'r') as f:
    
    #         read_ = f.read().split('\n')

    

    # reader_obj = csv.DictReader(f)
    # # ames = reader_obj.fieldnames
    # # print(ames)
    # for row in reader_obj:
    #     print(ro
        Names=input('Please enter contact info: ')
        Fruit=input('Please enter contact fruit: ')
        color=input('Please enter contact color: ')
        diction = {'Name': Names,'Fruit': Fruit,'color': color}

           
        contact.append(diction)
        print(diction)
        
        print(contact[-1])
              
        
def delete():

    Name = input('enter nam')
    with open('contact.csv', 'r') as readFile:

        reader = csv.reader(readFile)
        for row in reader:
            contact.append(row)
            for field in row:
                if field == Name:
                    contact.remove(row)

                         
def update():
    con = []  
    with open(k, 'r',encoding='utf-8') as file:
        lines = file.read().split('\n')
    
    headers = lines[0].split(',')

    for i in range(1,len(lines)):

        #iterate over the people
        #make a dictionary for each person
        contact = {}
        for h in range(len(headers)):

            person=lines[i].split(',')
            contact[headers[h]] = person[h]
    con.append(contact)
    contact.update()
   
   
main()
 