allowed_char = ['-',"'"] 

def NameCheck(name,first_last):

    while True: 
        flag = False
        if name.istitle():
            for char in name:
                if char.isalpha() or char in allowed_char:
                    continue
                else:
                    flag = True
                    break
            if flag:
                name = input(f"{first_last} name was not properly formatted. Please try again:")
            else:
                return name
        else:
            name = input(f"{first_last} name was not properly formatted. Please try again:")
