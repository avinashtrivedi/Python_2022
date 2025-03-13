class NAMECHECK:
    
    def __init__(self):
        self.allowed_char = ['-',"'"] 
        
    def NameCheck(self,name,first_last):

        while True: 
            flag = False
            if name.istitle():
                for char in name:
                    if char.isalpha() or char in self.allowed_char:
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
            