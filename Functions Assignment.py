#Please note that you will need to apply knowledge from prior weeks in order to complete this assignment!
#For this assignment, you need to modify the program you created in the While Loops Assignment and reorganize your submission to eliminate redundant code. promote code re-use, and use functions to gather each piece of data.
#Be sure to put comments in your code that clearly mark how you are performing your program logic.
list_of_dictionaries = []
counter = 0
# The counter is set to 0 to start with. It will increase by 1 when a user cycles through all the inputs.
# A user can choose not to enter more information for additional users if they enter skip during the address prompt at the end.
# Doing this will print out what has been given regarding user information.
while counter < 5:
    Employee_ID = input("Please enter your Employee ID number:")
    ID_OK = False
    while not ID_OK:
        if Employee_ID.isnumeric:
            try:
                int(Employee_ID)
                if len(Employee_ID) <= 7:
                    print("Your Employee number is: " + Employee_ID)
                    ID_OK = True
                    list_of_dictionaries.append('User :')
                    list_of_dictionaries.append(" Employee ID number: " + Employee_ID)
                else:
                    ID_OK = False   
            except:
                ID_OK = False
            if not ID_OK:
                Employee_ID = input("Employee ID was not properly formatted. Please try again:")    

    #end employee ID processing
    # Begin employee first name processing
    invalid = ["*","!", "'", "@", "$", "%", "^", "&", "*", "_", "=", "+", "<", ">", "?", ";", ":", "[", "]", "{", "}", ")"]
    Employee_first_name = input("Please enter your first name:")
    first_name_OK = False 
    while not first_name_OK:
        if Employee_first_name:
            try:
                if Employee_first_name.istitle():
                    first_name_OK = True
                    
                
                else:
                    if Employee_first_name.isnumeric():
                        first_name_OK = False
                for character in invalid:
                    if character in Employee_first_name:
                        first_name_OK = False
                        
                    
            except:
                first_name_OK = False
            if not first_name_OK:
                Employee_first_name = input("First name was not properly formatted. Please try again:")
    list_of_dictionaries.append(Employee_first_name)    

    invalid = ["*","!", "'", "@", "$", "%", "^", "&", "*", "_", "=", "+", "<", ">", "?", ";", ":", "[", "]", "{", "}", ")"]
    Employee_last_name = input("Please enter your last name:")
    last_name_OK = False 
    while not last_name_OK:
        if Employee_last_name:
            try:
                if Employee_last_name.istitle():
                    last_name_OK = True
                    
                else:
                    last_name_OK = False
                for character in invalid:
                    if character in Employee_last_name:
                        last_name_OK = False
                    
            except:
                last_name_OK = False
            if not last_name_OK:
                Employee_last_name = input("Last name was not properly formatted. Please try again:")
    list_of_dictionaries.append(Employee_last_name)
    #end last name processing
    #begin email address processing
    invalid_email_character = ["*","!", "'", "$", "%", "^", "&", "*", "_", "=", "+", "<", ">", "?", ";", ":", "[", "]", "{", "}", ")"]
    Employee_email = input("Please enter your email address:")
    Employee_email_ok = False

    while not Employee_email_ok:
        if Employee_email:
            try:
                if Employee_email:
                    Employee_email_ok = True
                    
                
                else:
                    
                    Employee_email = False
                for character in invalid_email_character:
                    if character in Employee_email:
                        Employee_email = False
                               
            except:
                Employee_email_ok = False
            if not Employee_email_ok:
                Employee_email = input("Email was not properly formatted. Please try again:")
    if Employee_email_ok:
        list_of_dictionaries.append(Employee_email)
#end employee email

    #Begin employee address processing
    invalid_address = ["*","!", "'", "$", "%", "^", "&", "*", "_", "=", "+", "<", ">", "?", ";", ":", "[", "]", "{", "}", ")"]
    Employee_address = input("This is not required you may enter skip to skip over this step. Please enter your home address if you would like.")
    address_ok = False

    while not Employee_address:
        if Employee_address:
            try:
                if Employee_address.isalnum():
                    address_ok = True
                    
                else:
                    address_ok = False
                for character in invalid_address:
                    if character in Employee_address:
                        address_ok = False

            except:
                address_ok = False
            if not address_ok:
                Employee_address = input("Your home address was not entered correctly please try again.")
               
    if Employee_address == "skip":
            print("Hello " + Employee_first_name + Employee_last_name + " " "You did not provide an address. ")
            list_of_dictionaries.append("No address was provided")
            counter += 5
            # Entering skip will exit out of the loop and print what the user has entered so far. This will also enter no address for the person's address.
            
    else:
            print("Hello " + Employee_first_name +" " + Employee_last_name + " Your address is " + Employee_address )
            list_of_dictionaries.append("Home address: " +Employee_address)
            counter += 1
        #This will increase the counter by 1 when a compelete set of information has been submitted. It will then loop back up to the first part and repeat.   
if counter >= 5:
    print(list_of_dictionaries)    
    
