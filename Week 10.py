#from Classes.pet import Pet


#For this assignment, you need to submit a Python program that gathers information about instructors and students according to the rules provided:
#What type of individual we're dealing with (instructor or student)
#If the individual is a student, we need to get their Student ID (this is required, and must be a number that is 7 or less digits long) 
#We also need to obtain their program of study (this is required)
#If the individual is an instructor, we need to get their Instructor ID (this is required, and must be a number that is 5 or less digits long) 
#We also need to obtain the name of the last institution they graduated from (this is required) and highest degree earned (this is required)
#The individual's name (this is required, and must be primarily comprised of upper- and lower-case letters. It also cannot contain any of the following characters: ! " @ # $ % ^ & * ( ) _ = + , < > / ? ; : [ ] { } \ ).
#The individual's email address (this is required, and must be primarily comprised of alphanumeric characters. It also cannot contain any of the following characters: ! " ' # $ % ^ & * ( )  = + , < > / ? ; : [ ] { } \ ).
#Write your program such that if a user inputs any of the above information incorrectly, the program asks the user to re-enter it before continuing on. Your program should accept any number of individuals until the person using the program says they are done.

#You will store the information you collect for each user in class instances that you will append to a list called "college_records". There will be an Instructor class
#and a Student Class (which should be pretty simple), and your submission must use inheritance principles.
#Create a method that displays all collected information for an individual called "displayInformation".
#You must also create a class called "Validator" that has methods attached to it that validate submitted information.
#When your submission is through gathering data, print out all entries in the college_records list.

programcycle = True
college_records = []

class validator():
    """ This class holds and estbalishes validations for the program  """

    def __init__(self,id_num=0,first_name="",last_name="",email="", programofstudy="", degree="", last_instit=""):
        self.id_num = id_num
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.programofstudy = programofstudy
        self.degree = degree
        self.last_instit = last_instit
#highest degree processing
    def validate_degree(self):

        ''' This is a method to validate the highest degree earned '''

        if self.degree.isalpha():
            college_records.append(self.degree)
            return False
            
        else:
            print("Please try again.")        
            return True
#email processing   
    def validate_email(self):

        ''' This is a method to validate the email address '''

        invalid_characters = ["!", "\"", "'", "#", "$", "%", "^", "&", "*", "(", ")",  "=", "+", ",", "<", ">", "/", "?", ";", ":", "[", "]", "{", "}", "\\"]

        if self.email:
            for character in self.email:
                if character in invalid_characters:
                    print("Invalid character try again")
                    return True
        if self.email.isalnum:
            college_records.append(self.email)  

        else:
            print("Please try again.")        
            return True         
#first name processing
    def validate_name(self):

        ''' This is a method to validate the program of study '''
        
        invalid_characters = ["!", "@", "\"", "'", "#", "$", "%", "^", "&", "*", "(", ")",  "=", "+", ",", "<", ">", "/", "?", ";", ":", "[", "]", "{", "}", "\\"]

        if self.first_name:
            for character in self.first_name:
                if character in invalid_characters:
                    print("Invalid character try again")
                    return True

        if self.first_name.istitle():
            college_records.append(self.first_name)
            return False
            
        else:
            print("Please try again.")        
            return True

# Id number validation processing           
    def validate_id_num(self, person_type):

        if person_type == "student" and self.id_num.isdigit() and len(self.id_num) <= 7:
            college_records.append(self.id_num)
            print(college_records)
            return False
  
        elif person_type == "instructor" and self.id_num.isdigit() and len(self.id_num) <= 5:
            college_records.append(self.id_num)
            print(college_records)
            return False

        else:
            print("Error. Try again.")
            return True
# last name validation processing
    def validate_last_name(self):

        ''' This is a method to validate the program of study '''
        
        invalid_characters = ["!", "@", "\"", "'", "#", "$", "%", "^", "&", "*", "(", ")",  "=", "+", ",", "<", ">", "/", "?", ";", ":", "[", "]", "{", "}", "\\"]

        if self.last_name:
            for character in self.last_name:
                if character in invalid_characters:
                    print("Invalid character try again")
                    return True

        if self.last_name.istitle():
            college_records.append(self.last_name)
            return False
            
        else:
            print("Please try again.")        
            return True
# validation of last institute an instructor graduated from
    def validate_last_instit(self):

        ''' This is a method to validate the program of study '''

        if self.last_instit.isalpha():
            college_records.append(self.last_instit)
            return False
            
        else:
            print("Please try again.")        
            return True
# validation of the program of study for students
    def validate_programofstudy(self):

        ''' This is a method to validate the program of study '''

        if self.programofstudy.isalpha():
            college_records.append(self.programofstudy)
            return False
            
        else:
            print("Please try again.")        
            return True
  
class person(validator):
    """The person class creates parameters for users """
    def __init__(self,id_num=0,first_name="",last_name="",email=""):
        self.id_num = id_num
        self.first_name = first_name
        self.last_name = last_name
        self.email = email

    def display_information(self):

        print(college_records)


class student(person):
    """ The student class inherits from the person class sharing attributes """
    def __init__(self,id_num=0,first_name="",last_name="",email="",programofstudy=""):
        self.id_num = id_num
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.programofstudy = programofstudy
      
               
class instructor(person):
    """ The instructor class also inherits from the person class """
    def __init__(self,id_num=0,first_name="",last_name="",email="",degree="",last_instit=""):
        self.id_num = id_num
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.degree = degree
        self.last_instit = last_instit
      
          
st = student()
ins = instructor()
# Main Program cycle begins
while programcycle:
    
    userinfo = input("Are you a student or instructor? ")
    
    if userinfo == "student":
        person_type = "student"
        
        id_num_check = True
        while id_num_check:
            st.id_num = input("Please enter your student id. ")
            id_num_check = st.validate_id_num(person_type)

        first_name_check = True
        while first_name_check:
            st.first_name = input("Enter your first name ")
            first_name_check = st.validate_name()

        last_name_check = True
        while last_name_check:
            st.last_name = input("Enter your last name ")
            last_name_check = st.validate_last_name()
 

        email_check = True
        while email_check:
            st.email = input("Please enter your email address. ")
            email_check = st.validate_email()
            

        programofstudy_check = True
        while programofstudy_check:
            st.programofstudy = input("Please enter your program of study. ")
            programofstudy_check = st.validate_programofstudy()

        
    elif userinfo == "instructor":
        person_type = "instructor"

        id_num_check = True
        while id_num_check:
            ins.id_num = input("Please enter your Instructor id. ")
            id_num_check = ins.validate_id_num(person_type)

        email_check = True
        while email_check:
            ins.email = input("Please enter your email address. ")
            email_check = ins.validate_email()

        first_name_check = True
        while first_name_check:
            ins.first_name = input("Enter your first name ")
            first_name_check = ins.validate_name()

        last_name_check = True
        while last_name_check:
            ins.last_name = input("Enter your last name ")
            last_name_check = ins.validate_last_name()

        last_instit_check = True
        while last_instit_check:
            ins.last_instit = input("Please enter the last institution you graduated from. ")
            last_instit_check = ins.validate_last_instit()

        degree_check = True
        while degree_check:
            ins.degree = input("Please enter your highest degree earned. ")
            degree_check = ins.validate_degree()
        
    
    
    else:
        print("Invalid response")   
# Asks the user if they wish to continue on an onput more individuals into the system.  
    program_check = True
    while program_check:
        continue_program = input("Do you want to continue the program? ")
        if continue_program.upper() =="Y":
            programcycle = True
            program_check = False
        
        elif continue_program.upper() =="N":
            programcycle = False
            program_check = False
              
            if userinfo == "student":
                st.display_information()

            elif userinfo == "instructor":
                ins.display_information()

        else:
            print("Error. Enter y/n to continue. ")
            program_check = True
