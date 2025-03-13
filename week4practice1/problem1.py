#Write a program that asks a user for 3 inputs - the month, the day of the month, and the year. All inputs must be numeric. 
#Next, write a function to validate each input and another that will accept three parameters - the month, the day of the month, and the year. The function should return a string of date information of the form
#"month/day of month/year" Thus, for a month of 2, a day of the month as 22, and a year of 2050, the return should be: 2/22/2050
# Store the functions in their own file. Next, create another file that you will use to test your new functions.
# Write a test case class and create methods on it to ensure that the functions do what they are supposed to do.
# Make sure that when you write your class that you run it and your test passes.


# CODE

def get_data():
    """Get """
    
    ret_mnth = input("Please enter month: ")
    ret_day = input("Please enter the day of the month: ")
    ret_year = input("Please enter Year: ")

    x = validate(ret_mnth,ret_day,ret_year)
    if x:
        return print_date(ret_mnth,ret_day,ret_year)
    else:
        return False

def validate(mnth,day,year):
    """Validates whether or not """
    info = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    mnth = mnth.strip()
    day = day.strip()
    year = year.strip()
    
    if mnth.isnumeric() and day.isnumeric() and year.isnumeric():
        if int(mnth)<=12:
            if int(day)<=info[int(mnth)-1]:
                return True
    return False

def print_date(mnth,day,year):
    # 2/22/2050
    create_date = mnth + '/' + day + '/' + year
    return create_date

import unittest
# import sys
# sys.path.append('../functions')
# import print_date,validate

class TestMyCode(unittest.TestCase):

    def test_print(self):
        self.assertEqual(print_date('2','12','2020'), '2/12/2020')

    def test_validate(self):
        self.assertEqual(validate('2','12','2020'), True)
        self.assertEqual(validate('2','30','2020'), False)

if __name__ == '__main__':
    print(get_data())
    unittest.main()