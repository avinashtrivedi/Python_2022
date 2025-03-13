import print_date
import validate

def get_data():
    """Get """
    
    ret_mnth = input("Please enter month: ")
    ret_day = input("Please enter the day of the month: ")
    ret_year = input("Please enter Year: ")

    x = validate.validate(ret_mnth,ret_day,ret_year)
    if x:
        return print_date.print_date(ret_mnth,ret_day,ret_year)
    else:
        return False