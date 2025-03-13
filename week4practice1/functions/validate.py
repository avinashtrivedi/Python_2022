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