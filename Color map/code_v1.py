def find_winner(record, find_max=True):
    """
    
    Returns the name of the country with either largest or
    smallest score, depending on the optional parameter.

    >>> find_winner([('Germany',23),('USA', 49), ('South Korea', 32)])
    'USA'
    >>> find_winner([('China', 12.88),('Japan', 15)], find_max=False)
    'China'
    >>> find_winner([('France', 10), ('UK', 10), ('Spain', 5)], find_max=True)
    'France'
    """
    return find_winner_helper(record, find_max)[0]
    
    # somewhere here you may want to use a call to a helper recursive function:
    # find_winner_helper(record, find_max)



# Recursive function. Think, what does it return?
def find_winner_helper(record, find_max):
    # add your own doct tests to check correctness. 
    if find_max:
        if len(record) == 1:
            return record[0]
        else:
            val = find_winner_helper(record[1:],find_max)
            int_val = val[1]
            state = val[0]
            return val if int_val > record[0][1] else record[0]
    else:
        if len(record) == 1:
            return record[0]
        else:
            val = find_winner_helper(record[1:],find_max)
            int_val = val[1]
            state = val[0]
            return val if int_val < record[0][1] else record[0]
