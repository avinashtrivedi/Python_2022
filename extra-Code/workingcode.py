# See instructions for a full description of this problem
def prefix_match(dictionary, query):
    """Returns the range of words in dictionary which have a prefix matching query"""
    if len(query) == 0:
        return (0, len(dictionary))

    print(len(dictionary))
    if len(dictionary) == 0:
        return None
    
    if len(dictionary)==1 and dictionary[len(dictionary)-1][:len(query)]==query:
        return (len(dictionary)-1,len(dictionary))

    if query not in dictionary:
        print('see')
        return None

#   if(len(dictionary) == 1) and query in dictionary[][:len(query)]:
#    print('len(dic) is 1')
#     return (len(dictionary) - 1, len(dictionary))
        #right_most_prefix(dictionary
    # ,query, 0,len(dictionary)-1)
    left_prefix = left_most_prefix(dictionary, query, 0, len(dictionary) - 1)
    right_prefix = right_most_prefix(dictionary, query, left_prefix,
                                     len(dictionary) - 1)
    return (left_prefix, right_prefix)
    #, right_most_prefix


def left_most_prefix(dictionary, query, left, right):
    if left > right:
        return (0, 0)
    mid = (left + right) // 2

    if len(query) <= len(dictionary[mid]):

        if query == dictionary[mid][:len(query)] and query not in dictionary[
                mid - 1][:len(query)] or mid == 0:
            print(mid - 1, 'query is not in mid -1')
            return mid
        else:
            return left_most_prefix(dictionary, query, left, mid - 1)


def right_most_prefix(dictionary, query, left, right):
    mid = (left + right) // 2
    if left > right:
        return (0, 0)
    if len(query) <= len(dictionary[mid]):
        if query == dictionary[mid][:len(query)] and query not in dictionary[
                mid + 1][:len(query)] or mid != len(dictionary) - 1:
            return mid + 1
        else:
            return right_most_prefix(dictionary, query, mid - 1, right)


lst = ['apple', 'car', 'carsport', 'dog', 'eggs', 'fan']

test1 = [
    "apple", "apricot", "azure", "be", "bes", "best", "bestiary", "better",
    "zebra"
]
t1 = 'bes'
test2 = ["word", "zebra"]
t2 = 'a'
test3 = [
    "apple", "apricot", "azure", "be", "bes", "best", "bestiary", "better",
    "zebra"
]
t3 = 'bes'
test4 = ["apple", "apricot", "best", "zebra"]
t4 = 'apl'
test5 = ['cat']
t5 = 'c'

# print(prefix_match(test1, t1))
