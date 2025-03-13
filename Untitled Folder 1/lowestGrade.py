def remove_lowest(lst):
    if len(lst)>1:
        minm = min(lst)
        lst.remove(minm)
        return lst
    else:
        return lst

a = remove_lowest ( [23, 90, 47, 55, 88] )
b = remove_lowest ( [85] )
c = remove_lowest ( [] )
d = remove_lowest ( [59, 92, 93, 47, 88, 47] )
print ("a =", a)
print ("b =", b)
print ("c =", c)
print ("d =", d)