def wallis(num):
    pi = 2.
    lst = []
    for i in range(1, num+1):
        left = (2. * i)/(2. * i - 1.)
        right = (2. * i)/(2. * i + 1.)
        lst.append(left*right)
        
    for i in lst:
        pi = pi*i
    return pi

print(wallis(2))
print(wallis(3))