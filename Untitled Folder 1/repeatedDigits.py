def repeated(num):
    num = str(num)
    
    if '-' in num:
        num = num[1:]
        
    for i in range(len(num)):
        if num[i] == '0':
            continue
        else:
            break
    num = num[i:]

    lst = []
    for i in ('0','1','2','3','4','5','6','7','8','9'):
        lst.append(num.count(i))
    return lst