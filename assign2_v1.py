def read():
    print("Enter names, one on each line. Type DONE to quit entering names.")
    names = []
    while True:
        name = input()
        if name=='DONE':
            break
        names.append(name)
    return names

def step_3(s):

    D = ''
    for i in s:
        for j in range(len(d)):
            if i in d[j]:
                D = D + str(j)
    return D

def step_4(D):
    x = ''
    p = ''
    for i in D:
        if i!=p:
            x = x + i
            p = i
    
    return x

def step_56(x,F):
    D = x.replace('0','')
    for j in range(len(d)):
        if F in d[j]:
            F_digit = str(j)

    if len(D) == 0:
        D = F
    elif F_digit == D[0]:
        D = F + D[1:]
    elif F_digit != D[0]:
        D = F + D
    return D

def step_7(D):
    if len(D)>4:
        D = D[:4]
    elif len(D)<4:
        D = D + '0'*(4-len(D))
        
    return D

def soundex(s):
    s = s.lower()
    F = s[0]
    D = step_3(s)
    D = step_4(D)
    D = step_56(D,F)
    D = step_7(D)
    return D

def main():

    names = read()
    sounding_list = []
    for name in names:
        sounding_list.append((soundex(name),name))

    output = []
    for i in range(len(sounding_list)):

        sname1 = sounding_list[i][0]
        name1 = sounding_list[i][1]

        for j in range(i+1,len(sounding_list)):

            sname2 = sounding_list[j][0]
            name2 = sounding_list[j][1]

            if name1 != name2:
                if sname1==sname2:
                    name1 , name2 = sorted([name1 , name2])
                    output.append(f"{name1} and {name2} have the same Soundex encoding")
    output.sort()
    if len(output)!=0:
        print(*output,sep='\n')
    

if __name__ == "__main__":
    
    d = [['a', 'e','i', 'o', 'u', 'y', 'h', 'w'],
            ['b', 'f', 'p', 'v'],
            ['c', 'g', 'j', 'k', 'q', 's', 'x', 'z'],
            ['d', 't'],
            ['l'] ,
            ['m', 'n'],
            ['r'] ]
    
    main()