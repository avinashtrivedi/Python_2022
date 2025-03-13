def abbreviate(a_string):
    t = ''
    for i in range(len(a_string)):
        if a_string[i] in 'aeiouAEIOU' and a_string[i-1]!=' ':
            pass
        else:
            t = t + a_string[i]
    return t

def main():
    print(abbreviate("Desirable unfurnished flat in quiet residential area"))
    print(abbreviate("Hi There"))
    
main()