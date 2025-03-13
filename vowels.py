def vowel_count(text):
    text = text.lower()
    
    lst = []
    for i in ('a','e','i','o','u'):
        lst.append(text.count(i))
    return lst

def main():
    print(vowel_count ("Hello There"))
    print(vowel_count ("This is in python"))
    print(vowel_count ("count the number of vowels"))
    
main()