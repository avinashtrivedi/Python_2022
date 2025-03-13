def display(data): 
    i = 1
    for k,v in data.items():
        print(f'#{i}. {k} - "{v}"')
        i = i + 1