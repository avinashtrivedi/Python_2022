with open('weblog.txt') as fp:
    data = fp.readlines()
    
new_data = [line for line in data if 20220101<=int(line.split('|')[0])<=20220115]

with open('weblog2022a.txt','w') as fp:
    for line in new_data:
        fp.write(line)