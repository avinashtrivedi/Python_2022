print('-----------------------------')
print('     Binary to Decimal       ')
print('-----------------------------')
print('Enter a 4 bit binary number')
print('\n')

binary = input('Binary: ')

d2b = 0
n = len(binary) - 1
print('\n')
for i in range(len(binary)):
    print("{:^10s}".format(binary[i]),end=' ')
print()


for i in range(n,-1,-1):
    d2b = d2b + int(binary[n-i]) * 2**i
    x = '2^' + str(i)
    print("{:^10s}".format(x),end=' ')
    
#     print("2^{:<10}".format(i),end=' ')
print('\n')
print('Decimal:',d2b)