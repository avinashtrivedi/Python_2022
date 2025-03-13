import random
import string

for i in range(20):
    d = random.randint(100, 999)
    s = ''.join(random.sample(string.ascii_uppercase,3))
    
    print(d,s)