class add_all:
    def __init__(self):
        self.sum = 0
    
    def total(self,my_list):
        
        for i in my_list:
            self.sum = self.sum + i*i
        return self.sum
    def another_func(self):
        print('New Function')
        
    def one_more(self):
        print('One more new function')