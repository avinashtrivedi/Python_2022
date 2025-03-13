class Task:
    
    def __init__(self,name,desc,default_time=10):
        self.name = name
        self.desc = desc
        self.default_time = default_time
      
    def increase_time(self):
        self.default_time = self.default_time + 1
#         return self.default_time
        
    def decrease_time(self):
        self.default_time = self.default_time - 1
#         return self.default_time
        
    def reset_time(self):
        self.default_time = 0
#         return self.default_time