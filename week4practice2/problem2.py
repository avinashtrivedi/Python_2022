#Write a class called Task. The constructor for the class should take in a task name, task description, and the default time it took to complete the task
#Each of these values should be stored as an attribute on the class
# Write a method called "increase_time" that takes the time property and increases it by one by default. Code your method so that this default can be overridden.
# Next, write a method called "decrease_time" that takes the time property and reduces it by one by default. Also, code your method so that this default can be overridden.
# Lastly, create a method called "reset_time" that sets the time value back to 0.
# Now, write a test case class for your Task class. Write test methods to test your increase_time and decrease time methods.
# You should test BOTH the default behavior AND what happens when the default behavior is overridden. Also, you need to test the reset_time method.
# In your test class, leverage the setUp() method technique described in the textbook so you only have to instantiate your Task class once.
# Lastly, run your test case class and ensure that all tests pass



#set up example

#def setup(self):
    #"""Get """

    #question = "what language"
    #self.my_survey = AnonymousSurvey(question)
    #self.responses = ['English', 'Spanish', 'Mandarin']

#def test_store_single_response(self):
    #"""Get """

    #self.my_survey.store_response(self.responses[0])
    #self.assertIn(self.responses[0], self.my_survey.responses)

#def test_store_three_reponses(self):
    #"""Get """

    #for response in self.responses:
        #self.my_survey.store_responses(response)
    #for response in self.responses:
        #self.assertIn(response, self.my_survey.responses)

#if __name__ '__main__':
    #unittest.main()

import unittest
class Task:
    
    def __init__(self,name,desc,default_time=10):
        self.name = name
        self.desc = desc
        self.default_time = default_time   
        
    def increase_time(self):
        self.default_time = self.default_time + 1
         
    def decrease_time(self):
        self.default_time = self.default_time - 1
       
    def reset_time(self):
        self.default_time = 0  