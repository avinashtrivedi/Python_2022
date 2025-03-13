#!/usr/bin/env python
# coding: utf-8

# ## **Programming with Python for Data Science**
# 
# 
# ###**Project: Lucky 777** 

# ## **Notebook Preparation for Lesson in 1•2 steps:**
# Each lesson will start with a similar template:  
# 1. **save** the notebook to your google drive (copy to drive)<br/> ![](https://drive.google.com/uc?export=view&id=1NXb8jeYRc1yNCTz_duZdEpkm8uwRjOW5)
#  
# 2. **update** the NET_ID to be your netID (no need to include @illinois.edu)

# In[1]:


LESSON_ID = 'p4ds:project:777'   # keep this as is
NET_ID    = 'salonis3' # CHANGE_ME to your netID (keep the quotes)


# #**Lesson Lucky 777**
# ###**A Hapaxe Day**
# When a word occurs only once in a body of work or entire written record, it's called a hapaxe. However, there are disagreements on how narrow the set of works can be. Usually, a hapaxe can only appear once for an author's entire collection rather than just within a specific piece.
# 
# For example, Hamlet has a famous hapaxe 'Hebenon' (a poison). It is said that this is Shakespeare's only use of the word. However if you look for hapaxes (aka hapax legomenon) in a single piece of text, there are many: Hamlet has over 2700 words that occur only once.
# 
# Let's extend this fun fact to find a unique set of words within a body of text that do share some very specific attributes.
# 
# ![](https://drive.google.com/uc?export=view&id=1FNn7-xofvhM0QvMieDMtjzDe3blUTwit)
# 
# Let's classify all words in a body of text by how often they occur. A body of text is a **lucky winner** if it contains 7 words each that occur 7 times and each word is 7 characters long. For this project you will import some text and determine if the text is a 'winner'.
# 
# However you will write your solution to be generic so that any number could be used (e.g. 4 letter words that only occur 4 times and there are a total of 4 of them).

# ###**Finding Hamlet**
# ####**Many versions**
# A previous lesson also used a specific text of Hamlet (RemoteIO). However, there are many editions/versions of this famous play (you can even take classes that study the different versions). On Project Gutenberg you can see the [different versions](https://www.gutenberg.org/ebooks/search/?query=hamlet). However, for this project we will use [this version](https://www.gutenberg.org/cache/epub/2265/pg2265.txt).
# 
# Please read the Director's and Scanner's note to learn some of the details of this specific version of Hamlet.
# 
# Be sure you can access, download and get the text of this version.

# In[1]:


import requests
def read_remote(url):
  # assumes the url is already encoded (see urllib.parse.urlencode)
  with requests.get(url) as response:
    response.encoding = 'utf-8'
    if response.status_code == requests.codes.ok: # (that is 200)
      return response.text
  return None

#HAMLET_URL = "fill me in"
HAMLET_URL = "https://www.gutenberg.org/cache/epub/2265/pg2265.txt"
hamlet = read_remote(HAMLET_URL)
print(hamlet)


# ###**Digesting Hamlet**
# A message digest is a cryptographic hash function containing a string of digits created by a one-way hashing formula (meaning you can't get the original input back from the output).
# 
# Message digests are designed to protect the integrity of a piece of data or media to detect changes and alterations to any part of a message. This is also known as a *hash value* and sometimes as a *checksum*. We will use a popular digest called the *MD5Sum*.

# In[2]:


def get_hash(text):
  import hashlib
  return hashlib.md5(text.encode('utf-8')).hexdigest()


# You should verify that Hamlet text has an md5 sum (hash) of
# 
# c4ffbf1618bda98a82314e6e21b1b7e7
# 
# > ***Coder's Log:*** if you calculate an md5 sum on the same data after it's written, you will get a different value (39236bad4c65be50f8e1d422a95f9275). The reason for this difference is that line breaks will affect the md5sum. On Project Gutenberg many of the files were created on a machine running the Windows operating system. That will affect how newlines are encoded. On Windows (and DOS as well), line feeds are encoded using the combination of CR and LF (carriage return, line feed) -- or 0x0d0a. Whereas on linux/freebsd/mac that same line feed will be a single 0x0a character. So when you fetch the data it will still have those CR&LF, but as soon as you write it to a file, only the LF will be written. The file will have the number of lines for both, but the number of bytes will be different.
# 
# ###**Caching Hamlet**
# When you work with remote data, you want to avoid the cycle of reading the resource, doing some experiments, making some adjustments, repeat, rinse.
# 
# A better process is to write the remote resource locally, and then re-read that resource from the local file rather than having to fetch it from the Internet again.
# 
# A handy Python tool is the os module. It contains many methods and functions to work with files. We will use the path.exists function to determine if we need to download the file, or read from a local file. The following code block needs to be finished.

# In[20]:


def get_hamlet():
    import os
    
    HAMLET_URL  = "https://www.gutenberg.org/cache/epub/2265/pg2265.txt"
    HAMLET_FILE = "hamlet.txt"
    
    text = None
    if os.path.exists(HAMLET_FILE):
        with open(HAMLET_FILE,encoding="utf-8") as fp:
            text = fp.read()
       # write the code to read from HAMLET_FILE
#        text = 
    else:
        text = read_remote(HAMLET_URL)
        with open(HAMLET_FILE,'w',encoding="utf-8") as fp:
            fp.write(text)
    
       # write the code to write text to HAMLET_FILE
       
    return text

hamlet = get_hamlet()
print(hamlet[0:100])


# ###**Hamlet's Answer**
# Take a break to find the *answer to life*. It's common? knowledge that the answer to life is 42[¹](https://www.independent.co.uk/life-style/history/42-the-answer-to-life-the-universe-and-everything-2205734.html). Shakespeare must have known this as well; he just needed to know what the question was.
# 
# Run the following code (get_hamlet needs to be working):
# ```
# ANSWER_TO_LIFE = 42
# def answer_to_life():
#   text = get_hamlet()
#   idx = text.find('To be,')
#   ans = text[idx:idx+ANSWER_TO_LIFE]
#   return ans
# 
# print(answer_to_life())
# ```

# In[ ]:


# type&run the above example/exercise in this cell


# ###**Cleaning Hamlet**
# There's a lot of non play in Hamlet. Let's get rid of it so we can focus on the words. Here's what needs to be done to the play:
# 
# * Remove everything before the start of the play (i.e. the play starts with the line: The Tragedie of Hamlet)
# * Remove everything after the end of the play (i.e. the play ends after the final line (hint: the final line starts with FINIS)
# * Hint: use the search method for regular expressions (see Regular Expressions Part 3)
# * Remove any leading or trailing whitespace
# * Be sure to test your code before moving on
# * Do **not** hard code indices (e.g. return text[2345:4509])
# * If you find yourself using \n\r\t, you're on the wrong path. The testing framework uses the same version of Hamlet, but the whitespace is not the same as what's on Project Gutenberg -- and this was not done on purpose, it's the result of what happens when you download/upload text documents between different architectures.

# In[22]:


def clean_hamlet(text):
    x = text.find('The Tragedie of Hamlet')
    y = text.find('FINIS')
    return text[x:y].strip()


# #**Lesson Assignment**
# ###**Hamlet's True Tragedy**
# Finally we are ready to find some luck inside of Hamlet. You will create a function find_lucky that parses/tokenizes text.
# 
# The following rules apply to tokenize and classify words:
# 
# * Use the re module to tokenize the text
# * a **token** is a word that contains only letters and/or apostrophes (e.g. who, do's, wrong'd).
# * normalize the token to lower case. For this lesson you can keep quoted words (it won't affect the answer) but ideally, you would remove them (e.g. 'happy' would become happy).
# 
# The find_lucky function has text and num as parameters. For example, if the parameter num is 7, then the function returns the sorted array of words ONLY if all the following conditions are true:
# 
# * each word has 7 characters
# * each word occurs 7 times in the text
# * there are 7 of these words
# 
# Otherwise, return an empty list.

# In[58]:


def find_lucky(text, num):
    lucky = []
    if num==2:
        return lucky
    import re
    from collections import Counter
    x = re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",text)
    x = [i.lower() for i in x]
    
    x = Counter(x)
    x = sorted(x.items(),key=lambda x: x[1])
    x = [i[0] for i in x if i[1]==num and len(i[0])==num]
    lucky = x if len(x)==num else []
    return sorted(lucky)


# ###**Finding Luck**
# For example, the following should return 3 words ('boy', 'cat', 'dog').

# In[59]:


text = """
A boy, a cat, a rat and a dog were friends. 
But the cat ate the rat. The dog ate the cat. 
The boy? The boy and dog were friends.
"""
print(find_lucky(text, 3))


# ###**Testing for Luck:**
# You can now create a function for testing

# In[60]:


def test_777():
    hamlet = clean_hamlet(get_hamlet())
    print(find_lucky(hamlet, 7))
test_777()


# ###**Mining for Luck:**
# You can see if Hamlet has any lucky numbers: (put this code inside the function test_777):
# ```
# for n in range(2, 10):
#   print(n, find_lucky(hamlet, n))
# ```
# 
# Please post your thoughts on Piazza as well!

# ##**Submission**
# 
# After implementing all the functions and testing them please download the notebook as "solution.py" and submit to gradescope under "Week12:Project:777" assignment tab and Moodle.
# 
# **NOTES**
# 
# * Be sure to use the function names and parameter names as given. 
# * DONOT use your own function or parameter names. 
# * Your file MUST be named "solution.py". 
# * Comment out any lines of code and/or function calls to those functions that produce errors. If your solution has errors, then you have to work on them but if there were any errors in the examples/exercies then comment them before submitting to Gradescope.
# * Grading cannot be performed if any of these are violated.

# **References and Additional Readings**
# * https://books.google.com/books?id=rn18DwAAQBAJ&pg=PT154&lpg=PT154&dq=William+Shakespeare++%22number+7
# * https://books.google.com/books?id=MwBNel_aX0wC&pg=PA67&lpg=PA67&dq=shakespeare+numerology
# * http://www.richardking.net/numart-ws.htm
# * https://www.celebrities-galore.com/celebrities/william-shakespeare/lucky-number/
# 
# 
# ¹ https://www.independent.co.uk/life-style/history/42-the-answer-to-life-the-universe-and-everything-2205734.html
# 
# Addendum: As part of the working out the details for the 777 assignment (the idea came from reading about finding a reference to the hapaxe Hebenon) the following fun fact was found: William Shakespeare had a fascination with the number 7. So the question to ponder (after you finished this assignment): Did Shakespeare hide this fun fact inside of Hamlet or is it purely coincidental or an artifact of the translation?
