


##def createWordlist(filename):
##    text = open(filename, "r")
##    wordlist = []
##    for word in text:
##        word = word.strip()
##        if len(word) == 5 or len(word) != len(set(word)) or word[-1] != 's':
##            wordlist.append(word)
##
##    count = len(wordlist)
##    return(wordlist, count)
            



##
def createWordlist(filename): 
####    """ Read words from the provided file and store them in a list.
####    The file contains only lowercase ascii characters, are sorted
####    alphabetically, one word per line. Filter out any words that are
####    not 5 letters long, have duplicate letters, or end in 's'.  Return
####    the list of words and the number of words as a pair. """
####    with open('C:\name\MyDesktop\words.txt') as f:
####        lines = f.read().splitlines()
    text = open("words.txt", "r")
    wordlist = []

    for word in text:
        word = word.strip()
        if len(word) == 5:
            for i in word:
                if len(set(word))!=len(word):
                    pass
                else:
                    if word[-1] != "s":
                        wordlist.append(word)
                        break
    count = len(wordlist)
    return(wordlist, count)

def containsAll(wordlist, include):
    keep = set()
    for word in wordlist:
        if set(include) <= set(word):
            keep.add(word)
    return keep
    
##    set1 = {}
##    for word in wordlist:
##        for i in word:
##            if include == i:
##                set1.append(word)
##    return(set1)             
##    """ Given your wordlist, return a set of all words from the wordlist
##    that contain all of the letters in the string include.  
##    """
##    ...
##
def containsNone(wordlist, exclude):
##    """ Given your wordlist, return a set of all words from the wordlist
##    that do not contain any of the letters in the string exclude.  
##    """
##    ...
    import re
    keep = set()
    for word in wordlist:
        if re.search (exclude, word):
            pass
        else:
            keep.add(word)
    print(keep)
    return keep
##
def containsAtPositions(wordlist, posInfo):
    keep = set()
##    """ posInfo is a dictionary that maps letters to positions.
##    You can assume that the positions are in [0..4].  Return a set of
##    all words from the wordlist that contain the letters from the
##    dictionary at the indicated positions. For example, given posInfo
##    {'a': 0, 'y': 4}.   This function might return the set:
##    {'angry', 'aptly', 'amply', 'amity', 'artsy', 'agony'}. """

  

    v = list(posInfo.values())
    k = list (posInfo.keys())

    for word in wordlist:
        if word[v[0]] == k[0] and word[v[1]] == k[1]:
            keep.add(word)
    return keep

##    ...
##
##def getPossibleWords(wordlist, posInfo, include, exclude):
##    """ Finally, given a wordlist, dictionary posInfo, and
##    strings include and exclude, return the set of all words from 
##    the wordlist that contains the words that satisfy all of 
##    the following:
##    * has letters in positions indicated in posInfo
##    * contains all letters from string include
##    * contains none of the letters from string exclude.
##    """


