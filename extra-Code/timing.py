import sys

def main():
    input = open(sys.argv[1],"r")
    rawText = input.read()
    lines = rawText.split("\n")
    listOfLists = [k.split("\t") for k in lines]
    dict_use = listOfLists

    listOfLists = [[i[1],i[-2]] for i in listOfLists if i[0]!='']
    listOfLists = listOfLists[1:]

    mydict = {}
    for i,j in listOfLists:
        j = int(j) if j!='\\N' else 0
        if i not in mydict:
            mydict[i] = j
        else:
            mydict[i] = mydict[i] + j

    listOfLists = [i[0] for i in listOfLists]

    for i in set(listOfLists):
        mydict[i] = round(mydict[i]/listOfLists.count(i),2)

    fp = open(sys.argv[2],'w')
    for k in mydict:
        s = k + ': ' + str(mydict[k]) + ' mins' + '\n'
        fp.write(s)
    fp.close()

def makeTimingDict(file):
    input = open(file","r")
    rawText = input.read()
    lines = rawText.split("\n")
    listOfLists = [k.split("\t") for k in lines]
    ret_dict = {int(i[0][2:]): {'primaryTitle':i[2] ,'titleType': i[1],'runtimeMinutes':i[-2],'startYear':i[5]} for i in listOfLists[1:-1]}
    return ret_dict

if __name__ == "__main__":
    main()