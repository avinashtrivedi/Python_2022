import math
import os
from functools import reduce

def dotProduct(D1, D2):
    Sum = 0.0
    for key in D1:
        if key in D2:
            Sum += (D1[key] * D2[key])
    return Sum


def vector_angle(D1, D2):
    numerator = dotProduct(D1, D2)
    denominator = math.sqrt(dotProduct(D1, D1))*math.sqrt(dotProduct(D2, D2))
    return round(math.degrees(math.acos(numerator / denominator)), 5)

def get_loc(x,text):
    text = text.split()
    return (i for i in range(len(text)) if text[i]==x)

def inverted_index(words):      
    inverted = {}
    for index, word in enumerate(words):
        locations = inverted.setdefault(word, [])
        locations.append(index)
    return inverted

def inverted_index_add(inverted, doc_id, doc_index):        
    for word in doc_index.keys():
        locations = doc_index[word]
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted

def boolean_search(inverted, file_names, query):
    words = [word for _, word in enumerate(query.split()) if word in inverted]
    results = [set(inverted[word].keys()) for word in words]
    docs = reduce(lambda x, y: x & y, results) if results else []
    return {i+1 for i in docs}


def readLines(fileName):
    try:
        file = open(fileName, 'r')
        lines = file.readlines()
        file.close()
        return lines
    except:
        print("File {} Not Found".format(fileName))
        quit()

if __name__ == "__main__":
    #Input Files
    docFileName = 'docs.txt'
    queryFileName = 'queries.txt'

    # Read All Lines From Files
    docs = readLines(docFileName)
    queries = readLines(queryFileName)

    # Create Vector of docs
    docVectors = []
    for doc in docs:
        docWords = doc.strip().split()
        docVectors.append({key: docWords.count(key) for key in docWords})

    # 1. Building the Dictionary
    dictionary = {}
    for docVector in docVectors:
        for key in docVector:
            dictionary[key] = dictionary.get(key, 0) + docVector[key]
    # Printing dictionary Word Count
    print("Words in dictionary: ", len(dictionary))

    # 2. Building Inverted Index
    inverted_doc_indexes = {}
    files_with_index = []
    files_with_tokens = {}
    doc_id=0
    for fname,line in enumerate(docs):
        words = line.split()
        files_with_tokens[doc_id] = words
        doc_index = inverted_index(words)
        inverted_index_add(inverted_doc_indexes, doc_id, doc_index)
        files_with_index.append(fname+1)
        doc_id = doc_id+1

    # 3. Document Searching
    for query in queries:
        # Removing Space and \n from the end of query
        query = query.strip()

        # Split the query in words
        queryWords = query.split()

        # Printing Query
        print('Query: {}'.format(query))

        # Finding All the Relevant documents
        Relevant_Documents = boolean_search(inverted_doc_indexes, files_with_index,query)
        # Printing Relevant documents
        print('Relevant documents: {}'.format(
            " ".join(map(str, Relevant_Documents))))

        # Finding the angel between query and each Relevant documents
        angles = []
        for i in Relevant_Documents:
            # Create Vector of query
            queryVactor = {
                key: 1 if key in queryWords else 0 for key in docVectors[i-1]}
            # Storing doc id and vector angel
            angles.append((i, vector_angle(docVectors[i-1], queryVactor)))

        # sort vector angle based on angle
        angles.sort(key=lambda x: x[1])

        # Printing vector angle of query
        for angel in angles:
            print(angel[0], angel[1])
