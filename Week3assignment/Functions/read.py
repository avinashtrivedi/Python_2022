import os.path
import json

def read(path):
    try:
        with open(path) as fp:
            data = json.load(fp)
        return data
    except:
        print('Wrong path',path)
        return False