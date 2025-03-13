
import json
import sys
import os

with open('deployment.json', 'r') as fp:
    deployment_data = json.load(fp)

org = sys.argv[1]
app = sys.argv[2]
stg = sys.argv[3]
version = sys.argv[4]
build_time = sys.argv[5]
json_file = sys.argv[6]

data_to_append = {"version":version,"build_time":build_time}

deployment_data['image'].append(data_to_append)

json_path = json_file #os.path.join("Deployment-registry",org,app,stg,json_file)

with open(json_path, 'w') as fp:
    json.dump(deployment_data, fp)
