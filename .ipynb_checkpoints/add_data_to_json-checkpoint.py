# +
import json
import sys
import os

try:
    org = sys.argv[1]
    app = sys.argv[2]
    stg = sys.argv[3]
    version = sys.argv[4]
    build_time = sys.argv[5]
    json_file = sys.argv[6]

    json_path = os.path.join("Deployment-registry",org,app,stg,json_file)

    with open(json_path, 'r') as fp:
        deployment_data = json.load(fp)

    data_to_append = {"version":version,"build_time":build_time}

    deployment_data['image'].append(data_to_append)

    with open(json_path, 'w') as fp:
        json.dump(deployment_data, fp)
except:
    print('wrong input sequence')
