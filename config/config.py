import json
from collections import OrderedDict

#load config
def get_config(config):
    json_str = ''
    with open(config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str, object_pairs_hook=OrderedDict)
    return config 