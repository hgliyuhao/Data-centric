import math
import orjson as json
import fairies as fa

def orjson_read(fileName):
    with open(fileName,'rb') as f:
        json_data = json.loads(f.read())
    return json_data

def orjson_write(fileName,res):
    with open(fileName, 'wb') as json_file:
        json_str = json.dumps(res)
        json_file.write(json_str)
    json_file.close()
    
