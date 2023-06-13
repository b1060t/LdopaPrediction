import json
import os
import os.path
import pandas as pd

def getPandas(name):
    data = pd.read_json(os.path.join('data', 'json', name+'.json'))
    return data

def getConfig(name):
    with open(os.path.join('config', name+'.json'), 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data
    
def getDict(name):
    with open(os.path.join(os.path.join('data', 'json', name+'.json')), 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data

def getDataPandas():
    data = pd.read_json(os.path.join('data', 'json', 'data.json'))
    return data

def writePandas(name, data):
    data = data.to_dict(orient='records')
    with open(os.path.join('data', 'json', name+'.json'), 'w+', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def writeData(name, data):
    with open(os.path.join('data', 'json', name+'.json'), 'w+', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def writeConfig(name, data):
    with open(os.path.join('config', name+'.json'), 'w+', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def writeGraph(name, data):
    with open(os.path.join('data', 'json', 'graph', name+'.json'), 'w+', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def getGraph(name):
    data = pd.read_json(os.path.join('data', 'json', 'graph', name+'.json'))
    return data