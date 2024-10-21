# Python program to convert JSON to Python
import os

def getDir(file_name):
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    dir = os.path.join(path, file_name)
    print(dir)
    return dir

def saveToTxt(data, name):
    dir = getDir(name)
    with open(dir, 'w', encoding='utf8') as f:
        for line in data:
            #print(line)
            f.write(str(line) + "\n")
    return

def smartParse(str):
    str = str.replace("\n","")
    #print(str)
    try:
        i = int(str)
        return i
    except:
        pass
    try:
        f = float(str)
        return f
    except:
        pass
    return str

def extractCSV(txt, filter=[]):
    title = None
    data = []
    for line in txt:
        line_data = line.split(',')
        for i in filter:
            line_data.pop(i)
        if not title:
            title = [smartParse(i) for i in line_data]
        new_data = {}
        for i in range(len(line_data)):
            new_data.update({title[i]: smartParse(line_data[i])})
        #print(new_data)
        data.append(new_data)
    return data[1:]

def extractData():
    dir_drug = getDir('drug200.csv')
    data_drug = open(dir_drug, encoding='utf8')
    data_drug = extractCSV(data_drug) # pop filter from last to 1st
    data_drug = sorted([dict(t) for t in {tuple(d.items()) for d in data_drug}], key=lambda d: (d['Age'], d['Cholesterol'])) # remove duplicates
    #for i in data_drug:
    #    print(i)
    print('Number of drug: ', len(data_drug))
    saveToTxt(data_drug, 'log/drug.txt') # save to log

    return data_drug
