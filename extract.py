# Python program to convert JSON to Python
import os

def transpose(matrix):
    matrix_t =[[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return matrix_t

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
    data = []
    for line in txt:
        line_data = line.split(',')
        for i in filter:
            line_data.pop(i)
        data.append([smartParse(i) for i in line_data])
    return data[1:]

def extractData():
    dir_drug = getDir('data.csv')
    data_drug = open(dir_drug, encoding='utf8')
    data_drug = extractCSV(data_drug) # pop filter from last to 1st
    #for i in data_drug:
    #    print(i)
    print('Number of drug: ', len(data_drug))
    saveToTxt(data_drug, 'log/drug.txt') # save to log
    
    #print(transpose(data_drug))
    
    return data_drug
