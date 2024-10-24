from extract import extractData, transpose, getDir
import decision_tree as dt

def saveToTxt(data, name):
    dir = getDir(name)
    with open(dir, 'w', encoding='utf8') as f:
        for line in data:
            #print(line)
            f.write(line)
    return

data = extractData()
data_t = transpose(data)
#data_t = data_t[2:] + [data_t[1]]
#data = transpose(data_t)

#print(data_t)
#print(data)
output = data_t[-1]

tree = dt.DTree(data)
rows = list(range(0, len(data)))
cols = list(range(0, len(data[0])))

tree.balanceSegment()
tree.train(rows, cols[:-1])

saveToTxt(tree.log(), 'log/DTree.txt')
pred = [tree.query(i) for i in data]
pred = [str(i) for i in pred]
saveToTxt('\n'.join(pred), 'log/pred.txt')
#print(pred)

count = 0
for i in range(len(data_t[-1])):
    #print(str(pred[i]) + '|' + str(data_t[-1][i]))
    if data_t[-1][i] == pred[i]:
        count += 1
print(count/len(pred))