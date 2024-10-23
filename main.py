from extract import extractData, transpose, getDir
import decision_tree as dt

data = extractData()
data_t = transpose(data)
output = data_t[-1]
#print(dt.extractBranch(data_t[2]))

#dt.entropy(data_t[0], dt.extractBranch(data_t[0]), output)
#dt.entropy(data_t[2], dt.extractBranch(data_t[2]), output, dt.extractBranch(output), [0, 1, 22, 30, 33])
#print(dt.attrBranchDivider(data_t[2], dt.extractBranch(data_t[2])))
#print(dt.percentage(data_t[2], dt.extractBranch(data_t[2])))

#data = data[:2]
tree = dt.DTree(data)
rows = list(range(0, len(data)))
cols = list(range(0, len(data[0])))
#cols = [-1]
print(tree.getEntropies(rows, cols))
print(tree.balanceSegment())
print(tree.getEntropies(rows, cols))
tree.train(rows, cols[:-1])

def saveToTxt(data, name):
    dir = getDir(name)
    with open(dir, 'w', encoding='utf8') as f:
        for line in data:
            #print(line)
            f.write(line)
    return
saveToTxt(tree.log(), 'log/DTree.txt')