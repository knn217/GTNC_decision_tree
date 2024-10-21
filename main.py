from extract import extractData, transpose
from decision_tree import extractBranch

data = extractData()
data_t = transpose(data)
print(extractBranch(data_t[2]))