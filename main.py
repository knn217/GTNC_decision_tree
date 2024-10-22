from extract import extractData, transpose
import decision_tree as dt

data = extractData()
data_t = transpose(data)
output = data_t[-1]
print(dt.extractBranch(data_t[2]))

#dt.entropy(data_t[0], dt.extractBranch(data_t[0]), output)
dt.entropy(data_t[2], dt.extractBranch(data_t[2]), output, dt.extractBranch(output), [0, 1, 22, 30, 33])