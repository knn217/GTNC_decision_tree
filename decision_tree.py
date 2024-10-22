from extract import transpose
import math

def extractBranch(attr):
    # get the branches of each attribute
    # example: 
    # input: ([3, 4, 7, 6, 9, 4, 4, 2, 3], 4)
    # output: {2, 3, 4, 6, 7, 9}
    branches = set(attr)
    return list(branches)

def convertBranch(continuous_branch, seg=1):
    # turn continuous to discrete branches of range (for int and float)
    # example: 
    # input: ([3, 5, 7, 6, 4, 9, 8], seg=5)
    # output: ['-inf', 4.2, 5.4, 6.6, 7.8, '+inf']
    if seg < 1 or not (isinstance(continuous_branch[0], int) or isinstance(continuous_branch[0], float)):
        return continuous_branch
    max_val = max(continuous_branch)
    min_val = min(continuous_branch)
    temp = max_val - min_val
    dicrete_branches = []
    for i in range(1, seg):
        dicrete_branches.append(i/seg * (temp) + min_val)
    return [float('-inf')] + dicrete_branches + [float('inf')]

def findMatch(arr, match):
    # get the indices in array that match a value
    # example: 
    # input: ([3, 4, 7, 6, 9, 4, 4, 2, 3], 4)
    # output: [1, 5, 6]
    index = []
    for i in range(0, len(arr)):
        if arr[i] == match:
            index.append(i)
    return index

def findRange(arr, low, high):
    # get the indices in array that is in a range (low, high]
    # example: 
    # input: ([3, 4, 7, 6, 9, 4, 4, 2, 3], 4, 7)
    # output: [2, 3]
    index = []
    for i in range(0, len(arr)):
        if arr[i] > low and arr[i] <= high:
            index.append(i)
    return index

def attrBranchDivider(attr, attr_branches, type = 'discrete', seg=1):
    # combine findMatch and findRange and use branches as input
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: [[2, 7], [0, 8], [1, 5, 6], [3, 4]]
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: [[0, 1, 3, 4, 7], [2, 5, 8, 10], [6, 9]]
    indx_list = []
    if type == 'continuous':
        #attr_branches = convertBranch(attr_branches, seg)
        attr_branches = attr_branches
        for i in range(len(attr_branches)-1):
            indx_list.append(findRange(attr, attr_branches[i], attr_branches[i+1]))
    elif type == 'discrete':
        for each_attr in attr_branches:
            indx_list.append(findMatch(attr, each_attr))
    return indx_list #list of indexes for branch

def countBranchInstance(attr, attr_branches, type = 'discrete'):
    # get number of instances of each branch value in the attribute column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: [2, 2, 3, 2]
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: [5, 4, 2]
    branch_list = attrBranchDivider(attr, attr_branches, type)
    #print(attr)
    #print(attr_branches)
    #print(branch_list)
    for i in range(len(branch_list)):
        branch_list[i] = len(branch_list[i])
    return branch_list

def percentage(attr, attr_branches, type = 'discrete'):
    # calculate percentage of each value for a column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: [0.2222222222222222, 0.2222222222222222, 0.3333333333333333, 0.2222222222222222]
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: [0.45454545454545453, 0.36363636363636365, 0.18181818181818182]
    p_list = countBranchInstance(attr, attr_branches, type)
    s = sum(p_list)
    p_list = [i / s for i in p_list]
    return p_list

def entropy(attr, attr_branches, type = 'discrete'):
    # calculate the entropy of a column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: 1.2460468309523616
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: 0.9431887805255734
    p = percentage(attr, attr_branches, type)
    entr = [-i * math.log(i, 3) for i in p]
    return sum(entr)

def gini(attr, attr_branches, type = 'discrete'):
    # calculate the gini of a column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: 0.7407407407407407
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: 0.628099173553719
    p = percentage(attr, attr_branches, type)
    gini = [i**2 for i in p]
    return (1 - sum(gini))

class DTreeNode:
    count = 0
    # constructor
    def __init__(self, rows, cols, type = 'discrete'):
        self.rows = rows
        self.cols = cols
        #self.level = level
        self.type = type
        self.conditions = []
        self.chance = 0
        self.children = []
        # increase static counter
        DTreeNode.count += 1
        return
        
    # log number of nodes
    def printCount(self):
        print(DTreeNode.count)
        return
        
    def discreteBranch(self, value):
        
        return
    
class DTree:
    def __init__(self, dataset):
        self.dataset = dataset # rows
        self.dataset_t = transpose(dataset) # columns
        self.root_node = None
        self.seg = 0
        return
    
    # get entropy list of the tree's sub_dataset for specific indices(rows) and columns
    def getEntropies(self, rows, cols, seg=1):
        sub_dataset = [self.dataset[i] for i in rows] # get the sub dataset from indices
        sub_dataset_t = transpose(sub_dataset)
        sub_dataset_t = [sub_dataset_t[i] for i in cols]
        #print(sub_dataset_t)
        if seg < 1: # shouldn't calculate entropy for continuous if seg < 1
            #sub_dataset_t = [i for i in sub_dataset_t if not (isinstance(i[0], int) or isinstance(i[0], float))] # filter int and float lists
            ##print(sub_dataset_t)
            #entropies = [entropy(i, extractBranch(i)) for i in sub_dataset_t]
            #return entropies
            pass
        # determine each column data is discrete or continuous for calculating entropy
        data_type = ['continuous' if (isinstance(i[0], int) or isinstance(i[0], float)) else 'discrete' for i in sub_dataset_t]
        branches_list = [convertBranch(extractBranch(sub_dataset_t[i]), seg) for i in range(len(sub_dataset_t))]
        entropies = [entropy(sub_dataset_t[i], branches_list[i], type=data_type[i]) for i in range(len(sub_dataset_t))]
        return entropies
    
    # calculate the most balanced value to segment continuous attributes
    def calculateSegment(self):
        branch_num_list = [0 if (isinstance(i[0], int) or isinstance(i[0], float)) else len(extractBranch(i)) for i in self.dataset_t]
        print(branch_num_list)
        self.seg = len(self.dataset)
        c_attr = 0
        for i in branch_num_list:
            if i != 0: self.seg /= i
            else: c_attr += 1 
        self.seg = self.seg ** (1/c_attr)
        return math.ceil(self.seg)
    
    # calculate entropy for this node's indices
    def train(self, node):
        #entr = 
        #node.conditions = 
        return
        
