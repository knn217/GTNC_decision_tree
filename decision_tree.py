from extract import transpose
import math

def extractBranch(attr):
    # get the branches of each attribute
    # example: 
    # input: ([3, 4, 7, 6, 9, 4, 4, 2, 3], 4)
    # output: {2, 3, 4, 6, 7, 9}
    branches = sorted(list(set(attr)))
    return branches

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
        if low < arr[i] and arr[i] <= high:
            index.append(i)
    return index

def attrBranchDivider(attr, attr_branches, type='discrete'):
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

def countBranchInstance(attr, attr_branches, type='discrete'):
    # get number of instances of each branch value in the attribute column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: [2, 2, 3, 2]
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: [5, 4, 2]
    branch_list = attrBranchDivider(attr, attr_branches, type)
    #print('=============================================================')
    #print('attr:', attr)
    #print('attr_branches', attr_branches)
    #print(branch_list)
    for i in range(len(branch_list)):
        branch_list[i] = len(branch_list[i])
    return branch_list

def percentage(attr, attr_branches, type='discrete'):
    # calculate percentage of each value for a column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: [0.2222222222222222, 0.2222222222222222, 0.3333333333333333, 0.2222222222222222]
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: [0.45454545454545453, 0.36363636363636365, 0.18181818181818182]
    p_list = countBranchInstance(attr, attr_branches, type)
    s = sum(p_list)
    if s == 0:
        return [0]
    p_list = [i / s for i in p_list]
    return p_list

def entropy(attr, attr_branches, type='discrete', z = 0.001):
    # calculate the entropy of a column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: 1.2460468309523616
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: 0.9431887805255734
    p = percentage(attr, attr_branches, type)
    #print(p)
    entr = [-(i+z) * math.log(i+z, 2) for i in p]
    return sum(entr)

def gini(attr, attr_branches, type='discrete', z = 0.001):
    # calculate the gini of a column
    # example: 
    # input: ([3, 4, 2, 6, 6, 4, 4, 2, 3], {2, 3, 4, 6}, type = 'discrete')
    # output: 0.7407407407407407
    
    # input: ([1, 2, 6, 0, 2, 4, 8, 1, 4, 7, 3], {float('-inf'), 2, 6, float('inf')}, type = 'continuous')
    # output: 0.628099173553719
    p = percentage(attr, attr_branches, type)
    gini = [(i+z)**2 for i in p]
    return (1 - sum(gini))

def extractIndx(arr, indices):
    # extract indices from an array
    # exmaple:
    # input: ([2, 4, 12, 3, 2, 5], [1, 2, 4])
    # output: [4, 12, 2]
    extr = [arr[i] for i in indices]
    #print(extr)
    return extr

class DTreeNode:
    count = 0
    # constructor
    def __init__(self, rows, col=0, name='', type='discrete'):
        self.rows = rows # records(rows) to handle
        self.col = col   # attr(column) index in the dataset
        self.name = name # attr(column) name
        self.type = type # discrete or continuous
        self.conditions = []
        self.chance = 0
        self.children = []
        # increase static counter
        DTreeNode.count += 1
        return
    
    def update(self, col=0, type='discrete', conditions=[]):
        self.col = col   # attr(column) index in the dataset
        self.type = type # discrete or continuous
        self.conditions = conditions
        return
        
    # log number of nodes
    def printCount(self):
        print(DTreeNode.count)
        return
    
    def logInfo(self, line_break = '\n', indent=''):
        log = indent + '=======================' + line_break
        log += indent + 'rows:' + str(self.rows) + line_break
        log += indent + 'col:' + str(self.col) + line_break
        log += indent + 'type:' + str(self.type) + line_break
        log += indent + 'conditions:' + str(self.conditions) + line_break
        log += indent + 'children:' + str(len(self.children)) + line_break
        return log
    
    # log string recursively (find children nodes to log)
    def log(self, line_break = '\n', indent = ''):
        log = ''
        log += self.logInfo(line_break, indent)
        for child in self.children:
            log += child.log(line_break, indent + '    ')
        return log
        
    def discreteBranch(self, value):
        for i in range(len(self.conditions)):
            if self.conditions[i] == value:
                return self.children[i]
        return None
    
    def continuousBranch(self, value):
        for i in range(len(self.conditions)-1):
            if self.conditions[i] < value and value < self.conditions[i+1]:
                return self.children[i]
        return None
    
    def branch(self, record):
        value = record[self.col]
        if self.type == 'result':
            return self.conditions
        elif self.type == 'discrete':
            return self.discreteBranch(value)
        elif self.type == 'continuous':
            return self.continuousBranch(value)
        return None
    
class DTree:
    def __init__(self, dataset):
        self.dataset = dataset # rows
        self.dataset_t = transpose(dataset) # columns
        self.dataset_output = self.dataset_t[-1]
        self.dataset_output_idx = len(self.dataset_t)
        self.root_node = None
        self.seg = [1] * len(self.dataset_t)
        self.conditions = [convertBranch(extractBranch(self.dataset_t[i]), self.seg[i]) for i in range(len(self.dataset_t))]
        self.datatype = ['continuous' if (isinstance(i[0], int) or isinstance(i[0], float)) else 'discrete' for i in self.dataset_t]
        print(self.conditions)
        print(self.datatype)
        return
    
    def log(self):
        return self.root_node.log()
    
    # get entropy list of the tree's sub_dataset for specific indices(rows) and columns
    def getEntropies(self, rows, output_cols):
        sub_dataset = [self.dataset[i] for i in rows] # get the sub dataset from indices
        sub_dataset_t = transpose(sub_dataset)
        sub_dataset_t = [sub_dataset_t[i] for i in output_cols]
        #print(sub_dataset_t)
        # determine each column data is discrete or continuous for calculating entropy
        entropies = [entropy(sub_dataset_t[i], self.conditions[i], self.datatype[i]) for i in range(len(sub_dataset_t))]
        return entropies
    
    # calculate the most balanced value to segment continuous attributes
    def balanceSegment(self):
        branch_num_list = [0 if (isinstance(i[0], int) or isinstance(i[0], float)) else len(extractBranch(i)) for i in self.dataset_t]
        #print(branch_num_list)
        self.seg = len(self.dataset)
        c_attr = 0
        print('branch_num_list: ', branch_num_list)
        for i in range(len(branch_num_list)-1):
            if branch_num_list[i] != 0: self.seg /= branch_num_list[i]
            else: c_attr += 1 
        balance_seg = math.ceil(self.seg ** (1/c_attr))
        self.seg = [balance_seg if i==0 else i for i in branch_num_list]
        print('seg: ', self.seg)
        # Update the conditions after updating seg
        self.conditions = [convertBranch(extractBranch(self.dataset_t[i]), self.seg[i]) for i in range(len(self.dataset_t))]
        return
    
    # calculate weighted segment for continuous attributes
    # Note that this function return conditions based on it's input dataset rows
    def weightedSegment(self, rows):
        # dataset is in rows and dataset_t is in columns
        dataset = [self.dataset[i] for i in rows]
        dataset_t = transpose(dataset)
        branch_num_list = [0 if (isinstance(i[0], int) or isinstance(i[0], float)) else len(extractBranch(i)) for i in dataset_t]
        #print(branch_num_list)
        seg = len(dataset)
        c_attr = 0
        print('branch_num_list: ', branch_num_list)
        for i in range(len(branch_num_list)-1):
            if branch_num_list[i] != 0: seg /= branch_num_list[i]
            else: c_attr += 1 
        balance_seg = math.ceil(seg ** (1/c_attr))
        seg = [balance_seg if i==0 else i for i in branch_num_list]
        print('seg: ', seg)
        # Update the conditions after updating seg
        conditions = [convertBranch(extractBranch(dataset_t[i]), seg[i]) for i in range(len(dataset_t))]
        return conditions
    
    # create the list of list of indexes for each branch
    def L_L_Idx(self, rows, input_cols):
        sub_dataset = [self.dataset[i] for i in rows] # get the sub dataset from indices(rows)
        sub_dataset_t = transpose(sub_dataset)
        #sub_dataset_t = [sub_dataset_t[i] for i in input_cols] # get all the attribute(cols) for input
        tmp_data = [sub_dataset_t[i] for i in input_cols]
        tmp_cond = [self.conditions[i] for i in input_cols]
        tmp_type = [self.datatype[i] for i in input_cols]
        # create the list of list of indexes for each branch
        L_L_Idx = [attrBranchDivider(tmp_data[i], tmp_cond[i], type = tmp_type[i]) for i in range(len(tmp_data))]
        return L_L_Idx
    
    # create the list of list of indexes for each branch
    def L_L_Idx_V2(self, rows, input_cols):
        sub_dataset = [self.dataset[i] for i in rows] # get the sub dataset from indices(rows)
        sub_dataset_t = transpose(sub_dataset)
        #sub_dataset_t = [sub_dataset_t[i] for i in input_cols] # get all the attribute(cols) for input
        tmp_data = [sub_dataset_t[i] for i in input_cols]
        //TODO tmp_cond = [self.conditions[i] for i in input_cols]
        tmp_type = [self.datatype[i] for i in input_cols]
        # create the list of list of indexes for each branch
        L_L_Idx = [attrBranchDivider(tmp_data[i], tmp_cond[i], type = tmp_type[i]) for i in range(len(tmp_data))]
        return L_L_Idx
    
    def postProccessChildren(self, node):
        max_occur = -1
        max_occur_result = ''
        for child in node.children:
            if len(child.rows) > max_occur and isinstance(child.conditions, str):
                max_occur == len(child.rows)
                max_occur_result = child.conditions
        for child in node.children:
            if len(child.rows) == 0:
                child.conditions = max_occur_result
        return
    
    # recursive train
    def train(self, rows, input_cols, node = None):
        # rows: the records(rows) that this node is built on
        # input_cols: the columns left for this node to find entropies
        tmp_output = extractIndx(self.dataset_output, rows) # extract the outputs that this node needs
        #===================================================================
        # [1] base cases
        if len(input_cols) == 0: # if there's only no input column then output column as node's condition
            if len(tmp_output) == 0:
                node = None
                return
            result = max(tmp_output,key=tmp_output.count)
            node.update(self.dataset_output_idx, type='result', conditions=result)
            return
        out_cond = self.conditions[-1]
        #print('out_cond: ', out_cond)
        out_entr = entropy(tmp_output, out_cond)
        if out_entr == 0:
            return
        #print('out_entr: ', out_entr)
        
        #===================================================================
        # [2] handle root node
        if node == None:
            node = DTreeNode(rows)
            self.root_node = node
        
        #===================================================================    
        # [3] Find the Entropies for each attr (column) 
        L_L_Idx = self.L_L_Idx(rows, input_cols) # (list of indexes for each branch) in each attr (collumn)
        #print('L_L_Idx: ', L_L_Idx)
        col_entropies = [sum([entropy(extractIndx(self.dataset_output, each_attr_idx[i]), out_cond) for i in range(len(each_attr_idx))]) for each_attr_idx in L_L_Idx]
        #print('col_entropies: ', col_entropies)
        #=======================================
        # Since DTrees want to maximize information gain (output entr - attribute entr)
        # this is equivalent to minimizing attribute (column) entropy
        # => we pick the smallest col_entr in the list
        #=======================================
        
        # [4] Pick the lowest entr column for current node and add information to it
        tmp_cond = [self.conditions[i] for i in input_cols]
        tmp_type = [self.datatype[i] for i in input_cols]
        print('tmp_cond:', tmp_cond)
        print('tmp_type:', tmp_type)
        
        min_entr_col_indx = col_entropies.index(min(col_entropies))
        print('min_entr_col_indx:', min_entr_col_indx)
        node.update(input_cols[min_entr_col_indx], type=tmp_type[min_entr_col_indx], conditions=tmp_cond[min_entr_col_indx])

        #===================================================================
        # [5] Create children for current node and train them recursively
        #print('old_input_cols: ',input_cols)
        new_input_cols = [i for i in input_cols if i != input_cols[min_entr_col_indx]]
        #input_cols.remove(input_cols[min_entr_col_indx])
        #print('new_input_cols: ',new_input_cols)
        for i in L_L_Idx[min_entr_col_indx]: # get each branch's rows 
            child = DTreeNode(i) # add rows to child node
            node.children.append(child)   # add child node to current node
            #print('rows',i)
            if len(i) == 0:
                #print('tmp_output: ', tmp_output)
                result = max(tmp_output,key=tmp_output.count)
                #print('result: ', result)
                child.update(self.dataset_output_idx, type='result', conditions=result)
                continue
            self.train(i, new_input_cols, child)
            #print(i)
        # post proccess
        self.postProccessChildren(node)
        return
        
    def query(self, record):
        node = self.root_node
        while node.type != 'result':
            node = node.branch(record)
        return node.conditions
