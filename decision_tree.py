from extract import transpose

def extractBranch(attr):
    # get the branches of each attribute
    branches = set(attr)
    return branches

def dicreteBranch(continuous_branches):
    # turn continuos to discrete braches
    dicrete_branches = continuous_branches
    return dicrete_branches

def findMatch(arr, match):
    # get the indices in array that match a value
    index = []
    for i in range(0, len(arr)):
        if arr[i] == match:
            index.append(i)
    return index

def findRange(arr, low, high):
    # get the indices in array that match a value
    index = []
    for i in range(0, len(arr)):
        if arr[i] > low and arr[i] <= high:
            index.append(i)
    return index

def attrDivider(attr, attr_branches, type = 'discrete'):
    indx_list = []
    if type == 'continuous':
        attr_branches = dicreteBranch(attr_branches)
        
    elif type == 'discrete':
        for each_attr in attr_branches:
            indx_list.append([])
        
    return indx_list #list of indexes for branch

def entropy(attr, attr_branches, output, output_branches, index = None, type = 'discrete'):
    if not index:
        index = list(range(0, len(output)))
    if type == 'continuous':
        attr_branches = dicreteBranch(attr_branches)
    print(index)
    new_attr = [attr[i] for i in index]
    new_output = [output[i] for i in index]
    print(new_attr)
    print(new_output)
    print(attr_branches)
    return

