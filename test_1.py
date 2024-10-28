from extract import extractData, transpose, getDir
import decision_tree as dt
import numpy as np

def saveToTxt(data, name):
    dir = getDir(name)
    with open(dir, 'w', encoding='utf8') as f:
        for line in data:
            #print(line)
            f.write(line)
    return

def train_test_split(X, y, random_state=40, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Parameters:
    X (ndarray): A 2D array of features (n_samples, n_features).
    y (ndarray): A 1D array of labels (n_samples,).
    random_state (int): The random state to use. Default is 40.
    test_size (float): The proportion of the data to include in the test set. Default is 0.2.

    Returns:
    tuple: A tuple of (X_train, X_test, y_train, y_test).
    """
    idx = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    idx_test = idx[:int(test_size * len(X))]
    idx_train = idx[int(test_size * len(X)):]
    X_train = [X[i] for i in idx_train]
    X_test = [X[i] for i in idx_test]
    y_train = [y[i] for i in idx_train]
    y_test = [y[i] for i in idx_test]
    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    data = extractData('breast-cancer.csv')
    data_t = transpose(data)
    X, y = transpose(data_t[2:]), data_t[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train = transpose(transpose(X_train) + [y_train])
    test = transpose(transpose(X_test) + [y_test])
    output = data_t[-1]
    
    print(train)
    
    tree = dt.DTree(train)
    rows = list(range(0, len(train)))
    cols = list(range(0, len(train[0])))
    
    tree.balanceSegment()
    tree.train(rows, cols[:-1])
    
    saveToTxt(tree.log(), 'log/DTree.txt')
    pred = [tree.query(i) for i in X_test]
    pred = [str(i) for i in pred]
    saveToTxt('\n'.join(pred), 'log/pred.txt')
    #print(pred)
    
    count = 0
    for i in range(len(y_test)):
        print(str(pred[i]) + '|' + str(y_test[i]))
        if y_test[i] == pred[i]:
            count += 1
    print(count/len(pred))