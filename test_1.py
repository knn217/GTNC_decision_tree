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

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The accuracy of the array.
    """
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 

def confusion_score(y_true, y_pred, mode="macro"):
    """
    Calculates the F1 score for multi-label classification in either macro or micro mode.

    Parameters:
    y_true (np.ndarray): A 1D array of true labels.
    y_pred (np.ndarray): A 1D array of predicted labels.
    mode (str): Either 'macro' or 'micro'. Determines the F1 score calculation mode.

    Returns:
    float: The F1 score.
    """

    def one_hot_encode(arr: np.ndarray):
        # Get unique classes and their indices
        unique_classes, indices = np.unique(arr, return_inverse=True)
        
        # Create one-hot encoded matrix
        one_hot = np.zeros((arr.size, unique_classes.size))
        one_hot[np.arange(arr.size), indices] = 1
        
        return one_hot
    
    # One-hot encode the true and predicted labels
    y_true_one_hot = one_hot_encode(y_true)
    y_pred_one_hot = one_hot_encode(y_pred)
    
    # Adjust shapes if y_pred_one_hot has fewer columns
    if y_true_one_hot.shape[1] != y_pred_one_hot.shape[1]:
        y_pred_one_hot = np.pad(
            y_pred_one_hot,
            ((0, 0), (0, y_true_one_hot.shape[1] - y_pred_one_hot.shape[1])),
            mode="constant",
        )
    
    # True positives, predicted positives, actual positives per label
    true_positives = np.sum((y_true_one_hot == 1) & (y_pred_one_hot == 1), axis=0)
    predicted_positives = np.sum(y_pred_one_hot == 1, axis=0)
    actual_positives = np.sum(y_true_one_hot == 1, axis=0)
    
    # Calculate precision, recall, and F1 scores for each label
    precision_per_label = true_positives / (predicted_positives + 1e-9)
    recall_per_label = true_positives / (actual_positives + 1e-9)
    f1_per_label = 2 * (precision_per_label * recall_per_label) / (precision_per_label + recall_per_label + 1e-9)

    if mode == "macro":
        # Macro F1: Average F1 scores across labels
        f1_score = np.mean(f1_per_label)
        precision = np.mean(precision_per_label)
        recall = np.mean(recall_per_label)
        return precision, recall, f1_score
    elif mode == "micro":
        # Micro F1: Calculate global precision and recall and then F1
        total_true_positives = np.sum(true_positives)
        total_predicted_positives = np.sum(predicted_positives)
        total_actual_positives = np.sum(actual_positives)

        micro_precision = total_true_positives / (total_predicted_positives + 1e-9)
        micro_recall = total_true_positives / (total_actual_positives + 1e-9)
        f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-9)
        return micro_precision, micro_recall, f1_score
    else:
        raise ValueError("Mode should be 'macro' or 'micro'")


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
    
    pred = np.asarray(pred)
    y_test = np.asarray(y_test)
    print("--- Our Model (DT) ---")
    print(f"Model's Accuracy: {accuracy(y_test, pred)}")
    precision, recall, f1_score = confusion_score(y_test, pred, "macro")
    print(f"Model's F1 (Macro): {f1_score}")
    print(f"Model's Precision (Macro): {precision}")
    print(f"Model's Recall (Macro): {recall}")
    precision, recall, f1_score = confusion_score(y_test, pred, "micro")
    print(f"Model's F1 (Micro): {f1_score}")
    print(f"Model's Precision (Micro): {precision}")
    print(f"Model's Recall (Micro): {recall}")
