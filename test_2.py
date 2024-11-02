from extract import extractData, transpose, getDir
import decision_tree as dt
from test_1 import saveToTxt, train_test_split, accuracy, confusion_score
import numpy as np

if __name__ == "__main__":
    data = extractData('drug200.csv')
    data_t = transpose(data)
    X, y = transpose(data_t[:-1]), data_t[-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    train = transpose(transpose(X_train) + [y_train])
    test = transpose(transpose(X_test) + [y_test])
    output = data_t[-1]
    
    #print(train)
    
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
    #for i in range(len(y_test)):
    #    print(str(pred[i]) + '|' + str(y_test[i]))
    #    if y_test[i] == pred[i]:
    #        count += 1
    #print(count/len(pred))
    
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
