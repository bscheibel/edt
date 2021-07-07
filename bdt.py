import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.tree import _tree
import numpy as np
import ast
import re
import pandas as pd
import sys
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt

np.random.seed(0)

def prob(val):
    return val / val.sum(axis=1).reshape(-1, 1)

def print_condition_old(node_indicator, leave_id, model, feature_names, feature, Xtrain, threshold, sample_id):
    print("WHEN", end=' ')
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    for n, node_id in enumerate(node_index):
        if leave_id[sample_id] == node_id:
            values = model.tree_.value[node_id]
            probs = prob(values)
            print('THEN Y={} (probability={}) (values={})'.format(
                probs.argmax(), probs.max(), values))
            print("\n")
            continue
        if n > 0:
            print('AND ', end='')
        if Xtrain.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = " <="

        else:
            threshold_sign = " > "
        if feature[node_id] != _tree.TREE_UNDEFINED:
            print(
                "%s %s %s" % (
                    feature_names[feature[node_id]],
                    threshold_sign,
                    threshold[node_id]
                    ),
                end=' ')


def int_to_string(x):
    if(x==1):
       x ="OK"
    else:
       x = "NOK"
    return x


def learn_tree(df, result_column, names):
    y_var = df[result_column].astype(int)
    X_var = df[names]
    features = list(X_var)
    no_features = len(features)
    print("No of features: " + str(no_features))
    X_train, X_test, y_train, y_test = train_test_split(X_var,y_var, test_size=0.10, shuffle=False, stratify=None)
    model = tree(criterion='gini', max_depth=None, max_features=no_features, splitter="best", random_state=0, min_samples_leaf=3) #criterion: entropy or gini, max_depth=None
    model.fit(X_train,y_train)
    pred_model = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_model)
    print(('Accuracy of the model is {:.0%}'.format(accuracy)))
    feature_names = df.columns[:no_features]
    target_names = (df[result_column].unique().tolist())
    if(type(target_names[0])==int):
        target_names = [int_to_string(i) for i in target_names]
    plot_tree(decision_tree=model,
                feature_names=feature_names,
                class_names=str(target_names),
                filled=True)
    plt.savefig('files/tree_bdt.png')

    node_indicator = model.decision_path(X_train)
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    leave_id = model.apply(X_train)
    tree_rules = export_text(model, features)
    print(tree_rules)
    first = (model.predict(X_train) == 0).nonzero()[0]
    #for f in first:
    print_condition_old(node_indicator, leave_id, model, features, feature, X_train, threshold, first[0])


    return accuracy, tree_rules

def run(file, result_column):
    df = pd.read_csv(file)

    df = df.fillna(False)
    num_attributes = []
    df = df.rename(columns={element: re.sub(r'\w*:', r'', element) for element in df.columns.tolist()})
    df = df.rename(columns={element: element.replace(" ", "") for element in df.columns.tolist()})
    df = df.rename(columns={element: element.replace("-", "") for element in df.columns.tolist()})

    for column in df:
        first_value = (df[column].iloc[0])
        if isinstance(first_value, (int, float, np.int64, bool)):
            num_attributes.append(column)
        else:
            try:
                df[column] = df[column].apply(lambda x: ast.literal_eval(str(x)))
                num_attributes.append(column)
            except:
                pass

    num_attributes = [i for i in num_attributes if i not in result_column]
    print(num_attributes)

    learn_tree(df, result_column, num_attributes)

if __name__ == "__main__":
    file = sys.argv[1]
    res = sys.argv[2]
    run(file, res)