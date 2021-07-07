from sklearn.tree import _tree
import numpy as np
import sklearn
import graphviz
from sklearn import tree as tr
import time
from sklearn.tree import DecisionTreeClassifier as tree # tree algorithm
from sklearn.ensemble import RandomForestClassifier# tree algorithm
from sklearn.model_selection import train_test_split # splitting the data
from sklearn.metrics import accuracy_score # model precision
from sklearn.tree import plot_tree, export_text # tree diagram
import matplotlib.pyplot as plt

np.random.seed(0)

# pruning empty leaves: https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
#https://stackoverflow.com/questions/52554712/sklearn-decision-rules-for-specific-class-in-decision-tree
#https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def value2prob(value):
    return value / value.sum(axis=1).reshape(-1, 1)

def print_condition_old(node_indicator, leave_id, model, feature_names, feature, Xtrain, threshold, sample_id):
    print("WHEN", end=' ')
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    for n, node_id in enumerate(node_index):
        if leave_id[sample_id] == node_id:
            values = model.tree_.value[node_id]
            probs = value2prob(values)
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


def print_condition(node_indicator, leave_id, model, feature_names, feature, Xtrain, threshold, sample_id):
    print("WHEN", end=' ')
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    for n, node_id in enumerate(node_index):
        if leave_id[sample_id] == node_id:
            values = model.tree_.value[node_id]
            probs = value2prob(values)
            print('THEN Y={} (probability={}) (values={})'.format(
                probs.argmax(), probs.max(), values))
            continue
        if n > 0:
            print('AND ', end='')
        if Xtrain.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "= false"

        else:
            threshold_sign = "= true"
        if feature[node_id] != _tree.TREE_UNDEFINED:
            print(
                "%s %s" % (
                    feature_names[feature[node_id]],
                    threshold_sign,
                    ),
                end=' ')

def int_to_string(x):
    if(x==1):
       x ="OK"
    else:
       x = "NOK"
    return x

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    def recurse(node, depth):
        result = ["NOK", "OK"]
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} = {}:".format(indent, name, "false"))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} = {}".format(indent, name, "true"))
            recurse(tree_.children_right[node], depth + 1)
        else:
            class_name = np.argmax(tree_.value[node][0])
            print("{}return {}".format(indent, result[class_name]))
            print("\n\n")

    recurse(0, 1)

def learn_tree(df, result_column, names, old):
    start = time.time()
    y_var = df[result_column].astype(int)
    X_var = df[names]
    #X_var = X_var[X_var.columns.drop('measured')]
    #X_var = pd.get_dummies(X_var)
    features = list(X_var)
    no_features = len(features)
    print("No of features: " + str(no_features))
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.10, shuffle=False, stratify=None)
    model = tree(criterion='gini', max_depth=None, max_features=no_features, splitter="best", random_state=0, min_samples_leaf=3) #criterion: entropy or gini, max_depth=None
    #model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train,y_train)
    #print(model.decision_path(X_train))
    pred_model = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_model)
    print(('Accuracy of the model is {:.0%}'.format(accuracy)))
    end = time.time()
    print("Time: ", end-start)
    feature_names = df.columns[:no_features]
    target_names = (df[result_column].unique().tolist())
    if(type(target_names[0])==int):
        target_names = [int_to_string(i) for i in target_names]
    plot_tree(decision_tree=model,
                feature_names=feature_names,
                class_names=str(target_names),
                filled=True)
    plt.savefig('files/tree.png')

    node_indicator = model.decision_path(X_train)
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    leave_id = model.apply(X_train)

    tree_rules = export_text(model, features)
    print(tree_rules)
    first = (model.predict(X_train) == 0).nonzero()[0]
    for f in first:
         if old:
             print_condition_old(node_indicator, leave_id, model, features, feature, X_train, threshold, f)
         else:
             print_condition(node_indicator, leave_id, model, features, feature, X_train, threshold, f)

    return accuracy, tree_rules