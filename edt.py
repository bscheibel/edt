import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
import ast
import sys
import tree_code as tc


def create_features(df, result_column, names):
    new_names = []
    df_new = df
    comparison_operators = ['==', '!=','<', '>', '<=', '>=']
    for name1 in df_new:
        if name1 not in names:
            continue
        for name2 in df_new:
            for op in comparison_operators:
                if name1 == name2 or name2 not in names:
                    continue
                expression = "row." + name1 + op + "row." + name2
                new_name = str(expression)
                #print(expression)
                df[expression] = df.apply(lambda row: (eval(expression)), axis=1)
                new_names.append(new_name)
    return df, new_names


def define(file, result_column):

    df = pd.read_csv(file)

    df = df.fillna(False)
    num_attributes = []
    df = df.rename(columns={element: re.sub(r'\w*:', r'', element) for element in df.columns.tolist()})
    df = df.rename(columns={element: element.replace(" ", "") for element in df.columns.tolist()})
    df = df.rename(columns={element: element.replace("-", "") for element in df.columns.tolist()})

    for column in df:
        first_value = (df[column].iloc[0])
        #first_value = (df[column][0])
        if isinstance(first_value, (int, float, np.int64, bool)):
            num_attributes.append(column)
        #else: ##only needed if all numerical variables are coded as string
        #     try:
        #         df[column] = df[column].apply(lambda x: ast.literal_eval(str(x)))
        #         num_attributes.append(column)
        #     except:
        #         pass

    num_attributes = [i for i in num_attributes if i not in result_column]
    print(num_attributes)
    run(df,result_column, num_attributes)

def run(df, result_column, names):
    if(isinstance(result_column, (int, float))):
        df[result_column] = df[result_column].map(tc.int_to_string)
    df, new_names = create_features(df, result_column, names)
    #df.to_csv('files/temp.csv', index=False)  # , header = None)
    important_feat = new_names
    tc.learn_tree(df, result_column, important_feat)


if __name__ == "__main__":
    file = sys.argv[1]
    res = sys.argv[2]
    define(file, res)


