# EDT

Python version: 3.9

Use pip to install the needed packages: \
-sklearn \
-numpy\
-pandas\
-re\
-ast\
-sys\
-tree_code\
-matplotlib\
-warnings\

BDT = Basic Decision Tree\
Sklearn implementation of decision tree + Generate decision rule

EDT = Extended Decision Tree\
BDT with extension of latent variables to discover binary decision rules

Start in command line using: python [edt/bdt].py [file] [result_variable] \
E.g.: "python edt.py files/out.csv res"
 
Input file has to be a Dataframe, each row corresponds to one trace. Conversion from an XES event log to dataframe can be done using PM4PY (https://pm4py.fit.fraunhofer.de/)

Running Example: \
    file="files/out.csv" \
    result_column = "res" \
    program call: "python edt.py files/out.csv res" 
    

