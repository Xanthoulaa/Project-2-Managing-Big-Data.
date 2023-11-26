#Theme 4


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
column_names = ['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                'gill-spacing','gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring','veil-type',
                'veil-color','ring-number','ring-type','spore-print-color','population','habitat' ]

df = pd.read_csv('C:\\Users\\xrusa\\OneDrive\\Υπολογιστής\\agaricus-lepiota.data', header=None, na_values='?', names=column_names)

df.dropna(axis=0, inplace=True) 
print(df)
categoricalVariables=df.select_dtypes([object]).columns


for var in categoricalVariables:
    
    print("\tOne-Hot-Encoding variable ",var, " .....", sep="", end="")
    if var == "class":
       print("Ignored") 
       continue 
    df[var]=pd.Categorical(df[var])
    varDummies = pd.get_dummies(df[var], prefix = var)
    df = pd.concat([df, varDummies], axis=1)
    df=df.drop([var], axis=1)
    print("Done")
print("\n\tVariables of DataFrame mushroom after One-hot encoding:")
print("\t", df.columns)
df['newclass'] = ( df['class'].map( {'p':0, 'e':1}) )
df=df.drop(['class'], axis=1)
print("\n\nPreprocessing done.")
features = df.iloc[:, :-1]
classVariable = df['newclass']
X_train,X_test,Y_train,Y_test = train_test_split(features, classVariable,test_size=0.2, random_state = 100)
print("\nTraining the model (decision tree)......", sep='', end='')

model = DecisionTreeClassifier(criterion="entropy")
model_fit = model.fit(X_train , Y_train)
print("done.")
text_representation = tree.export_text(model)
print(text_representation)

print("\nUsing testing set to predict class attribute......", sep='', end='')
predictions = model.predict(X_test)
print("done.")
print("\nCalculating confusion matrix on the testing set......", sep="", end="")

cm = confusion_matrix(Y_test, predictions)
cm = pd.DataFrame(cm)
print("done.")
print("Confusion matrix:")
print(cm)
result=model.score(X_test, Y_test)
print ("\nModel's predictive accuracy is: %.2f%%" % (accuracy_score(Y_test,predictions)*100))

fig = plt.figure(figsize=(10,5))
_=tree.plot_tree(model, max_depth = 3,proportion = True, filled = True)
plt.show()
