from sklearn import tree
import importlib
import graphviz 
import os

# [height, weight, shoe size]
X = [[181,80,44], [117,70,43], [160,60,38], [154,54,37],
     [166,65,40], [190,90,47], [175,64,39], [177,70,40],
     [159,55,37], [171,75,42], [181,85,43] ]

# X = [ int(x) for x in X]

# Genders
Y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

# initializing decision tree object
clf = tree.DecisionTreeClassifier()

# Decision tree var for training model
clf = clf.fit(X,Y)

# variable for storing prediction to test model
prediction = clf.predict([[190,70,43]])

# Print to terminal
# Creates data
# graph reners into pdf

print (prediction)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")

os.system('iris.pdf')
