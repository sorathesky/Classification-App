from sklearn import tree

# [height, weight, shoe size]
X = [[181,80,44], [117,70,43], [160,60,38], [154,54,37],
     [166,65,40], [190,90,47], [175,64,39], [177,70,40],
     [159,55,37], [171,75,42], [181,85,43] ]

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
print (prediction)