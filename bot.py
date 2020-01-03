import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.tree import export_graphviz
import warnings
from tkinter import *
import tkinter
import pandas as pd
import numpy as np
import random
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
warnings.filterwarnings("ignore", category=DeprecationWarning)

#greeting file
gr = pd.read_csv(r'E:\BAHRIA UNIVERSITY (BSE 3A)\5th semester\AI lab\projects\AI-Healthcare-chatbot-final\Greeting Dataset.csv', engine='python')
gr = np.array(gr)
gd = gr[:,0]

#name file
nm = pd.read_csv(r'E:\BAHRIA UNIVERSITY (BSE 3A)\5th semester\AI lab\projects\AI-Healthcare-chatbot-final\Name Dataset.csv', engine='python')
nm = np.array(nm)
nd = nm[:,0]

training = pd.read_csv(r'E:\BAHRIA UNIVERSITY (BSE 3A)\5th semester\AI lab\projects\AI-Healthcare-chatbot-final\Training.csv')
testing  = pd.read_csv(r'E:\BAHRIA UNIVERSITY (BSE 3A)\5th semester\AI lab\projects\AI-Healthcare-chatbot-final\Testing.csv')
cols     = training.columns
cols     = cols[:-1]
x        = training[cols]
y        = training['prognosis']
y1       = y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
#print(clf.score(x_train,y_train))
#print ("cross result========")
#scores = cross_validation.cross_val_score(clf, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())
#print(clf.score(testx,testy))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

#feature_importances
#for f in range(10):
#    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))
def stopWords(text):
    #text is a sentence
    stopw = set(stopwords.words('english'))
    filtered = []
    words = word_tokenize(text)
    for i in words:
        if i not in stopw:
            filtered.append(i)
    return filtered

def stemming(text):
    #text could be a sent or word
    ps = PorterStemmer()
    empty = []
    for w in text:
        empty.append(w)
    return empty
def greet():
    k = random.randint(0,50)
    print(gd[k%11])

def askName():
    k = random.randint(0,50)
    print(nd[k%7])
    inp = input()
    return inp
def getName(text):
    filtered = stopWords(text)
    stemmed = stemming(filtered)
    tag = nltk.pos_tag(stemmed)
    noun=[]
    for i in range(len(tag)):
        if ((str(tag[i][1])=='NN' or str(tag[i][1])=='NNP') and str(tag[i][0])!='name'):
            noun.append(tag[i][0])
    return noun

greet()
print('I\'m MedBot, your personal health assistant.')
print("I can help you find out what's going on with a simple symptom assisment.")
ufName = askName()
name = getName(ufName)
print("Please reply Yes or No for the following symptoms") 
def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    #print(val)
    disease = le.inverse_transform(val[0])
    return disease
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    #print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    symptoms_present = []
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(name + " ?")
            ans = input()
            ans = ans.lower()
            if ans == 'yes':
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            print( "You may have " +  present_disease)
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("symptoms present  " + str(list(symptoms_present)))
            print("symptoms given "  +  str(list(symptoms_given)) )  
            confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            print(" confidence level is " + str(confidence_level))

    recurse(0, 1)

tree_to_code(clf,cols)

