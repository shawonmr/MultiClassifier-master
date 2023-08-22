
# coding: utf-8

# In[246]:


#import necessary libs

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#read the data
data = pd.read_csv('seed.csv')
#print(data)


# In[247]:


#Creating the dependent variable class equ to numeric value
factor = pd.factorize(data['_Category'])
data._Category = factor[0]
definitions = factor[1]
#print(data._Category.head())
#print(definitions)


# In[248]:


#Split train and test data 70-30 ratio
X_train, X_valid, Y_train, Y_valid = train_test_split(data['_Body'],data['_Category'],train_size = 0.70, test_size = 0.30)
#print(X_train, Y_train)


# In[249]:


#Vector feature count
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
#print(X_train_vectorized)


# In[250]:


#Logisitic model analysis
model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=500)
#print(max_value)
#print(Y_train)
model.fit(X_train_vectorized, Y_train)


# In[251]:


#print(result)


# In[252]:


#vect = CountVectorizer().fit(X_valid)
X_valid_vectorized = vect.transform(X_valid)

#predictions = model.predict_proba(X_valid_vectorized)
#print(predictions)
#Y_valid = Y_valid*10 
print(model.score(X_valid_vectorized,Y_valid))


# In[253]:





# save the model to disk
joblib.dump(model,'model.joblib')
# save CountVectorizer() to disk
joblib.dump(vect,'vect.joblib')


# In[254]:


# Analysis with Tf-idf vectorizer
vectorizer = TfidfVectorizer()

vect_tf_idf = vectorizer.fit(X_train)
X_train_vec_tf_idf = vect_tf_idf.transform(X_train)
model.fit(X_train_vec_tf_idf, Y_train)
print(model.score(vect_tf_idf.transform(X_valid),Y_valid))
#dump Tfid
joblib.dump(vectorizer,'vect_tf_idf.joblib')


# In[255]:


# training a Naive Bayes classifier 

nbv = GaussianNB()
nbv = nbv.fit(X_train_vectorized.toarray(), Y_train) 
print(nbv.score(vect.transform(X_valid).toarray(),Y_valid))
joblib.dump(nbv,'nbv.joblib')


# In[256]:


nb_tf_idf = nbv.fit(X_train_vec_tf_idf.toarray(), Y_train) 
print(nb_tf_idf.score(vect_tf_idf.transform(X_valid).toarray(),Y_valid))


# In[257]:


# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 53)
classifier.fit(X_train_vectorized, Y_train)
# Predicting the valid set results
Y_pred = classifier.predict(X_valid_vectorized)
print(accuracy_score(Y_valid, Y_pred))
joblib.dump(classifier,'classifier.joblib')


# In[258]:


#Fitting  tf-idf features
classifier.fit(X_train_vec_tf_idf, Y_train)
#Predicitng the valid test results
Y_pred = classifier.predict(vect_tf_idf.transform(X_valid))
print(accuracy_score(Y_valid, Y_pred))


# In[259]:


#Jaccard similarity to find out similarity between two texts
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

#token1 = data.loc[0,'_Body']
#token2 = input_data.loc[0,'_Body']
#print(get_jaccard_sim(token1,token2))


# In[260]:


#read input data
input_data = pd.read_csv('input_data.csv')
#print(input_data)
input_data['_Category'] = 0
data['_Body'] = data['_Body'].astype(str) #Convert to string 
input_data['_Body'] = input_data['_Body'].astype(str) #Convert to string

#Categorize body of input data by jaccard similarity 
for i in range(len(input_data)):
    max_value = -1
    for j in range(len(data)):
        js = get_jaccard_sim(input_data.loc[i,'_Body'],data.loc[j,'_Body'])
        if max_value < js:
            max_value = js
            input_data.loc[i,'_Category'] = data.loc[j,'_Category']
        if max_value == 1:
            input_data.loc[i,'_Category'] = data.loc[j,'_Category']
            break
#print(input_data['Category_Value'])


# In[261]:


#Test data from the input.csv
testX = input_data['_Body']
testY = input_data['_Category']


# In[262]:


#print(testX)
# load the model from disk
loaded_model = joblib.load('model.joblib')
# load the CountVectorizer()
vect = joblib.load('vect.joblib')
#transform the test data to the corresponding vectorizer
X_train_vectorized = vect.transform(testX)

loaded_model.predict(X_train_vectorized)
loaded_model.score(X_train_vectorized,testY)
#loaded_model.close()


# In[263]:



#load the tf-idf vectorizer
vect_tf_idf = joblib.load('vect_tf_idf.joblib')
X_train_vec_tf_idf = vect_tf_idf.transform(testX)
#loaded_model.predict(X_train_vec_tf_idf)
loaded_model.score(X_train_vec_tf_idf,testY)
#loaded_model.close()
#model.fit(X_train_vec_tf_idf, Y_train)
#model.predict(X_train_vec_tf_idf)


# In[264]:


#Naive Bayes classifier analysis with the test data

nbv = joblib.load('nbv.joblib')
print(nbv.score(X_train_vectorized.toarray(),testY))
print(nbv.score(X_train_vec_tf_idf.toarray(),testY))


# In[265]:


#load the random forest classifier
classifier = joblib.load('classifier.joblib')
Y_pred = classifier.predict(X_train_vectorized)
print(accuracy_score(testY, Y_pred))


# In[266]:


# random forest analysis with tf-idf vectorizer()
Y_pred = classifier.predict(X_train_vec_tf_idf)
print(accuracy_score(testY, Y_pred))

