
# coding: utf-8

# In[2]:


import nltk
import numpy as np
from sklearn.metrics import make_scorer
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import scipy

sno = nltk.stem.SnowballStemmer('english')
import warnings
warnings.filterwarnings(action='ignore')


# In[3]:


file=r"ner.txt"


# In[4]:


#reading data
with open(file, 'r') as f:
    data=f.readlines()


# In[5]:


#creating list of list of tuples
docs=[]
doc=[]
for sent in data:
    if(len(sent)==1):
        docs.append(doc)
        doc=[]
    else:
        word1,word2=sent.split()
        word_tuple=(word1,word2[-2:])
        doc.append(word_tuple)


# In[6]:


print(docs[0])


# In[7]:



# Appending the POS tags
data=[]
for doc in docs:
    words = [word for word,label in doc ]
    pos_tags=nltk.pos_tag(words)
    data_sent=[]
    for i in range(len(pos_tags)):
        data_sent.append((doc[i][0],pos_tags[i][1],doc[i][1]))
    data.append(data_sent)
    
print(data[0])


# In[8]:


# features from word net 

def no_of_contexts(word):
    temp=0
    for syn in wn.synsets(word):
        temp+=1
    return temp

# if it is alphanumeric
def contain_digit(word):
    for ch in list(word):
        if ch.isdigit()==True:
            return True
    return False

   


# In[9]:


#print the report

def showreport(y_test,y_pred):
    label_dict = {"O": 0, "D": 1,"T":2}
   # creating predicted list of entities
    model_output=[]
    for row in y_pred:
        for entity in row:
            model_output.append(label_dict[entity])
    #creating true list of entities
    true_output=[]
    for row in y_test:
        for entity in row:
            true_output.append(label_dict[entity])       
    
    # Print out the classification report
    print(classification_report(true_output, model_output, target_names=["O", "D","T"]))


# In[10]:


def word_to_features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'suffix_3': word[-3:],
        #'suffix_2': word[-2:],
        'prefix_3':word[:3],
        'wordlen':len(word),
       'word.isupper': word.isupper(),
     'word.isdigit': contain_digit(word),
      'postag': postag,
        'no_of_contexts':no_of_contexts(word),
        'word_stem':sno.stem(word.lower())

      }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({

         
              '-1:wordlen': len(word1),
           '-1:word.isupper': word1.isupper(),
         '-1:word.isdigit': contain_digit(word1),
         '-1:postag': postag1,
            '-1:no_of_contexts':no_of_contexts(word1)
            
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({

          '+1:word.isupper': word1.isupper(),
               '+1:wordlen': len(word1),
           '+1:word.isdigit':contain_digit(word1),
           '+1:postag': postag1,
           '+1:no_of_contexts':no_of_contexts(word1)
            
        })
    else:
        features['EOS'] = True

    return features


# In[11]:


# convert words of each document into features reresented in form of dictionary
X=[]
Y=[]
for doc in data:
    X.append([word_to_features(doc,i) for i in range(len(doc))])
    final_y=[label for (word,pos_tag,label) in doc]
    Y.append(final_y)
    
    


# In[12]:


#splitting in ratio of 70:10:20
X_train, X_testanddev, y_train, y_testanddev= train_test_split(X, Y, test_size=0.3,random_state=4)

X_test,X_dev, y_test,y_dev = train_test_split(X_testanddev, y_testanddev, test_size=0.33,random_state=4)



# In[77]:


# hyper parameter tunning 
#code referred from crf suite examples


labels=["D","T","O"]

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', 
                           max_iterations=1000,
                           all_possible_transitions=True,
                           verbose=False)

params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}


f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='macro', labels=labels)

rs = RandomizedSearchCV(crf, params_space,
                        cv=10,
                        verbose=1,
                        n_jobs=1,
                        n_iter=20,
                        scoring=f1_scorer)
rs.fit(X_train, y_train)


# In[78]:




print('Best params:', rs.best_params_)
print('Best F-1 score:', rs.best_score_)


# In[14]:


# fitting the models with obtained hyperparameters c1=.055 and c=.066


crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.055,c2=0.066 ,
                           max_iterations=1000,
                           all_possible_transitions=True,
                           verbose=False)
crf.fit(X_train,y_train)
labels=["O","D","T"]

#predicting the entities for test data
y_pred=crf.predict(X_test)
print("F1 score for D, T and O label(average) is %lf "% (metrics.flat_f1_score(y_test, y_pred,
                      average='macro', labels=labels)))
#printing the classfication report
showreport(y_test,y_pred)

