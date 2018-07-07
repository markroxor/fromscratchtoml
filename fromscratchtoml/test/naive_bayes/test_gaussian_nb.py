
# coding: utf-8

# In[10]:


from fromscratchtoml.naive_bayes import GaussianNB
from unittest import TestCase
import numpy as np


# In[9]:


def TestGaussianNB(TestCase):
    def testPredictions(self):
        my_data = np.genfromtxt('data.csv', delimiter=',',dtype=str)
        clf = GaussianNB()
        clf.fit(my_data)
        my_preds = clf.predict([[0,0,0,0], [1,1,1,0], [1,2,0,0], [2,1,1,1], [0,1,1,0], [1,2,0,0], [2,0,1,1]])

        X = my_data[1:]
        X = np.delete(X,0,axis=1)
        y = X[:,-1].astype(int)
        X = np.delete(X,-1,axis=1).astype(int)
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(X, y)
        sklearn_predicts =  gnb.predict([[0,0,0,0], [1,1,1,0], [1,2,0,0], [2,1,1,1], [0,1,1,0], [1,2,0,0], [2,0,1,1]])
        self.assertTrue(np.allclose(sklearn_predicts, my_preds)) 


