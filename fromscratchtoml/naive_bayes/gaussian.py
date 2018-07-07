
# coding: utf-8

# In[51]:


import numpy as np
import copy
from collections import OrderedDict


# In[72]:


my_data = np.genfromtxt('data.csv', delimiter=',',dtype=str)


# In[73]:


class GaussianNB():
    def __init__(self):
        self.prior_dict = {}

    def fit(self, full_data):
        columns = full_data[0:1][0]
        full_data = full_data[1:].astype(int)
        label_columns = full_data[:,-1]
        n_samples_yes = len(label_columns[label_columns==1])
        n_samples_no = len(label_columns[label_columns==0])
        col_id = 1
        prob_dict = OrderedDict()
        prob_dict[1]= ((n_samples_yes)*1.0)/len(full_data)
        prob_dict[0]= ((n_samples_no)*1.0)/len(full_data)
        for col_name in columns[1:-1]:
            temp_data = np.dstack([full_data[:,col_id],label_columns[:]])[0]
    #         print temp_data
            prob_dict[col_name]= {}
            for unique_feature in np.unique(full_data[:,col_id]):
    #             print "unique f",unique_feature,['Rainy','overcast','sunny'][unique_feature]
                prob_dict[col_name][unique_feature] = {}
    #             print unique_feature
                get_rows = temp_data[temp_data[:,0]==unique_feature]
                get_rows = get_rows.astype(int)
    #             print get_rows
                #class_1 = str((len(get_rows[np.where(get_rows[:,1] == 1)])*1.0))+'//'+str(n_samples_yes)
                class_1 = (len(get_rows[np.where(get_rows[:,1] == 1)])*1.0)/n_samples_yes
    #             print class_1
                #class_0 = str(len(get_rows[np.where(get_rows[:,1] == 0)])*1.0)+'//'+str(n_samples_no)
                class_0 = (len(get_rows[np.where(get_rows[:,1] == 0)])*1.0)/n_samples_no
    #             print class_0
                prob_dict[col_name][unique_feature][0] = class_0
                prob_dict[col_name][unique_feature][1] = class_1
            col_id+=1

        self.prior_dict =  copy.deepcopy(prob_dict)
    def predict(self, to_predict):
        results = []
        for sample_input in to_predict:
            yes_prob = 1.0
            no_prob = 1.0
            for f_id,feature_names in enumerate(self.prior_dict.keys()[2:]):
                yes_prob *= self.prior_dict[feature_names][sample_input[f_id]][1]
                no_prob *= self.prior_dict[feature_names][sample_input[f_id]][0]

            yes_prob *=self.prior_dict[1]
            no_prob *=self.prior_dict[0]

    #         if yes_prob>no_prob:
    #             print "Play Golf"
    #         else:
    #             print "Dont play Golf"
            if yes_prob>no_prob:
                results.append(1)
            else:
                results.append(0)
        return results
        
    


# In[69]:


clf = GaussianNB()
clf.fit(my_data)
my_preds = clf.predict([[0,0,0,0], [1,1,1,0], [1,2,0,0], [2,1,1,1], [0,1,1,0], [1,2,0,0], [2,0,1,1]])


# In[55]:


# load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
 
# store the feature matrix (X) and response vector (y)
X = my_data[1:]
X = np.delete(X,0,axis=1)
y = X[:,-1].astype(int)
X = np.delete(X,-1,axis=1).astype(int)
print X,y
# # splitting X and y into training and testing sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
 
# # training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, y)
 
# # making predictions on the testing set
# y_pred = gnb.predict(X_test)
 
# # comparing actual response values (y_test) with predicted response values (y_pred)
# from sklearn import metrics
# print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[71]:


sklearn_predicts =  gnb.predict([[0,0,0,0], [1,1,1,0], [1,2,0,0], [2,1,1,1], [0,1,1,0], [1,2,0,0], [2,0,1,1]])
print np.allclose(sklearn_predicts, my_preds)

