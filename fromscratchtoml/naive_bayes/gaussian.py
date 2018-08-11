
# coding: utf-8

# In[51]:


import numpy as np
import copy
from collections import OrderedDict


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
            
            prob_dict[col_name]= {}
            
            for unique_feature in np.unique(full_data[:,col_id]):
                prob_dict[col_name][unique_feature] = {}
                
                get_rows = temp_data[temp_data[:,0]==unique_feature]
                get_rows = get_rows.astype(int)
                
                class_1 = (len(get_rows[np.where(get_rows[:,1] == 1)])*1.0)/n_samples_yes
                class_0 = (len(get_rows[np.where(get_rows[:,1] == 0)])*1.0)/n_samples_no
                
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

            if yes_prob>no_prob:
                results.append(1)
            else:
                results.append(0)
        return results
        
    

