#!/usr/bin/env python
# coding: utf-8

# In[218]:


import numpy as np
import pandas as pd
import string
import re
import nltk
import random
from sklearn.svm import SVC
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize

class AuthorClassifier:
    
    vectorize = None
    vectorized_train_data = None
    train_data_labels = None
    
    def preprocess_data(self,train_data_frm):
        
        ''' dropping first column of indices'''
        train_data_frm = train_data_frm.drop(train_data_frm.columns[0], axis='columns')
        
        rows,cols = train_data_frm.shape
               
        ''' removing punctuations'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            fval = val.translate(val.maketrans('','', string.punctuation))
            train_data_frm.at[i, 'text'] = fval
            
        ''' removing numbers'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            fval = re.sub(r'\d+', '', val)
            train_data_frm.at[i, 'text'] = fval
            
        ''' converting to lowercase'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            fval = val.lower()
            train_data_frm.at[i, 'text'] = fval
            
        '''removing stop words'''
        stop_words = set(stopwords.words('english'))
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            tokens = word_tokenize(val)
            fval = [i for i in tokens if not i in stop_words]
            ffval = " "
            train_data_frm.at[i, 'text'] = ffval.join(fval)
            
        '''removing whitespaces'''
        for i, rows in train_data_frm.iterrows():
            val = train_data_frm.iat[i,0]
            train_data_frm.at[i, 'text'] = val.strip()
                                  
        return train_data_frm
    
    def train_validation_split(self,data_frm,validation_data_size):
       
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(data_frm))
        indices=data_frm.index.tolist()
        valid_indices=random.sample(indices, validation_data_size)
        valid_datafrm=data_frm.loc[valid_indices]
        train_datafrm=data_frm.drop(valid_indices)
        return train_datafrm, valid_datafrm
    
    
    def prepare_data(self,data_frm):
        data_labels = data_frm.iloc[:,-1]
        data_frm = data_frm.iloc[:,:-1]
        return data_frm, data_labels
    
    
    def run_svm(self, vectorized_validation_data, validation_data_labels):
        ''' kernel linear function '''
        obj = SVC(kernel='linear', decision_function_shape='ovr', C=10)
              
        obj.fit(self.vectorized_train_data, self.train_data_labels)
        predict_values = obj.predict(vectorized_validation_data)
#         print(predict_values)
#         print(validation_data_labels)
#         print(accuracy_score(predict_values, validation_data_labels))

    
    
    def check_validation_data(self,train_data_frm, validation_data_frm):
        validation_data_frm, validation_data_labels = self.prepare_data(validation_data_frm)
        validation_data = validation_data_frm.values.flatten()
        vectorized_validation_data = self.vectorize.transform(validation_data)
#         print(vectorized_validation_data)
        validation_data_labels = validation_data_labels.values.flatten()
        self.train_data_labels = self.train_data_labels.values.flatten()
        self.run_svm(vectorized_validation_data, validation_data_labels)
#         print(vectorized_validation_data.shape)
        
    
    def train(self,train_path):
        train_data_frm = pd.read_csv(train_path)
        train_data_frm = self.preprocess_data(train_data_frm)
        
        random.seed(0)
        train_data_frm, validation_data_frm = self.train_validation_split(train_data_frm, validation_data_size = 0.3)
        train_data_frm, self.train_data_labels = self.prepare_data(train_data_frm)
        self.vectorize = TfidfVectorizer()
        train_data_frm = train_data_frm.values.flatten()
        self.vectorized_train_data = self.vectorize.fit_transform(train_data_frm)
#         self.check_validation_data(train_data_frm, validation_data_frm)
        
    def predict(self, test_path):
        test_data_frm = pd.read_csv(test_path)
        test_data_frm = self.preprocess_data(test_data_frm)
        test_data_frm = test_data_frm.values.flatten()
        vectorized_test_data = self.vectorize.transform(test_data_frm)
        ''' kernel linear function '''
        obj = SVC(kernel='linear', decision_function_shape='ovr', C=10)
        
#         ''' kernel poly'''
#         obj = SVC(kernel='poly', degree = 4, C = 10)
        
#         ''' kernel rbf'''
#         obj = SVC(kernel='rbf', C = 100)
        
#         ''' kernel sigmoid'''
#         obj = SVC(kernel='sigmoid', C = 100)
        
        obj.fit(self.vectorized_train_data, self.train_data_labels)
        predict_values = obj.predict(vectorized_test_data)
        return predict_values
        
        
        


# In[219]:


# auth_classifier = AuthorClassifier()
# auth_classifier.train('/home/jyoti/Documents/SMAI/assign2/Q5/Question-5/Train.csv')
# predictions = auth_classifier.predict('/home/jyoti/Documents/SMAI/assign2/Q5/Question-5/Test.csv')
# print(predictions)


# In[ ]:





# In[ ]:




