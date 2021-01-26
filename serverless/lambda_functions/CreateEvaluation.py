
# coding: utf-8

# In[ ]:


import json
import boto3
import csv
import io
import os
import pandas as pd
from space_bandits import load_model
# space_bandits is in s3 env.

s3_client = boto3.client('s3')

def lambda_handler(event, context): 
    
    try:
        
        # get info
        EvaluationDataSourceId = event.get(DataSourceId)
        MLModelID = event.get(MLModelID)
    
        # load model 
        bucket = os.environ['S3_BUCKET']
        model = load_model(bucket + '\' + MLModelID)
        # assuming data is in format contexts, actions, rewards
        data = pd.read_csv(bucket + '\' + EvaluationDataSourceId).to_numpy()
        contexts = data[:,:-2]
        actions = data[:,-2]
        rewards = data[:,-1]                           
                
        # get model score 
        result = model.get_sscore(contexts, actions, rewards) 
         
        return {
            'statusCode': 200,
            'body': result
        }
    
    except: 
        return {
            'statusCode':400,
            'body': 'Error, bad parameters'
        }

