
# coding: utf-8

# In[ ]:


import json
import boto3
import csv
import io

s3_client = boto3.client('s3')

def lambda_handler(event, context): 
    
    try:
    
        EvaluationDataSourceId = event.get(DataSourceId)
        MLModelID = event.get(MLModelID)
        
        body = event['body']
        model = load_model()
        result = model.get_sscore(body)  
        
        return {
            'statusCode': 200,
            'body': result
        }
    
    except: 
        
        return {
            'statusCode':400,
            'body': 'Error, bad parameters'
        }

