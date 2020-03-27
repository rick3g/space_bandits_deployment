
# coding: utf-8

# In[ ]:


import json
import boto3
import csv
import io
import os
import pickle
import uuid

s3_client = boto3.client('s3')

def lambda_handler(event, context): 
    
    try:
        
        # get info
        EvaluationDataSourceId = event.get(DataSourceId)
        MLModelID = event.get(MLModelID)
    
        # load model 
        bucket = os.environ['S3_BUCKET']
        key = 'model.pckl'
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
        s3_client.download_file(bucket, key, download_path)
        
        f = open(download_path, 'rb')
        model = pickle.load(f)
        f.close()
        
        # get model score 
        result = model.get_sscore() 
         
        return {
            'statusCode': 200,
            'body': result
        }
    
    except: 
        return {
            'statusCode':400,
            'body': 'Error, bad parameters'
        }

