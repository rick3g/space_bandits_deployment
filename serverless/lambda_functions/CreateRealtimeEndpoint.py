
# coding: utf-8

# In[1]:


iimport json
import boto3
import csv
import io
import os

client = boto3.client('sagemaker')

def lambda_handler(event, context): 
    
    try:
        
        #get info
        EndpointName = event.get('EndpointName')
        EndpointConfigName = event.get('EndpointConfigName')
        
        
        #create endpoint
        response = client.create_endpoint(
            EndpointName='string',
            EndpointConfigName='string'
            
        )
        
        return {
            'statusCode': 200,
            'EndpointArn': 'string'
        }
    
    except: 
        
        return {
            'statusCode':400,
            'body': 'Error, bad parameters'
        }

