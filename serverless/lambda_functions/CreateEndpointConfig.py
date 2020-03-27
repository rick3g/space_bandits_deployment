
# coding: utf-8

# In[ ]:


import json
import boto3
import csv
import io
import os

client = boto3.client('sagemaker')

def lambda_handler(event, context): 
    
    try:  
        
        #get info
        EndpointConfigName = event.get('EndpointConfigName')
        ProductionVariants = event.get('ProductionVariants')
        
        #get endpoint confid
        response = client.create_endpoint_config(
            EndpointConfigName='string',
            ProductionVariants=[
                {
                    'VariantName1': 'VariantName1',
                    'ModelName': 'string',
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium'|'ml.t2.large'|'ml.t2.xlarge'|'ml.t2.2xlarge'|'ml.m4.xlarge'|'ml.m4.2xlarge'|'ml.m4.4xlarge'|'ml.m4.10xlarge'|'ml.m4.16xlarge'|'ml.m5.large'|'ml.m5.xlarge'|'ml.m5.2xlarge'|'ml.m5.4xlarge'|'ml.m5.12xlarge'|'ml.m5.24xlarge'|'ml.m5d.large'|'ml.m5d.xlarge'|'ml.m5d.2xlarge'|'ml.m5d.4xlarge'|'ml.m5d.12xlarge'|'ml.m5d.24xlarge'|'ml.c4.large'|'ml.c4.xlarge'|'ml.c4.2xlarge'|'ml.c4.4xlarge'|'ml.c4.8xlarge'|'ml.p2.xlarge'|'ml.p2.8xlarge'|'ml.p2.16xlarge'|'ml.p3.2xlarge'|'ml.p3.8xlarge'|'ml.p3.16xlarge'|'ml.c5.large'|'ml.c5.xlarge'|'ml.c5.2xlarge'|'ml.c5.4xlarge'|'ml.c5.9xlarge'|'ml.c5.18xlarge'|'ml.c5d.large'|'ml.c5d.xlarge'|'ml.c5d.2xlarge'|'ml.c5d.4xlarge'|'ml.c5d.9xlarge'|'ml.c5d.18xlarge'|'ml.g4dn.xlarge'|'ml.g4dn.2xlarge'|'ml.g4dn.4xlarge'|'ml.g4dn.8xlarge'|'ml.g4dn.12xlarge'|'ml.g4dn.16xlarge'|'ml.r5.large'|'ml.r5.xlarge'|'ml.r5.2xlarge'|'ml.r5.4xlarge'|'ml.r5.12xlarge'|'ml.r5.24xlarge'|'ml.r5d.large'|'ml.r5d.xlarge'|'ml.r5d.2xlarge'|'ml.r5d.4xlarge'|'ml.r5d.12xlarge'|'ml.r5d.24xlarge'|'ml.inf1.xlarge'|'ml.inf1.2xlarge'|'ml.inf1.6xlarge'|'ml.inf1.24xlarge',
                },
            ]
        )
        
        
        return {
            'statusCode': 200,
            'EndpointConfigArn': 'string'
        }
    
    except: 
        
        return {
            'statusCode':400,
            'body': 'Error, bad parameters'
        }

