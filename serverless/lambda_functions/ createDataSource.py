import boto3
import os
import pandas as pd
import logging
from six import StringIO

s3 = boto3.client('s3')
db = boto3.client('dynamodb')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_bucket = os.environ['S3_BUCKET']
db_table = os.environ['DATASOURCE_TABLE']


class DataSpec:
    def __init__(self, path, context_cols, reward_cols):
        self.path = path 
        self.context_cols = context_cols
        self.reward_cols = reward_cols
    
    


class DataSource:
    def __init__(self, id, name, stats=False, data_spec):
        self.id = id
        self.path = path
        self.name = name
        self.stats = stats
        self.data_spec = data_spec
    
    def compute_stats(self):
        pass



def handler(event,context):
    logger.info('got event: {}'.format(event))

    return {
        'statusCode': 200
        'DataSourceId': 'id'
    }