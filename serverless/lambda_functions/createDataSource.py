import logging
import os

import boto3
import pandas as pd
from six import StringIO

s3 = boto3.client('s3')
db = boto3.resource('dynamodb')

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
    def __init__(self, id, name, data_spec, stats=False):
        self.id = id
        self.path = path
        self.name = name
        self.stats = stats
        self.data_spec = data_spec

    def compute_stats(self):
        pass



def handler(event,context):
    logger.info('got event: {}'.format(event))

    data_spec = event['S3DataSpec']
    table = db.Table(db_table)
    response = table.put_item(
        Item=event
    )

    logger.info('Response from DB: {}'.format(response))
    status = response['ResponseMetadata']['HTTPStatusCode']

    return {
        'statusCode': status,
        'DataSourceId': event['DataSourceId']
    }
