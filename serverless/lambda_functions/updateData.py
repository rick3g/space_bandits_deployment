import boto3
import os
import pandas as pd
import logging
from six import StringIO

s3 = boto3.client('s3')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_bucket = os.environ['S3_BUCKET']



def handler(event, context):
    logger.info('got event{}'.format(event))
    #payload = json.loads(event['body'])
    data = event['data']
    key = event['key']

    logger.info('Data: {}'.format(data))
    logger.info('Key: {}'.format(key))
    df = pd.DataFrame.from_dict(data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)

    s3_base_path = "s3://{}".format(s3_bucket)
    file_path = os.path.join(s3_base_path,key)

    #Add a file to your Object Store
    response = s3.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=csv_buffer.getvalue()
    )

    logger.info("RESPONSE FROM S3: {}".format(response))
    
    status = response['ResponseMetadata']['HTTPStatusCode']
    
    logger.info("STATUS: {}".format(status))
    
    return {'statusCode': status,
            'path': file_path}
    
