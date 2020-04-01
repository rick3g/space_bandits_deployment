import logging
import os
from time import strftime, gmtime
import boto3
import pandas as pd
from six import StringIO



s3 = boto3.client('s3')
db = boto3.resource('dynamodb')
smclient = boto3.Session().client('sagemaker')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_bucket = os.environ['S3_BUCKET']
s3_base_path = "s3://{}".format(s3_bucket)

datasource_table = os.environ['DATASOURCE_TABLE']
model_table = os.environ['MODEL_TABLE']


#Ensure that the train and validation data folders generated above are reflected in the "InputDataConfig" parameter below.
def create_training_params(hyperparams,s3_data_spec, training_job_name):
    hyperparams['sagemaker_submit_directory'] = os.path.join(s3_base_path,"bandits_train_deploy/src.tar.gz")
    hyperparams['sagemaker_program'] = "train.py"
    hyperparams['context_dim'] = str(s3_data_spec["ContextDim"])
    hyperparams['action_dim'] = str(s3_data_spec['ActionDim'])
    common_training_params = \
    {
        "TrainingJobName": training_job_name,
        "AlgorithmSpecification": {
            "TrainingImage": os.environ['CONTAINER_URI'],
            "TrainingInputMode": "File"
        },
        "RoleArn": os.environ['LAMBDA_ROLE'],
        "OutputDataConfig": {
            "S3OutputPath": s3_base_path
        },
        "ResourceConfig": {
            "InstanceCount": 1,   
            "InstanceType": "ml.m4.xlarge",
            "VolumeSizeInGB": 5
        },
        "HyperParameters": hyperparams,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_data_spec["DataLocationS3"],
                        "S3DataDistributionType": "FullyReplicated" 
                    }
                },
                "ContentType": "text/csv",
                "CompressionType": "None"
            }
        ]
    }
    return common_training_params


def get_datasource(id):
    table = db.Table(datasource_table)
    response = table.get_item(Key={'DataSourceId': id})
    item = response['Item']
    return item

class Model():
    def __init__(self, name, id, datasource_id, status, model_file, training_job_arn):
        self.name = name
        self.datasource_id = datasource_id
        self.status = status 
        self.model_file = model_file
        self.training_job_arn = training_job_arn
        self.id = id
    
    def format_json(self):
        return {
            "ModelName":self.name,
            "ModelId":self.id,
            "DataSourceId":self.datasource_id,
            "Status":self.status,
            "ModelWeightsPath":self.model_file,
            "TrainingJobArn":self.training_job_arn

        }

def handler(event,context):
    logger.info('Received event: {}'.format(event))
    try:
        data_source = get_datasource(event['TrainingDataSourceId'])
        
        logger.info('Retrieved Datasource: {}'.format(data_source))
        training_job_name = '{}-{}'.format(event['ModelName'], strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
        
        training_params = create_training_params(event['Parameters'],data_source['S3DataSpec'],training_job_name)
        logger.info('Training Params: {}'.format(training_params))
        
        response = smclient.create_training_job(**training_params)
        job_arn = response['TrainingJobArn']

        status = smclient.describe_training_job(TrainingJobName=training_job_name)
        logger.info("Status of training: {}".format(status))

        model_filepath = os.path.join(s3_base_path, training_job_name, "output", "model.tar.gz")

        model = Model(event["ModelName"], event["ModelId"], event["TrainingDataSourceId"], status["TrainingJobStatus"], model_filepath, job_arn)
        # smclient.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=training_job_name)
        # status = smclient.describe_training_job(TrainingJobName=training_job_name)['TrainingJobStatus']
        # logger.info("Training job ended with status: " + status)
        # if status == 'Failed':
        #     message = smclient.describe_training_job(TrainingJobName=training_job_name)['FailureReason']
        #     print('Training failed with the following error: {}'.format(message))
        #     raise Exception('Training job failed')
        
        model_json =  model.format_json()

        table = db.Table(model_table)
        table_response = table.put_item(
            Item=model_json
        )
        logger.info("Model Table Response: {}".format(table_response))

        model_json["statusCode"] = 200
        return model_json

    except Exception as e:
        logger.exception("Got an error during lambda function: {}".format(e))
        return {"Error": e, "statusCode": 400}
