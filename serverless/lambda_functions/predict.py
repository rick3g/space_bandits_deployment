import os
import io
import json
import csv
from six import BytesIO
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
runtime = boto3.client('runtime.sagemaker')

class Prediction():
    def __init__(self, actions, rewards):
        self.actions = actions
        self.rewards = rewards
    def format_response(self):
        return {"PredictedAction": self.actions,
                "PredictedReward": self.rewards}
                

class BadRequest():
    """Custom exception class to be thrown when local error occurs."""
    def __init__(self, error, message='bad input', status=400):
        self.message = message
        self.status = status
        self.error = error
    
    def format_response(self):
        return {"statusCode": self.status,
                "description": self.message,
                "error": self.error }

def _json_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a JSON object to a numpy array.
        Args:
            string_like (str): JSON string.
            dtype (dtype, optional):  Data type of the resulting array.
                If None, the dtypes will be determined by the contents
                of each column, individually. This argument can only be
                used to 'upcast' the array.  For downcasting, use the
                .astype(t) method.
        Returns:
            (np.array): numpy array
        """
    data = json.loads(string_like)
    return np.array(data, dtype=dtype)

def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()
    
def handler(event, context):
    logger.info('Received event: {}'.format(event))
    try:
        df = pd.DataFrame.from_dict(event['Records'])
        context_arr = df.to_numpy()
        payload = _npy_dumps(context_arr)

        response = runtime.invoke_endpoint(EndpointName=event['PredictEndpoint'],
                                        ContentType='application/x-npy',
                                        Body=payload)
        
        logger.info("RESPONSE ------ {}".format(response))
        
        result = json.loads(response['Body'].read().decode())
        logger.info("RESULT ------ {}".format(result))
        return result
    except Exception as e:
        r = BadRequest(e)
        return r.format_response()


    