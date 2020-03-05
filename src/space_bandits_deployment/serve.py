import logging
import os
import time
import torch
from space_bandits import load_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# loads the model into memory from disk and returns it
def model_fn(model_dir):
    logger.info('model_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = load_model(os.path.join(model_dir, 'model.pickle'))

    return model

def _npy_loads(data):
    """
    Deserializes npy-formatted bytes into a numpy array
    """
    stream = BytesIO(data)
    stream.seek(0) 
    return np.load(stream)


def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


# def input_fn(input_bytes, content_type):
#     """This function is called on the byte stream sent by the client, and is used to deserialize the
#     bytes into a Python object suitable for inference by predict_fn -- in this case, a NumPy array.
#     This implementation is effectively identical to the default implementation used in the Chainer
#     container, for NPY formatted data. This function is included in this script to demonstrate
#     how one might implement `input_fn`.
#     Args:
#         input_bytes (numpy array): a numpy array containing the data serialized by the Chainer predictor
#         content_type: the MIME type of the data in input_bytes
#     Returns:
#         a NumPy array represented by input_bytes.
#     """
#     if content_type == 'application/x-npy':
#         return _npy_loads(input_bytes)
#     elif content_type == 'application/numpy':
#         return input_bytes
#     else:
#         raise ValueError('Content type must be application/x-npy')


# Use default input_fn implemented here: https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/default_inference_handler.py

def predict_fn(input_data, model):
    '''
    Predicts the best action to take for given input contexts
    Args:
        input_data: torch tensor of contexts
        model: the bandit model loaded by model_fn
    Returns:
        actions: torch tensor of the predicted actions for each context
    '''
    logger.info("Calling model")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    actions = model.predict(input_data)
    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print("The return type from predict_fn is: {}".format(actions))
    return actions

#Serialize the prediction result into the desired response content type
# def output_fn(prediction, accept=JSON_CONTENT_TYPE):
#     logger.info('Serializing the generated output.')
#     if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
#     raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))

    
# def output_fn(prediction_output, accept):
#     """This function is called on the return value of predict_fn, and is used to serialize the
#     predictions back to the client.
    
#     This implementation is effectively identical to the default implementation used in the Chainer
#     container, for NPY formatted data. This function is included in this script to demonstrate
#     how one might implement `output_fn`.
#     Args:
#         prediction_output (numpy array): a numpy array containing the data serialized by the Chainer predictor
#         accept: the MIME type of the data expected by the client.
#     Returns:
#         a tuple containing a serialized NumPy array and the MIME type of the serialized data.
#     """
#     print("The input output_fn is: {}".format(prediction_output))
#     print("The accept type output_fn is: {}".format(accept))
#     if accept == 'application/x-npy':
#         return _npy_dumps(prediction_output), 'application/x-npy'
#     else:
#         raise ValueError('Accept header must be application/x-npy')
# Use default output_fn implemented here: https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/default_inference_handler.py
