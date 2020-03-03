import logging
import os
import time
import torch
from space_bandits import load_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# loads the model into memory from disk and returns it
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, 'model.pickle'), 'rb') as f:
        model = load_model(f)

    return model.to(device)

# Deserialize the Invoke request body into an object we can perform prediction on
# from six import BytesIO

# def input_fn(request_body, request_content_type):
#     """An input_fn that loads a pickled tensor"""
#     if request_content_type == 'application/python-pickle':
#         return torch.load(BytesIO(request_body))
#     else:
#         # Handle other content-types here or raise an Exception
#         # if the content type is not supported.
#         pass


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
    model.to(device)
    actions = model.predict(input_data)
    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    return torch.Tensor(actions)

# Serialize the prediction result into the desired response content type
# def output_fn(prediction, accept=JSON_CONTENT_TYPE):
#     logger.info('Serializing the generated output.')
#     if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
#     raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))

# Use default output_fn implemented here: https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/default_inference_handler.py
