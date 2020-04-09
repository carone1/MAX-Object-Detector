#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import netifaces as ni
import platform
import socket
import json
from maxfw.core import MAX_API, PredictAPI, CustomMAXAPI
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from core.model import ModelWrapper

model_label = MAX_API.model('ModelLabel', {
    'id': fields.String(required=True, description='Class label identifier'),
    'name': fields.String(required=True, description='Class label'),
})

labels_response = MAX_API.model('LabelsResponse', {
    'count': fields.Integer(required=True,
                            description='Number of class labels returned'),
    'labels': fields.List(fields.Nested(model_label),
                          description='Class labels that can be predicted by '
                                      'the model')
})

model_wrapper = ModelWrapper()


class ModelLabelsAPI(CustomMAXAPI):

    @MAX_API.doc('labels')
    @MAX_API.marshal_with(labels_response)
    def get(self):
        """Return the list of labels that can be predicted by the model"""
        return {
            'labels': model_wrapper.categories,
            'count': len(model_wrapper.categories)
        }


input_parser = MAX_API.parser()
input_parser.add_argument('image', type=FileStorage, location='files', required=True,
                          help='An image file (encoded as PNG or JPG/JPEG)')
input_parser.add_argument('threshold', type=float, default=0.7,
                          help='Probability threshold for including a detected object in the response in the range '
                               '[0, 1] (default: 0.7). Lowering the threshold includes objects the model is less '
                               'certain about.')


label_prediction = MAX_API.model('LabelPrediction', {
    'label_id': fields.String(required=False, description='Class label identifier'),
    'label': fields.String(required=True, description='Class label'),
    'probability': fields.Float(required=True, description='Predicted probability for the class label'),
    'detection_box': fields.List(fields.Float(required=True), description='Coordinates of the bounding box for '
                                                                          'detected object. Format is an array of '
                                                                          'normalized coordinates (ranging from 0 to 1'
                                                                          ') in the form [ymin, xmin, ymax, xmax].')
})

environment_variables = MAX_API.model('EnvironmentVariables', {
    'name': fields.String(required=False, description='Environemnt variable name'),
    'value': fields.String(required=False, description='Environment variable value')
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(label_prediction),
                               description='Predicted class labels, probabilities and bounding box for each detected '
                                           'object'),
    'environment_variables': fields.List(fields.Nested(environment_variables), description='Environment Variables')
})


class ModelPredictAPI(PredictAPI):

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        threshold = args['threshold']
        image_data = args['image'].read()
        image = model_wrapper._read_image(image_data)
        label_preds = model_wrapper._predict(image, threshold)

        result['predictions'] = label_preds
        result['status'] = 'ok'

        environment_variables = []
        environment_network = {}

        for k, v in os.environ.items():
            # only display certain env variable
            if "CLUSTER_NAME" in k or "CLOUD_PROVIDER" in k or "VERSION" in k :
                environment_variable = {'name':k, 'value':v}
                environment_variables.append(environment_variable)

        result['environment_variables'] = environment_variables

        # logging
        log_predictions = os.environ['LOG_PREDICTIONS']
        log_predictions_file = os.environ['LOG_PREDICTIONS_FILE']
        if "TRUE" in log_predictions :
            if os.path.exists(log_predictions_file):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            
            with open(log_predictions_file, append_write) as outfile:
                result_json = json.dumps(str(result))
                outfile.write(result_json)
                outfile.close()

        return result
