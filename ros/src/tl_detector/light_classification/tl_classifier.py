import json
import operator
import cv2
import numpy as np
from keras.preprocessing import image
from keras.optimizers import Adam

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from styx_msgs.msg import TrafficLight
import tensorflow as tf

def get_model():

    with open('model.json', 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))
	
    model.compile(optimizer = Adam(lr = 0.0001), loss='mse', metrics=['accuracy'])
    model.load_weights('model.h5')

    return model
	
class TLClassifier(object):
    def __init__(self): 
        self.model = get_model()
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = cv2.resize(image, (128, 96), interpolation = cv2.INTER_CUBIC)
        transformed_image_array = img[None, :, :, :]
        #p = self.model.predict(transformed_image_array)
        with self.graph.as_default():
            p = self.model.predict(transformed_image_array)
        state, value = max(enumerate(p[0]), key=operator.itemgetter(1))
		        
        return state
