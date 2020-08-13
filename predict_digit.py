import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("my_digit_recognizer.model")

def predict(test):
	resized_img = tf.image.resize(test,[28,28])	
	x = tf.keras.preprocessing.image.img_to_array(resized_img,data_format='channels_first',dtype='float32')	
	prediction = model.predict(x)
	output = np.argmax(prediction)
	print(output)
	return output