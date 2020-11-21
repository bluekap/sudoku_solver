import cv2
import numpy as np
from predict_digit import predict,predict_with_image_output
import tensorflow as tf

# model = tf.keras.models.load_model("my_digit_recognizer_19_NOV.model")
model = tf.keras.models.load_model("my_digit_recognizer.model")

def predict_block(img):
	resized_img = cv2.resize(img,(28,28))
	resized_img = cv2.medianBlur(resized_img, 1)
	gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	inv_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
	inv_img = inv_img.reshape(1,28,28,1)
	inv_img = inv_img / 255
	prediction = model.predict(inv_img)
	output = np.argmax(prediction)
	print(prediction)
	print(output)
	return output

def preprocess_block(img):
	inv_img = remove_lines(img)
	# cv2.imshow("Processed",np.hstack((inv_img,cv2.medianBlur(inv_img, 1))))
	# cv2.waitKey(0)
	inv_img = cv2.medianBlur(inv_img, 1)
	inv_img = 255 - inv_img
	detect_blank(img)
	row,col = inv_img.shape	
	new_img = inv_img.reshape(1,28,28,1)
	new_img = new_img / 255
	prediction = model.predict(new_img)
	prediction[0][0] = 0
	output = np.argmax(prediction)
	print(prediction)
	print(output)

def remove_lines(img):
	resized_img = cv2.medianBlur(img, 1)
	gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	inv_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
	new_img = cv2.resize(inv_img,(28,28))

	horizontal_img = new_img.copy()
	vertical_img = new_img.copy()

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
	horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
	horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
	vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
	vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

	no_border = np.bitwise_or(255 - new_img, horizontal_img + vertical_img)
	cv2.imshow("Processed",no_border)
	cv2.waitKey(0)
	return no_border

def detect_blank(inv_img):
	resized_img = 255 - remove_lines(inv_img)
	pixel_sum = np.sum(resized_img)

	if pixel_sum < 2550:
		return True
	else:
		return False



if __name__ == '__main__':
	img = cv2.imread("8_2.png")
	print('*********************** Predicted on Img ************************')
	# predict_block(img)
	blank = detect_blank(img)
	print(blank)
	img = preprocess_block(img)