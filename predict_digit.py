import tensorflow as tf
import numpy as np
import cv2

# model = tf.keras.models.load_model("my_digit_recognizer.model")
model = tf.keras.models.load_model("my_digit_recognizer_19_NOV.model")

def preprocess_block(img):
	inv_img = 255 - remove_lines(img)
	inv_img = cv2.medianBlur(inv_img, 1)
	is_blank = False
	if detect_blank(inv_img):
		is_blank = True
	return inv_img, is_blank

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
	# cv2.imshow("Processed",no_border)
	# cv2.waitKey(0)
	return no_border

def detect_blank(inv_img):
	# print(inv_img.shape)
	pixel_sum = np.sum(inv_img)
	# print(pixel_sum)
	if pixel_sum < 2550:
		return True
	else:
		return False


def predict(img):
	img, is_blank = preprocess_block(img)
	if is_blank:
		return ""
	resized_img = img.reshape(1,28,28,1)
	resized_img = resized_img / 255
	prediction = model.predict(resized_img)
	prediction[0][0] = 0 # to remove the possibility to have a '0' prediction
	output = np.argmax(prediction)
	#print(output)
	return output


def predict_with_image_output(img):
	output = predict(img)
	cv2.imshow(str(output),img)
	cv2.waitKey(0)
	return output