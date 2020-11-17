import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("my_digit_recognizer.model")

def preprocess_block(img):
	resized_img = cv2.medianBlur(img, 1)
	row,col,depth = img.shape
	gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	inv_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
	# cv2.imshow("Inverted",inv_img)
	# cv2.waitKey(0)
	# print(inv_img.shape)
	x1,x2,y1,y2 = (0,0,0,0)

	y =  int(col/2) ## mid point
	for x in range(row):
		if inv_img[x,y] <= 0.2:
			x1 = x
			break

	for x in range(row-1,-1,-1):
		if inv_img[x,y] <= 0.2:
			x2 = x
			break

	x = int(row/2) ## mid point
	for y in range(col):
		if inv_img[x,y] <= 0.2:
			y1 = y
			break

	for y in range(col-1,-1,-1):
		if inv_img[x,y] <= 0.2:
			y2 = y
			break

	new_img = inv_img[x1:x2,y1:y2]
	return new_img

def predict(img):
	img = preprocess_block(img)
	resized_img = cv2.resize(img,(28,28))
	resized_img = resized_img.reshape(1,28,28,1)
	resized_img = resized_img / 255
	prediction = model.predict(resized_img)
	output = np.argmax(prediction)
	print(output)
	return output


def predict_with_image_output(img):
	output = predict(img)
	cv2.imshow(str(output),img)
	cv2.waitKey(0)
	return output