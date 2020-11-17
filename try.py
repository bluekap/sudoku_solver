import cv2
import numpy as np
from predict_digit import predict,predict_with_image_output
import tensorflow as tf

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
	resized_img = cv2.medianBlur(img, 1)
	row,col,depth = img.shape
	gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
	inv_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
	cv2.imshow("Inverted",inv_img)
	cv2.waitKey(0)
	print(inv_img.shape)
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
	cv2.imshow("Processed",new_img)
	cv2.waitKey(0)
	
	new_img = cv2.resize(new_img,(28,28))
	new_img = new_img.reshape(1,28,28,1)
	new_img = new_img / 255
	prediction = model.predict(new_img)
	output = np.argmax(prediction)
	print(prediction)
	print(output)




if __name__ == '__main__':
	img = cv2.imread("3.png")
	# print('***********************Predicted on Img************************')
	# predict_block(img)
	img = preprocess_block(img)
