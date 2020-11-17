import cv2
import numpy as np
import pytesseract
from predict_digit import *

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def preprocess_text(text):
	string = str(text)
	for char in string:
		if char in ['1','2','3','4','5','6','7','8','9']:
			return char
	return ""

def print_board(board):
	for blocks in board:
		print('_'*63)
		print_str = ""
		for block in blocks:
			if block == "":
				print_str = print_str + "|     |"
			else:
				print_str = print_str  + ("|  {}  |".format(block))
		print(print_str)


img = cv2.imread("sample_image.png")
resized_img = cv2.resize(img,(300,300))
(width,height,depth) = resized_img.shape

block1 = resized_img[ 0: int(height/3) , 0 : int(width/3)]
gray_block = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray_block, 1)


blocks = []

for h_index in range(0,9):
	for w_index in range(0,9):
		blocks.append(resized_img[int(h_index * height/9): int( (h_index+1) * height/9) ,
								 int(w_index * width/9) : int((w_index+1) * width/9)])



# print("####################################################Prediction using Tesseract-OCR ############################")
# board = np.full([9,9], None)
# h_index=0
# w_index=0
# count = 0
# for block in blocks:
# 	text = pytesseract.image_to_string(block, config="--psm 8")
# 	if text != None or text != "":
# 		board[h_index][w_index] = preprocess_text(text)
# 	# print("text @({},{}) is {}".format(h_index,w_index,text))
# 	count = count + 1
# 	h_index = int(count/9)
# 	w_index = count % 9


# print_board(board)

print("####################################################Prediction using my model############################")
board = np.full([9,9], None)
h_index=0
w_index=0
count = 0
for block in blocks:
	text = predict(block)
	if text != None or text != "":
		board[h_index][w_index] = preprocess_text(text)
	# print("text @({},{}) is {}".format(h_index,w_index,text))
	count = count + 1
	h_index = int(count/9)
	w_index = count % 9


print_board(board)

# block1 = cv2.threshold(gray_block, 10, 255, cv2.THRESH_BINARY_INV)[1]
# block2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]
# cv2.imshow("Image",np.hstack((block1,block2)))
# cv2.waitKey(0)

# cv2.imshow("Image",blocks[68])
# cv2.waitKey(0)
# text = pytesseract.image_to_string(blocks[68], config="--psm 8")
# print(preprocess_text(text))

# for i in range(0,81):
# 	cv2.imshow("Image",blocks[i][4:-4,4:-4])
# 	cv2.waitKey(500)

# gray = gray_block[2:int(height/9)-1,2:int(width/9)]

# cv2.imshow("Image",gray)
# cv2.waitKey(0)
# text = pytesseract.image_to_string(gray)
# print(text)


# new_img = cv2.imread("sample_image.png")
# text = pytesseract.image_to_string(gray_block, config="--psm 1")
# print(text)


# new_img = cv2.imread("01.png")
# text = pytesseract.image_to_string(new_img, config="--psm 8")
# print(text)

# gray_img = cv2.threshold(new_img, 100, 255, cv2.THRESH_BINARY_INV)[1]
# text = pytesseract.image_to_string(gray_img, config="--psm 8")
# print(preprocess_text(text))


