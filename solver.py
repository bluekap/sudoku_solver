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

def check(x,y,board):
	num = board[y][x]
	
	# Check rows and columns
	for i in range(9):
		if (board[y][i] == num) and x!=i:
			return False
		if (board[i][x] == num) and y!=i:
			return False

	x_block = int(x/3)
	y_block = int(y/3)

	for m in range(3):
		for n in range(3):
			if (board[y_block*3 + m][x_block*3 + n] == num) and x!=(x_block*3 + n):
				return False
	return True

def solve(board):
	def backtrack(x,y):
		#print("Solving {},{}".format(x,y))
		if x == 9 and y == 8 :
			return True

		if x == 9:
			y = y + 1
			x = 0

		if board[y][x] != "":
			return backtrack(x+1,y)

		for i in range(1,10):
			board[y][x] = i
			if check(x,y,board):
				if(backtrack(x+1,y)):
					return True
			board[y][x] = ""					
		return False
	return backtrack(0,0)



if __name__ == "__main__":

	board = [
			[3, 0, 6, 5, 0, 8, 4, 0, 0],
			[5, 2, 0, 0, 0, 0, 0, 0, 0], 
	        [0, 8, 7, 0, 0, 0, 0, 3, 1], 
	        [0, 0, 3, 0, 1, 0, 0, 8, 0], 
	        [9, 0, 0, 8, 6, 3, 0, 0, 5], 
	        [0, 5, 0, 0, 9, 0, 6, 0, 0], 
	        [1, 3, 0, 0, 0, 0, 2, 5, 0], 
	        [0, 0, 0, 0, 0, 0, 0, 7, 4], 
	        [0, 0, 5, 2, 0, 6, 3, 0, 0]
	        ]	

	if(solve(board)):
		print_board(board)
	else:
		print("No solution possible.")