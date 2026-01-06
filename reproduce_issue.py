
import time
from solver import solve_sudoku, print_board

# The board from the user's logs
board = [
    [0, 2, 0, 0, 3, 0, 0, 4, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 4, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 8, 0, 6, 0, 0, 0],
    [8, 0, 0, 0, 1, 0, 0, 0, 6],
    [0, 0, 0, 1, 0, 5, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 4, 0, 0, 2, 0]
]

print("Starting solver...")
start_time = time.time()
solved = solve_sudoku(board)
end_time = time.time()

if solved:
    print("Solved!")
    print_board(board)
else:
    print("Could not solve.")

print(f"Time taken: {end_time - start_time:.4f} seconds")
