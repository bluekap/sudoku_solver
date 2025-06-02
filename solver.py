def is_valid_move(board, row, col, num):
    """
    Check if placing 'num' at position (row, col) is valid
    """
    # Check row
    for j in range(9):
        if board[row][j] == num:
            return False

    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False

    # Check 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3

    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if board[i][j] == num:
                return False

    return True


def find_empty_cell(board):
    """
    Find the next empty cell (returns row, col or None if board is full)
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None


def solve_sudoku(board):
    """
    Solve Sudoku using backtracking algorithm
    Returns True if solved, False if unsolvable
    """
    empty_cell = find_empty_cell(board)

    if not empty_cell:
        return True  # Board is complete

    row, col = empty_cell

    for num in range(1, 10):
        if is_valid_move(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            # Backtrack
            board[row][col] = 0

    return False


def is_valid_sudoku(board):
    """
    Check if the current board state is valid (no conflicts)
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                # Temporarily remove the number to check validity
                num = board[i][j]
                board[i][j] = 0

                if not is_valid_move(board, i, j, num):
                    board[i][j] = num  # Restore the number
                    return False

                board[i][j] = num  # Restore the number

    return True


def print_board(board):
    """
    Pretty print the Sudoku board
    """
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")

        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")

            if board[i][j] == 0:
                print(". ", end="")
            else:
                print(str(board[i][j]) + " ", end="")

        print()


def count_empty_cells(board):
    """
    Count the number of empty cells in the board
    """
    count = 0
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                count += 1
    return count


def get_board_difficulty(board):
    """
    Estimate difficulty based on number of empty cells
    """
    empty_cells = count_empty_cells(board)

    if empty_cells <= 30:
        return "Easy"
    elif empty_cells <= 45:
        return "Medium"
    elif empty_cells <= 55:
        return "Hard"
    else:
        return "Expert"


def copy_board(board):
    """
    Create a deep copy of the board
    """
    return [row[:] for row in board]

print("Solver Script loaded successfully")
