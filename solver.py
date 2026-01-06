
class SudokuSolver:
    def __init__(self, board, limit=100000):
        self.board = board
        self.rows = [0] * 9
        self.cols = [0] * 9
        self.boxes = [0] * 9
        self.empty_cells = []
        self.iterations = 0
        self.limit = limit
        
        # Initialize state
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val != 0:
                    mask = 1 << val
                    self.rows[r] |= mask
                    self.cols[c] |= mask
                    box_idx = (r // 3) * 3 + (c // 3)
                    self.boxes[box_idx] |= mask
                else:
                    self.empty_cells.append((r, c))
    
    def get_possible_values(self, r, c):
        box_idx = (r // 3) * 3 + (c // 3)
        used = self.rows[r] | self.cols[c] | self.boxes[box_idx]
        
        possible = []
        for val in range(1, 10):
            if not (used & (1 << val)):
                possible.append(val)
        return possible

    def solve(self):
        self.iterations += 1
        if self.iterations > self.limit:
            return False # Timeout
            
        if not self.empty_cells:
            return True # Solved
            
        # Find best cell (MRV)
        # Scan only remaining empty cells
        best_idx = -1
        min_options = 10
        possible_vals = []
        
        for idx, (r, c) in enumerate(self.empty_cells):
            vals = self.get_possible_values(r, c)
            num_options = len(vals)
            
            if num_options == 0:
                return False # Dead end immediately if any cell has no options
            
            if num_options < min_options:
                min_options = num_options
                best_idx = idx
                possible_vals = vals
                if min_options == 1:
                    break
        
        # Select best cell
        r, c = self.empty_cells[best_idx]
        
        # Efficiently remove from list (swap with last and pop)
        # We need to restore it later, so we just remember it and slicing might be easier or swap
        # Swap is O(1)
        last_idx = len(self.empty_cells) - 1
        self.empty_cells[best_idx], self.empty_cells[last_idx] = self.empty_cells[last_idx], self.empty_cells[best_idx]
        self.empty_cells.pop()
        
        box_idx = (r // 3) * 3 + (c // 3)
        
        for val in possible_vals:
            mask = 1 << val
            
            # Place
            self.board[r][c] = val
            self.rows[r] |= mask
            self.cols[c] |= mask
            self.boxes[box_idx] |= mask
            
            if self.solve():
                return True
                
            # Backtrack
            self.rows[r] &= ~mask
            self.cols[c] &= ~mask
            self.boxes[box_idx] &= ~mask
            
        # Restore empty cell for backtracking
        self.board[r][c] = 0
        self.empty_cells.append((r, c))
        # Keep list stability not required for correctness, but pure append is O(1)
        return False


def solve_sudoku(board):
    """
    Solve Sudoku using optimized Bitmask Solver with MRV
    """
    if board is None: return False
    solver = SudokuSolver(board)
    return solver.solve()


def is_valid_move(board, row, col, num):
    """
    Legacy helper for is_valid_sudoku
    """
    # Check row
    for j in range(9):
        if j != col and board[row][j] == num:
            return False
    # Check col
    for i in range(9):
        if i != row and board[i][col] == num:
            return False
    # Check box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if (i != row or j != col) and board[i][j] == num:
                return False
    return True


def is_valid_sudoku(board):
    """
    Check if the current board state is valid (no conflicts)
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                if not is_valid_move(board, i, j, board[i][j]):
                    return False
    return True


def copy_board(board):
    return [row[:] for row in board]


def print_board(board):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("| ", end="")
            print(f"{board[i][j]} " if board[i][j] != 0 else ". ", end="")
        print()

