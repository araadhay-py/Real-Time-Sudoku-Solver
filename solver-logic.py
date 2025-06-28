def solve_sudoku(board):
    from copy import deepcopy
    def find_empty(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return None

    def valid(board, num, pos):
        row, col = pos
        if num in board[row]:
            return False
        if num in [board[i][col] for i in range(9)]:
            return False
        box_x, box_y = col // 3, row // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if board[i][j] == num:
                    return False
        return True

    def backtrack(board):
        empty = find_empty(board)
        if not empty:
            return True
        row, col = empty
        for num in range(1, 10):
            if valid(board, num, (row, col)):
                board[row][col] = num
                if backtrack(board):
                    return True
                board[row][col] = 0
        return False

    board_copy = deepcopy(board)
    backtrack(board_copy)
    return board_copy

def grid_values(grid_list):
    return [grid_list[i:i+9] for i in range(0, 81, 9)]
