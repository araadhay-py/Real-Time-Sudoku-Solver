import cv2
import numpy as np
from digit_recognizer import predict_digit, load_model
from solver_logic import solve_sudoku

def extract_sudoku_grid(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            grid = approx
            break

    pts = grid.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (450, 450))
    return warp

def split_cells(warp):
    cells = []
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    step = 50
    for y in range(9):
        for x in range(9):
            roi = warp_gray[y*step:(y+1)*step, x*step:(x+1)*step]
            roi = cv2.resize(roi, (28, 28))
            cells.append(roi)
    return cells

def recognize_board(cells, model):
    board = []
    for cell in cells:
        digit = predict_digit(cell, model)
        board.append(digit)
    return board

def display_solution(original_img, board, solved_board):
    step = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(81):
        y, x = divmod(i, 9)
        if board[i] == 0:
            cv2.putText(original_img, str(solved_board[i]), 
                        (x*step+15, y*step+35), font, 1, (0, 255, 0), 2)
    return original_img

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            grid = extract_sudoku_grid(frame)
            cells = split_cells(grid)
            board = recognize_board(cells, model)
            board_2d = [board[i:i+9] for i in range(0, 81, 9)]

            solved = solve_sudoku(board_2d)
            solved_flat = [cell for row in solved for cell in row]
            solved_image = display_solution(grid.copy(), board, solved_flat)

            cv2.imshow("Solved Sudoku", solved_image)
        except:
            cv2.imshow("Solved Sudoku", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
