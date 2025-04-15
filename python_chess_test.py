# Ensure the chess module is installed and accessible
try:
    import chess
except ImportError:
    print("The 'chess' module is not installed. Install it using 'pip install chess'.")
    raise
import chess.engine

board = chess.Board()

# print legal moves
print(list(board.legal_moves))
