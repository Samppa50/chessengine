import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import chess
import torch
from engine.chess_net import ChessNet
from engine.engine_logic import engine_move

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(BASE_DIR, "models", "value_model.pth")

model = ChessNet()
model.load_state_dict(torch.load(model_path))
model.eval()


def play_game():
    board = chess.Board()
    print(board)
    while not board.is_game_over():
        if board.turn:  # Human plays white
            move = input("Your move (UCI): ")
            try:
                move_obj = chess.Move.from_uci(move)
                if move_obj in board.legal_moves:
                    board.push(move_obj)
                else:
                    print("Illegal move.")
            except:
                print("Invalid format.")
        else:  # Engine (black)
            print("Engine thinking...")
            move = engine_move(board, model)
            print(f"Engine plays: {move}")
            board.push(move)
        print(board)
    print("Game over:", board.result())

if __name__ == "__main__":
    play_game()
