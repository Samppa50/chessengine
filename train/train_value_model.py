import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import chess.pgn
import torch
import torch.optim as optim
from engine.chess_net import ChessNet
from engine.board_utils import board_to_tensor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)

def train_from_pgn(pgn_file, model_path):
    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers["Result"]
            score = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(result, 0)
            board = game.board()

            for move in game.mainline_moves():
                x = board_to_tensor(board)
                y = torch.tensor([score], dtype=torch.float32)
                pred = model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                board.push(move)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Example usage
# train_from_pgn("data/games.pgn", "models/value_model.pth")
train_from_pgn(get_path("data/games.pgn"), get_path("models/value_model.pth"))
