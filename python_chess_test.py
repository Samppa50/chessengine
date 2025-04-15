import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim

# simple NN
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(773, 256),  # 773 features from board + side to move
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.fc(x)

# convert a board to a simple feature vector
def board_to_tensor(board):
    board_fen = board.board_fen()
    pieces = {
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0,
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0, '.': 0
    }
    vec = [pieces.get(c, 0) for c in board_fen.replace('/', '')]
    vec += [1 if board.turn else -1]  # whose turn
    return torch.tensor(vec + [0] * (773 - len(vec)), dtype=torch.float32)

# parse PGN and train
def train_from_pgn(pgn_file):
    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    with open(pgn_file) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers["Result"]
            if result == "1-0":
                score = 1
            elif result == "0-1":
                score = -1
            else:
                score = 0
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
    return model
