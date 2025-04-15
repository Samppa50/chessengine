import torch

def board_to_tensor(board):
    pieces = {
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0,
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0, '.': 0
    }
    board_fen = board.board_fen()
    vec = [pieces.get(c, 0) for c in board_fen.replace('/', '')]
    vec += [1 if board.turn else -1]
    return torch.tensor(vec + [0] * (773 - len(vec)), dtype=torch.float32)
