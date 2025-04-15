import torch
from .board_utils import board_to_tensor

def evaluate_position(board, model):
    model.eval()
    x = board_to_tensor(board)
    with torch.no_grad():
        return model(x).item()
