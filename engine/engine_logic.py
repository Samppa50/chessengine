def engine_move(board, model):
    from .evaluation import evaluate_position

    best_score = -float('inf') if board.turn else float('inf')
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        score = evaluate_position(board, model)
        board.pop()
        if board.turn:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move
    return best_move
