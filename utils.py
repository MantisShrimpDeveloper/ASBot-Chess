from chess import Board, square_file, square_rank, Move, square, PAWN, QUEEN
from torch import zeros, stack
import torch
from tqdm import tqdm

def get_move(raw_move:int, position:Board):
    from_square = square(raw_move // 512, (raw_move % 512) // 64)
    to_square = square((raw_move % 64) // 8, raw_move % 8)
    piece = position.piece_at(from_square)
    if piece == PAWN and square_rank(to_square) in [0, 7]:
        piece = QUEEN
    else:
        piece = None
    return Move(from_square, to_square, piece)


def get_raw_move(move: Move, white_to_move: bool):
    raw_move = square_file(move.from_square) * 512
    if white_to_move:
        raw_move += square_rank(move.from_square) * 64
    else:
        raw_move += (7 - square_rank(move.from_square)) * 64
    raw_move += square_file(move.to_square) * 8
    if white_to_move:
        raw_move += square_rank(move.to_square)
    else:
        raw_move += (7 - square_rank(move.to_square))
    return raw_move

def generate_legality_masks(positions:list[Board]):
    masks = []
    for position in positions:
        mask = zeros(8,8,8,8, dtype=torch.bool)
        for move in position.generate_legal_moves():
           mask[square_file(move.from_square)][square_rank(move.from_square)][square_file(move.to_square)][square_rank(move.to_square)] = True
        masks.append(mask)
    return stack(masks)

def generate_initial_tensors(boards:list[Board]):
    tensors = []
    for board in boards:
        tensor = zeros(8,8,8)
        for square, piece in board.piece_map().items():
           tensor[square_file(square)][square_rank(square)][int(piece.color != board.turn)] = 1
           tensor[square_file(square)][square_rank(square)][int(piece.piece_type) + 1] = 1
        if not board.turn:
            tensor = tensor.flip(1)
        tensors.append(tensor)
    return stack(tensors)

def generate_answer_tensors(positions: list[Board], moves: list[Move], white_to_move: bool):
    tensors = []
    print("generating answers")

    for position in tqdm(positions):
        tensor = zeros(8,8,8,8)
        for legal_move in position.legal_moves:
            tensor[square_file(legal_move.from_square)][square_rank(legal_move.from_square)][square_file(legal_move.to_square)][square_rank(legal_move.to_square)] = 1
        # tensor[square_file(move.from_square)][square_rank(move.from_square)][square_file(move.to_square)][square_rank(move.to_square)] = 1
        tensors.append(tensor)
    tensors = stack(tensors)
    if not white_to_move:
        tensors = tensors.flip(2, 4)
    return tensors

def generate_action_index(move: Move, white_to_move: bool):
    tensor = zeros(8,8,8,8, dtype=torch.int64)
    tensor[square_file(move.from_square)][square_rank(move.from_square)][square_file(move.to_square)][square_rank(move.to_square)] = 1
    if not white_to_move:
        tensor = tensor.flip(1, 3)
    return tensor

