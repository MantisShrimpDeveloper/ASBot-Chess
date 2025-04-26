from chess import Board, square_file, square_rank, WHITE
from torch import flatten, argmax
from models.network import Network
import torch
from os.path import isfile
from torch.nn import functional as func
from time import time

from utils import generate_initial_tensors, generate_legality_masks, get_move

NETWORK_PATH = "saved_network"
SEARCH_DEPTH = 4
PARALLEL_SEARCH = 50


class Position:
    def __init__(self, board:Board, trunk_relavence:float, branch_relavence:float, value:float, parent:'Position'):
        self.board = board
        self.trunk_relavence = trunk_relavence
        self.branch_relavence = branch_relavence
        self.value = value
        self.parent = parent
        self.children = []
        self.completed = False
        if board.is_checkmate():
            if board.turn:
                self.value = 0
            else:
                self.value = 1
            self.completed = True

        if board.can_claim_draw() or board.is_stalemate():
            self.value = 0.5
            self.completed = True
    
    def absolute_relavence(self):
        return self.trunk_relavence * self.branch_relavence
    
    def generate_leaves(self, boards_to_values:map, seen_boards:map) -> list:
        self.children = [Position(seen_boards[board], 0, 0, value, self) for board, value in boards_to_values.items()]
        self.update_branch_relevance()
        self.value = self.best_child_value()
        if self.parent:
            self.parent.new_child_value()
        else:
            self.update_trunk_relevance(self.trunk_relavence)
        return [child for child in self.children if not child.completed]
    
    def best_child_value(self):
        best = max if self.board.turn else min
        return best(child.value for child in self.children)

    def new_child_value(self):
        self.update_branch_relevance()
        new_value = self.best_child_value()
        if new_value != self.value:
            self.value = new_value
            if self.parent:
                self.parent.new_child_value()
            else:
                self.update_trunk_relevance(self.trunk_relavence)
        else:
            self.update_trunk_relevance(self.trunk_relavence)

    def update_branch_relevance(self):
        if self.board.turn:
            total = sum(child.value for child in self.children)
            for child in self.children:
                child.branch_relavence = child.value / total
        else:
            total = sum(1 - child.value for child in self.children)
            for child in self.children:
                child.branch_relavence = (1 - child.value) / total

    def update_trunk_relevance(self, trunk_relevance):
        self.trunk_relavence = trunk_relevance
        for child in self.children:
            child.update_trunk_relevance(self.absolute_relavence())

class Player:
    def __init__(self):
        self.network = Network(torch.device("cpu"))
        if isfile(NETWORK_PATH):
            self.network = torch.load(NETWORK_PATH, weights_only=False, map_location=torch.device("cpu"))

    def play(self, position:Board) -> str:
        white_to_move = position.turn == WHITE
        positions = [position]
        initial_tensors = generate_initial_tensors(positions)
        selections = self.network(initial_tensors)
        selections = func.sigmoid(selections)
        if not white_to_move:
            selections = selections.flip(2, 4)
        leglity_masks = generate_legality_masks(positions)
        selections[~leglity_masks] = 0
        flat_selections = flatten(selections, 1)
        moves_raw = argmax(flat_selections, 1)
        moves = [get_move(move.item(), position) for move, position in zip(moves_raw, positions)]
        return moves[0].uci()
    

    def gather_move_evaluations(self, positions:list, selections, seen_boards:map):
        positions_to_boards_to_values = {}
        for position, selection in zip(positions, selections):
            boards_to_values = {}
            for move in position.board.generate_legal_moves():
                board = position.board.copy()
                board.push(move)
                if board.fen() not in seen_boards.keys():
                    seen_boards[board.fen()] = board
                    value = selection[square_file(move.from_square)][square_rank(move.from_square)][square_file(move.to_square)][square_rank(move.to_square)]
                    if not position.board.turn:
                        value = 1 - value
                    boards_to_values[board.fen()] = value
            positions_to_boards_to_values[position] = boards_to_values
        return positions_to_boards_to_values

    def evaluate(self, positions:list, seen_boards:map) -> map:
        boards = [position.board for position in positions]
        initial_tensors = generate_initial_tensors(boards)
        selections = self.network(initial_tensors)
        selections = func.sigmoid(selections)
        for index, position in enumerate(positions):
            if not position.board.turn:
                selections[index] = selections[index].flip(1, 3)
        return self.gather_move_evaluations(positions, selections, seen_boards)

    def tree_search(self, root_board: Board):
        start = time()
        seen_boards = {root_board.fen():root_board}
        root = Position(root_board, 1, 1, 0.5, None)
        leaves = [root]

        for _ in range(SEARCH_DEPTH):
            # print(_, len(leaves))
            if len(leaves) == 0:
                break
            search_count = min(PARALLEL_SEARCH, len(leaves))
            branches = leaves[:search_count]
            leaves = leaves[search_count:]
            branches_to_boards_to_values = self.evaluate(branches, seen_boards)
            new_leaves = [leaf for branch, boards_to_values in branches_to_boards_to_values.items() for leaf in branch.generate_leaves(boards_to_values, seen_boards)]
            leaves.extend(new_leaves)
            leaves.sort(key=Position.absolute_relavence, reverse=True)

        root.children.sort(key=Position.absolute_relavence, reverse=True)
        best_moves = [position.board.pop() for position in root.children]
        # print(time() - start)
        return best_moves[0].uci()

