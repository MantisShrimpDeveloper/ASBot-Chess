from chess import Board, WHITE
from collections import namedtuple
from models.network import Network
import torch
from os.path import isfile
from torch.nn import functional as func, SmoothL1Loss
from torch.optim import AdamW
from torch import flatten, argmax, cat, stack
import random
from tqdm import tqdm

from utils import generate_initial_tensors, generate_legality_masks, get_move, get_raw_move

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

NETWORK_PATH = "saved_network"
REINFORCEMENT_NETWORK_PATH = "reinforcement_network"

EPOCHS = 20
EPISODES = 200
GAMES = 100
MOVE_LIMIT = 100
BATCH_SIZE = 500
TAU = 0.01

class ReinforcementLearner():
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_gpu = True
            print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            self.is_gpu = False
            print("Using CPU")
        self.baseline = Network(self.device)
        self.player = Network(self.device)
        if isfile(NETWORK_PATH):
            self.baseline = torch.load(NETWORK_PATH, weights_only=False, map_location=self.device)
    
    def select_moves(self, boards: Board, white_to_move: bool):
        with torch.no_grad():
            initial_tensors = generate_initial_tensors(boards).to(self.device)
            selections = self.player(initial_tensors)
            selections = func.sigmoid(selections)
            if not white_to_move:
                selections = selections.flip(2, 4)
            leglity_masks = generate_legality_masks(boards)
            selections[~leglity_masks] = 0
            flat_selections = flatten(selections, 1)
            moves_raw = argmax(flat_selections, 1)
        return [get_move(m, p) for m, p in zip(moves_raw, boards)]
       

    def play_games(self):
        white_wins = []
        black_wins = []
        draws = []
        # generate starting positions
        boards = [Board.from_chess960_pos(pos) for pos in random.sample(range(960), GAMES)]

        for _ in tqdm(range(MOVE_LIMIT)):
            # white moves
            completed_games = []
            for board, move in zip(boards, self.select_moves(boards, True)):
                board.push(move)

                if board.is_checkmate():
                    white_wins.append(board)
                    completed_games.append(board)
                    break

                if board.can_claim_draw() or board.is_stalemate():
                    draws.append(board)
                    completed_games.append(board)
                    break

            for board in completed_games:
                boards.remove(board)
            if len(boards) == 0:
                break

            # black moves
            completed_games = []
            for board, move in zip(boards, self.select_moves(boards, False)):
                board.push(move)

                if board.is_checkmate():
                    black_wins.append(board)
                    completed_games.append(board)
                    break

                if board.can_claim_draw() or board.is_stalemate():
                    draws.append(board)
                    completed_games.append(board)
                    break
            
            for board in completed_games:
                boards.remove(board)
            if len(boards) == 0:
                break
            
        draws += boards

        return white_wins, black_wins, draws
    

    def generate_transitions(self, board: Board, white_reward: float, black_reward: float):
        transitions = []
        board.pop()
        white_to_move = board.turn
        state = generate_initial_tensors([board]).to(torch.device("cpu"))
        while len(board.move_stack) > 0:
            move = board.pop()
            white_to_move = not white_to_move
            next_state = state
            state = generate_initial_tensors([board]).to(torch.device("cpu"))
            action = torch.tensor(get_raw_move(move, white_to_move), dtype=torch.int64, device=torch.device("cpu"))
            reward = torch.tensor([white_reward if white_to_move else black_reward], dtype=torch.int64, device=torch.device("cpu"))
            transitions.append(Transition(state, action, next_state, reward))
        return transitions

    def train(self):
        self.player.load_state_dict(self.baseline.state_dict())
        criterion = SmoothL1Loss()
        optimizer = AdamW(self.player.parameters(), amsgrad=True)

        for epoch in range(EPOCHS):
            print("Epoch:", epoch)

            white_wins, black_wins, draws = self.play_games()
            print("Decisive games:", len(white_wins) + len(black_wins))

            all_transitions = []
            for game in white_wins:
                all_transitions += self.generate_transitions(game, 1.0, 0.0)
            for game in black_wins:
                all_transitions += self.generate_transitions(game, 0.0, 1.0)
            for game in draws:
                all_transitions += self.generate_transitions(game, 0.5, 0.5)

            loss_vals = []
            for episode in tqdm(range(EPISODES)):
                transitions = random.sample(all_transitions, BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                state_batch = cat(batch.state).to(self.device)
                action_batch = stack(batch.action).unsqueeze(1).to(self.device)
                next_state_batch = cat(batch.next_state).to(self.device)
                reward_batch = cat(batch.reward).to(self.device)
        
                selections = self.player(state_batch)
                selections = func.sigmoid(selections)
                selections = flatten(selections, 1)
                actual_values = selections.gather(1, action_batch)

                with torch.no_grad():
                    next_state_selections = self.baseline(next_state_batch)
                    next_state_selections = func.sigmoid(next_state_selections)
                    next_state_selections = flatten(next_state_selections, 1)
                    next_state_values = next_state_selections.max(1).values
                    expected_values = ((1 - next_state_values) * 0.5) + (reward_batch * 0.5)

                loss = criterion(actual_values, expected_values.unsqueeze(1))
                loss_vals.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Average loss:", sum(loss_vals) / len(loss_vals))
            
            baseline_state_dict = self.baseline.state_dict()
            player_state_dict = self.player.state_dict()
            for key in player_state_dict:
                baseline_state_dict[key] = player_state_dict[key]*TAU + baseline_state_dict[key]*(1-TAU)
            self.baseline.load_state_dict(baseline_state_dict)

            torch.save(self.baseline, REINFORCEMENT_NETWORK_PATH)
            print("model saved!")
