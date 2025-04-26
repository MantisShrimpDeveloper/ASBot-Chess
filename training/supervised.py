from chess import Board, pgn
from torch import concatenate
from torch.utils.data import DataLoader
from models.network import Network
import torch
from os.path import isfile
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from tqdm import tqdm

from utils import generate_initial_tensors, generate_answer_tensors

NETWORK_PATH = "saved_network"

class SupervisedTrainer():
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_gpu = True
            print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            self.is_gpu = False
            print("Using CPU")
        self.network = Network(self.device)
        if isfile(NETWORK_PATH):
            self.network = torch.load(NETWORK_PATH, weights_only=False)

    def train(self):
        positions, answers = self.load_data("fics_2023_data.pgn")
        positives = answers.sum(0)
        negatives = torch.ones(8,8,8,8).mul(4096).sub(positives)
        pos_weight = negatives.div(positives)
        pos_weight[pos_weight == torch.inf] = 1000
        for i in range(1, 11):
            print("epoch and depth set to: ", i)
            self.network.set_depth(1)
            if self.is_gpu:
                self.network.cuda(self.device)
            dataset = list(zip(positions, answers))
            loss_function = BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
            dataloader = DataLoader(dataset, 1000, True)
            optimizer = SGD(self.network.parameters())
            loss_vals = []
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.network(x)
                loss = loss_function(y_pred, y)
                loss_vals.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("loss: ", sum(loss_vals) / len(loss_vals))

        torch.save(self.network, NETWORK_PATH)
        print("model saved!")
    
    def load_data(self, path):
        file = open(path)
        white_positions = []
        white_answers = []
        black_positions = []
        black_answers = []
        for i in tqdm(range(5000)):
            game = pgn.read_game(file)
            if game == None:
                break
            white_to_move = True
            board = Board()
            for game_node in game.mainline():
                if game_node == None:
                    break
                if white_to_move:
                    white_positions.append(board)
                    white_answers.append(game_node.move)
                    board = game_node.board()
                else:
                    black_positions.append(board)
                    black_answers.append(game_node.move)
                    board = game_node.board()
                white_to_move = not white_to_move
        
        positions = concatenate((generate_initial_tensors(white_positions), generate_initial_tensors(black_positions)))
        answers = concatenate((generate_answer_tensors(white_positions, white_answers, True), generate_answer_tensors(black_positions, black_answers, False)))
        print(positions.shape)
        print(answers.shape)
        return positions, answers