#!/usr/bin/python3.9

from queue import Queue
from threading import Thread
import chess
import logging
import sys
logger = logging.getLogger(__name__)

from player import Player
from minmax import Position

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

class ASBotChessEngine:

    def __init__(self, debug=False):
        self.debug = debug
        self.position = chess.Board()
        self.command_queue = Queue(30)
        self.runner_thread = Thread(target=self.run)
        self.runner_thread.start()
        logger.info("started")

    def add_command(self, command):
        self.command_queue.put(command)

    def execute(self, command):
        logger.info(command)
        tokens =  [t.strip() for t in command.split()]
        logger.info(tokens)
        token_iter = iter(tokens)
        try:
            while True:
                token = next(token_iter)
                if token == "uci":
                    logger.info('responding to uci')
                    print("id name ASBOT Chess")
                    print("id author Nicholas Entzi")
                    print("uciok")
                    logger.info('responded to uci')
                elif token == "debug":
                    token = next(token_iter)
                    if token == "on":
                        debug = True
                    if token == "off":
                        debug = False
                elif token == "isready":
                    print("readyok")
                elif token == "ucinewgame":
                    pass
                elif token == "position":
                    token = next(token_iter)
                    if token == "startpos":
                        self.position = chess.Board()
                        token = next(token_iter)
                    if token == "fen":
                        token = next(token_iter)
                        self.position = chess.Board(token)
                        token = next(token_iter)
                    if token == "moves":
                        while True:
                            self.position.push_uci(next(token_iter))
                elif token == "go":
                    move = Player().tree_search(self.position)
                    print("bestmove", move)
                elif token == "minmax":
                    pos = Position(self.position, int(next(token_iter)))
                    pos.print_continuation()
                    print(pos.total_count())
                elif token == "quit":
                    logger.info("quitting asbot")
                    sys.exit()
                

        except StopIteration:
            return
    
    def run(self):
        while True:
            command = self.command_queue.get()
            self.execute(command)


if __name__ == "__main__":
    logging.basicConfig(filename='/home/nick/Desktop/ASBOT/myapp.log', level=logging.INFO)
    engine = ASBotChessEngine()
    #SupervisedTrainer().load_data("fics_2023_data.pgn")
    while True:
        command = input()
        logger.info(command)
        engine.add_command(command)
        if command == "quit":
            sys.exit()