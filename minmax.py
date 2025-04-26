from chess import Board, PAWN, ROOK, KNIGHT, BISHOP, QUEEN, WHITE

class Position:
    def __init__(self, board:Board, depth=0, move=None):
        self.board = board
        self.last_move = move
        self.is_game_over = False
        self.current_depth = 0
        if board.can_claim_draw():
            self.val = 0
            self.is_game_over = True
        if board.is_checkmate():
            self.is_game_over = True
            if board.turn == WHITE:
                self.val = -100
            else:
                self.val = 100
        self.val = self.value()
        self.future_positions = []
        if depth > 0:
            self.evaluate(depth)
        else:
            self.current_depth = depth

    def evaluate(self, depth: int):
        if self.is_game_over or depth <= self.current_depth:
            return
        if len(self.future_positions) == 0:
            for move in self.board.generate_legal_moves():
                new_board = self.board.copy()
                new_board.push(move)
                self.future_positions.append(Position(new_board, 0, move))
        for d in range(1, depth):
            for position in self.future_positions:
                if self.last_move == None:
                    print(d)
                position.evaluate(d)
        self.future_positions.sort(key=lambda pos: pos.val, reverse=self.board.turn == WHITE)
        if len(self.future_positions) > 0:
            self.val = self.future_positions[0].val
        self.current_depth = depth

    def value(self):
        white_total_value = 0
        black_total_value = 0
        for piece in self.board.piece_map().values():
            val = 0
            if piece.piece_type == PAWN:
                val = 1
            elif piece.piece_type == ROOK:
                val = 5
            elif piece.piece_type == KNIGHT:
                val = 3
            elif piece.piece_type == BISHOP:
                val = 3
            elif piece.piece_type == QUEEN:
                val = 9
            if piece.color == WHITE:
                white_total_value += val
            else:
                black_total_value += val

        return white_total_value - black_total_value
    
    def print_continuation(self):
        if self.last_move:
            print(self.last_move.uci())
        else:
            print(self.val)
        if len(self.future_positions) > 0:
            self.future_positions[0].print_continuation()

    def total_count(self):
        return 1 + sum([x.total_count() for x in self.future_positions])