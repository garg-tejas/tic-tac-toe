import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.winner = None
        self.game_over = False
        self.move_history = []
        print(f"[GAME] New tic-tac-toe game initialized")
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.move_history = []
        print(f"[GAME] Game reset")
        return self.get_state()
    
    def get_state(self):
        return self.board.copy()
    
    def get_valid_moves(self):
        if self.game_over:
            return []
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def make_move(self, position):
        if self.game_over:
            print(f"[GAME] Attempted move on finished game")
            return self.get_state(), 0, True
        
        i, j = position
        player_symbol = "X" if self.current_player == 1 else "O"
        
        if self.board[i, j] != 0:
            print(f"[GAME] Invalid move by {player_symbol} at position ({i},{j})")
            return self.get_state(), -10, False  # Invalid move penalty
        
        self.board[i, j] = self.current_player
        self.move_history.append((i, j))
        print(f"[GAME] Player {player_symbol} placed at position ({i},{j})")
        
        # Check for win
        reward = 0
        if self._check_win():
            self.winner = self.current_player
            self.game_over = True
            reward = 1 if self.current_player == 1 else -1
            print(f"[GAME] Player {player_symbol} wins! Reward: {reward}")
        # Check for draw
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
            reward = 0.5  # Small reward for draw
            print(f"[GAME] Game ended in a draw. Reward: {reward}")
        
        # Switch player
        self.current_player *= -1
        
        return self.get_state(), reward, self.game_over
    
    def _check_win(self):
        # Check rows, columns, and diagonals
        for i in range(3):
            # Check rows
            if abs(np.sum(self.board[i, :])) == 3:
                return True
            # Check columns
            if abs(np.sum(self.board[:, i])) == 3:
                return True
        
        # Check diagonals
        if abs(np.sum(np.diag(self.board))) == 3:
            return True
        if abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        
        return False
    
    def render(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        board_str = ""
        for i in range(3):
            for j in range(3):
                board_str += symbols[self.board[i, j]]
                if j < 2:
                    board_str += "|"
            if i < 2:
                board_str += "\n-+-+-\n"
        return board_str