import pygame
import numpy as np
import sys
import time
from game import TicTacToe
from rl_agent import QAgent, RandomAgent, MiniMaxAgent, train_agent

# Initialize pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
RED = (255, 50, 50)
BLUE = (50, 50, 255)
GREEN = (50, 200, 50)
DARK_GREEN = (0, 150, 0)
LIGHT_BLUE = (200, 200, 255)
LIGHT_RED = (255, 200, 200)
LIGHT_GREEN = (200, 255, 200)
DARK_BLUE = (0, 0, 150)
DARK_RED = (150, 0, 0)
GOLD = (255, 215, 0)

# Screen dimensions
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600

# Board dimensions
BOARD_SIZE = 360
CELL_SIZE = BOARD_SIZE // 3
BOARD_X = 50
BOARD_Y = 120

# Button dimensions
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20

# Training visualization settings
TRAIN_HISTORY_WIDTH = 400
TRAIN_HISTORY_HEIGHT = 200
TRAIN_GRAPH_X = BOARD_X + BOARD_SIZE + 40
TRAIN_GRAPH_Y = BOARD_Y
MAX_HISTORY_POINTS = 100

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tic Tac Toe - RL Agent")
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 18)
large_font = pygame.font.SysFont("Arial", 36)
title_font = pygame.font.SysFont("Arial", 48, bold=True)


class Button:
    def __init__(self, x, y, width, height, text, color=BLUE, hover_color=DARK_BLUE, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
    
    def draw(self, surface):
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=10)
        
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def is_hovered(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
    
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False


class RadioButton:
    def __init__(self, x, y, text, group=None, selected=False):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.text = text
        self.selected = selected
        self.group = group
        if group is not None:
            group.append(self)
    
    def draw(self, surface):
        pygame.draw.circle(surface, BLACK, self.rect.center, 10, 2)
        if self.selected:
            pygame.draw.circle(surface, BLACK, self.rect.center, 6)
        
        text_surf = small_font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        surface.blit(text_surf, text_rect)
    
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if (pos[0] - self.rect.centerx) ** 2 + (pos[1] - self.rect.centery) ** 2 <= 100:
                return True
        return False
    
    def select(self):
        if self.group is not None:
            for btn in self.group:
                btn.selected = False
        self.selected = True


class TicTacToeGUI:
    def __init__(self):
        self.game = TicTacToe()
        self.rl_agent = QAgent(player=1)  # RL agent plays X (changed from O to X)
        self.random_agent = RandomAgent(player=-1)  # Random agent plays O (changed from X to O)
        self.minimax_agent = MiniMaxAgent(player=-1)  # MiniMax agent plays O (changed from X to O)
        
        # Try to load a pre-trained agent
        self.rl_agent.load()
        
        self.opponent = self.random_agent  # Default opponent
        self.training = False
        self.training_speed = 0.05  # Made training faster for quicker results
        self.episode_count = 0
        self.target_episodes = 5000
        self.training_history = []
        self.win_rate_history = []
        
        self.human_player = -1  # Human plays O by default (changed from X to O)
        self.current_player = 1  # X starts (RL agent goes first)
        self.game_over = False
        self.result_message = ""
        
        # Create UI elements
        self.create_buttons()
        
        # Radio button groups
        self.opponent_group = []
        self.player_group = []
        
        # Create radio buttons for opponent selection
        self.rb_random = RadioButton(TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 20, 
                                    "Random Opponent", self.opponent_group, True)
        self.rb_minimax = RadioButton(TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 45, 
                                     "MiniMax Opponent", self.opponent_group)
        
        # Create radio buttons for player selection
        self.rb_play_o = RadioButton(TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 90, 
                                    "Play as O", self.player_group, True)
        self.rb_play_x = RadioButton(TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 115, 
                                    "Play as X", self.player_group, False)
    
    def create_buttons(self):
        self.buttons = {
            "train": Button(BOARD_X, BOARD_Y + BOARD_SIZE + 30, BUTTON_WIDTH, BUTTON_HEIGHT, 
                           "Train Agent", GREEN, DARK_GREEN),
            "reset": Button(BOARD_X + BUTTON_WIDTH + BUTTON_MARGIN, BOARD_Y + BOARD_SIZE + 30, 
                           BUTTON_WIDTH, BUTTON_HEIGHT, "Reset Game", BLUE),
            "save": Button(TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 160, 
                          BUTTON_WIDTH, BUTTON_HEIGHT, "Save Agent", GREEN, DARK_GREEN),
            "load": Button(TRAIN_GRAPH_X + BUTTON_WIDTH + BUTTON_MARGIN, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 160, 
                          BUTTON_WIDTH, BUTTON_HEIGHT, "Load Agent", BLUE)
        }
    
    def draw_board(self):
        # Draw the board background
        pygame.draw.rect(screen, WHITE, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE))
        pygame.draw.rect(screen, BLACK, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE), 3)
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(screen, BLACK, 
                            (BOARD_X + i * CELL_SIZE, BOARD_Y), 
                            (BOARD_X + i * CELL_SIZE, BOARD_Y + BOARD_SIZE), 
                            3)
            # Horizontal lines
            pygame.draw.line(screen, BLACK, 
                            (BOARD_X, BOARD_Y + i * CELL_SIZE), 
                            (BOARD_X + BOARD_SIZE, BOARD_Y + i * CELL_SIZE), 
                            3)
        
        # Draw X's and O's
        for i in range(3):
            for j in range(3):
                cell_value = self.game.board[i, j]
                if cell_value == 1:  # X
                    self.draw_x(BOARD_X + j * CELL_SIZE, BOARD_Y + i * CELL_SIZE)
                elif cell_value == -1:  # O
                    self.draw_o(BOARD_X + j * CELL_SIZE, BOARD_Y + i * CELL_SIZE)
    
    def draw_x(self, x, y):
        # Draw X
        margin = 20
        pygame.draw.line(screen, RED, (x + margin, y + margin), 
                        (x + CELL_SIZE - margin, y + CELL_SIZE - margin), 8)
        pygame.draw.line(screen, RED, (x + CELL_SIZE - margin, y + margin), 
                        (x + margin, y + CELL_SIZE - margin), 8)
    
    def draw_o(self, x, y):
        # Draw O
        margin = 20
        center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
        radius = CELL_SIZE // 2 - margin
        pygame.draw.circle(screen, BLUE, center, radius, 8)
    
    def draw_training_stats(self):
        # Draw training stats background
        pygame.draw.rect(screen, LIGHT_BLUE, 
                        (TRAIN_GRAPH_X, TRAIN_GRAPH_Y, TRAIN_HISTORY_WIDTH, TRAIN_HISTORY_HEIGHT))
        pygame.draw.rect(screen, BLACK, 
                        (TRAIN_GRAPH_X, TRAIN_GRAPH_Y, TRAIN_HISTORY_WIDTH, TRAIN_HISTORY_HEIGHT), 2)
        
        # Draw title
        title = small_font.render("Training Progress", True, BLACK)
        screen.blit(title, (TRAIN_GRAPH_X + 10, TRAIN_GRAPH_Y + 10))
        
        # Draw win rate
        win_rate = self.rl_agent.get_win_rate()
        win_text = small_font.render(f"Win Rate: {win_rate:.2f}", True, BLACK)
        screen.blit(win_text, (TRAIN_GRAPH_X + 10, TRAIN_GRAPH_Y + 35))
        
        # Draw episode count
        episode_text = small_font.render(f"Episodes: {self.episode_count}/{self.target_episodes}", True, BLACK)
        screen.blit(episode_text, (TRAIN_GRAPH_X + 10, TRAIN_GRAPH_Y + 60))
        
        # Draw exploration rate
        explore_text = small_font.render(f"Exploration Rate: {self.rl_agent.exploration_rate:.4f}", True, BLACK)
        screen.blit(explore_text, (TRAIN_GRAPH_X + 10, TRAIN_GRAPH_Y + 85))
        
        # Draw win/loss/draw counts
        stats = self.rl_agent.training_history
        stats_text = small_font.render(
            f"Wins: {stats['wins']}  Losses: {stats['losses']}  Draws: {stats['draws']}", 
            True, BLACK)
        screen.blit(stats_text, (TRAIN_GRAPH_X + 10, TRAIN_GRAPH_Y + 110))
        
        # Draw win rate graph
        if len(self.win_rate_history) > 1:
            graph_margin = 40
            graph_width = TRAIN_HISTORY_WIDTH - 20
            graph_height = 70
            graph_x = TRAIN_GRAPH_X + 10
            graph_y = TRAIN_GRAPH_Y + 130
            
            # Draw graph axes
            pygame.draw.line(screen, BLACK, 
                            (graph_x, graph_y + graph_height), 
                            (graph_x + graph_width, graph_y + graph_height), 2)
            pygame.draw.line(screen, BLACK, 
                            (graph_x, graph_y), 
                            (graph_x, graph_y + graph_height), 2)
            
            # Plot win rate history
            points = []
            step = max(1, len(self.win_rate_history) // 50)
            for i in range(0, len(self.win_rate_history), step):
                x = graph_x + (i / len(self.win_rate_history)) * graph_width
                y = graph_y + graph_height - (self.win_rate_history[i] * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, DARK_GREEN, False, points, 2)
    
    def draw_ui(self):
        # Clear screen
        screen.fill(GRAY)
        
        # Draw title
        title_text = title_font.render("Tic Tac Toe RL", True, BLACK)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 20))
        
        # Draw subtitle based on state
        if self.training:
            subtitle = font.render("Training in progress...", True, DARK_GREEN)
        else:
            if self.game_over:
                subtitle = font.render(self.result_message, True, RED if "lost" in self.result_message.lower() else
                                      GOLD if "draw" in self.result_message.lower() else GREEN)
            else:
                player = "X" if self.current_player == 1 else "O"
                subtitle = font.render(f"Current player: {player}", True, BLACK)
        screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2, 80))
        
        # Draw board
        self.draw_board()
        
        # Draw training stats
        self.draw_training_stats()
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(screen)
        
        # Draw radio buttons
        for rb in self.opponent_group:
            rb.draw(screen)
        for rb in self.player_group:
            rb.draw(screen)
        
        # Draw section titles
        opponent_title = font.render("Opponent Selection:", True, BLACK)
        screen.blit(opponent_title, (TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT))
        
        player_title = font.render("Player Selection:", True, BLACK)
        screen.blit(player_title, (TRAIN_GRAPH_X, TRAIN_GRAPH_Y + TRAIN_HISTORY_HEIGHT + 70))
        
        # Update the screen
        pygame.display.flip()
    
    def handle_click(self, pos):
        if self.training:
            return
        
        # Check if clicked on board
        if (BOARD_X <= pos[0] <= BOARD_X + BOARD_SIZE and 
            BOARD_Y <= pos[1] <= BOARD_Y + BOARD_SIZE and 
            not self.game_over):
            
            # Convert click position to board coordinates
            j = (pos[0] - BOARD_X) // CELL_SIZE
            i = (pos[1] - BOARD_Y) // CELL_SIZE
            
            # Check if it's human's turn and the cell is empty
            if self.current_player == self.human_player and self.game.board[i, j] == 0:
                self.make_move((i, j))
                
                # If game not over, let agent make a move
                if not self.game_over and self.current_player != self.human_player:
                    self.make_agent_move()
    
    def make_move(self, position):
        if self.game_over:
            return
        
        # Make the move
        _, reward, done = self.game.make_move(position)
        
        # Update game state
        self.current_player = self.game.current_player
        self.game_over = done
        
        # Update result message if game is over
        if done:
            if self.game.winner is None:
                self.result_message = "Game ended in a draw!"
            elif self.game.winner == self.human_player:
                self.result_message = "You won!"
            else:
                self.result_message = "You lost!"
    
    def make_agent_move(self):
        # Determine which agent to use
        agent = self.rl_agent if self.current_player == self.rl_agent.player else self.opponent
        
        # Short delay to make the game feel more natural
        self.draw_ui()
        pygame.display.flip()
        time.sleep(0.3)  # 300ms delay before agent makes its move
        
        # Get the agent's action with look-ahead for RL agent
        look_ahead = True if agent == self.rl_agent else False
        action = agent.choose_action(self.game.board, training=False, look_ahead=look_ahead)
        
        if action is not None:
            # Make the move
            self.make_move(action)
    
    def reset_game(self):
        self.game.reset()
        self.current_player = 1  # X starts
        self.game_over = False
        self.result_message = ""
        
        # Always make the RL agent go first when playing against a human
        # This gives the agent an unfair advantage
        if self.rl_agent.player == 1:  # RL agent is X
            self.make_agent_move()
        elif self.current_player != self.human_player:
            self.make_agent_move()
    
    def train_rl_agent(self):
        self.training = True
        self.episode_count = 0
        self.training_history = []
        self.win_rate_history = []
        
        # Start training in a separate loop
        train_env = TicTacToe()
        old_exploration = self.rl_agent.exploration_rate
        
        # Set higher exploration for training
        self.rl_agent.exploration_rate = 0.3
        
        while self.episode_count < self.target_episodes and self.training:
            # Train for a batch of episodes
            batch_size = 100
            for _ in range(batch_size):
                if not self.training:
                    break
                    
                # Sometimes train against minimax even if random opponent is selected
                # This helps the agent learn more advanced strategies
                if random.random() < 0.3:  # 30% of the time, train against minimax
                    train_opponent = self.minimax_agent
                else:
                    # Use selected opponent
                    train_opponent = self.random_agent if self.rb_random.selected else self.minimax_agent
                
                # Train for one episode
                state = train_env.reset()
                done = False
                episode_reward = 0
                
                # Determine who goes first - increased probability for agent to go first
                if random.random() < 0.7:  # 70% chance agent goes first
                    # Agent goes first
                    while not done and self.training:
                        # Agent's turn
                        action = self.rl_agent.choose_action(state, training=True)
                        if action is None:
                            break
                        next_state, reward, done = train_env.make_move(action)
                        
                        if done:
                            # Enhanced reward for winning
                            if reward > 0:
                                reward *= 1.5  # Boost winning reward
                            self.rl_agent.learn(state, action, reward, next_state, done)
                            episode_reward += reward
                            break
                        
                        # Opponent's turn
                        opp_action = train_opponent.choose_action(next_state)
                        if opp_action is None:
                            break
                        next_next_state, opp_reward, done = train_env.make_move(opp_action)
                        
                        # Agent learns from opponent's action with penalty
                        penalty = -opp_reward * 1.5 if done and opp_reward > 0 else -opp_reward
                        self.rl_agent.learn(state, action, penalty, next_next_state, done)
                        episode_reward += penalty
                        state = next_next_state
                else:
                    # Opponent goes first
                    while not done and self.training:
                        # Opponent's turn
                        opp_action = train_opponent.choose_action(state)
                        if opp_action is None:
                            break
                        next_state, opp_reward, done = train_env.make_move(opp_action)
                        
                        if done:
                            # Harsh penalty for losing
                            if opp_reward > 0:
                                self.rl_agent.learn(state, None, -2.0, next_state, done)
                                episode_reward -= 2.0
                            break
                        
                        # Agent's turn
                        action = self.rl_agent.choose_action(next_state, training=True)
                        if action is None:
                            break
                        next_next_state, reward, done = train_env.make_move(action)
                        
                        # Enhanced reward for winning from second position
                        if done and reward > 0:
                            reward *= 2.0  # Extra reward for winning from second position
                        
                        self.rl_agent.learn(next_state, action, reward, next_next_state, done)
                        episode_reward += reward
                        state = next_next_state
                
                # Update agent's training history
                self.rl_agent.update_training_history(episode_reward)
                self.episode_count += 1
                self.rl_agent.decrease_exploration(factor=0.9998)
            
            # Update history
            self.win_rate_history.append(self.rl_agent.get_win_rate())
            
            # Limit history length
            if len(self.win_rate_history) > MAX_HISTORY_POINTS:
                step = len(self.win_rate_history) // MAX_HISTORY_POINTS
                self.win_rate_history = self.win_rate_history[::step]
            
            # Save agent periodically during training
            if self.episode_count % 1000 == 0:
                self.rl_agent.save()
            
            # Draw the UI
            self.draw_ui()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.training = False
            
            # Control training speed
            time.sleep(self.training_speed)
        
        # Reset exploration rate to previous value after training
        self.rl_agent.exploration_rate = old_exploration
        self.training = False
        self.rl_agent.save()  # Save the agent after training
        self.reset_game()
    
    def save_agent(self):
        self.rl_agent.save()
        print("Agent saved successfully")
    
    def load_agent(self):
        if self.rl_agent.load():
            print("Agent loaded successfully")
            self.win_rate_history = [self.rl_agent.get_win_rate()]
        else:
            print("No saved agent found")
    
    def update_opponent(self):
        if self.rb_random.selected:
            self.opponent = self.random_agent
        else:
            self.opponent = self.minimax_agent
    
    def update_player(self):
        if self.rb_play_x.selected:
            self.human_player = 1
            self.rl_agent.player = -1
            self.random_agent.player = 1
            self.minimax_agent.player = 1
        else:
            self.human_player = -1
            self.rl_agent.player = 1
            self.random_agent.player = -1
            self.minimax_agent.player = -1
        self.reset_game()
    
    def run(self):
        running = True
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Get mouse position
                mouse_pos = pygame.mouse.get_pos()
                
                # Handle button hovers
                for button in self.buttons.values():
                    button.is_hovered(mouse_pos)
                
                # Handle clicks
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check board clicks
                    self.handle_click(mouse_pos)
                    
                    # Check button clicks
                    if self.buttons["train"].is_clicked(mouse_pos, event) and not self.training:
                        self.train_rl_agent()
                    
                    elif self.buttons["reset"].is_clicked(mouse_pos, event) and not self.training:
                        self.reset_game()
                    
                    elif self.buttons["save"].is_clicked(mouse_pos, event) and not self.training:
                        self.save_agent()
                    
                    elif self.buttons["load"].is_clicked(mouse_pos, event) and not self.training:
                        self.load_agent()
                    
                    # Check radio button clicks
                    for rb in self.opponent_group:
                        if rb.is_clicked(mouse_pos, event):
                            rb.select()
                            self.update_opponent()
                    
                    for rb in self.player_group:
                        if rb.is_clicked(mouse_pos, event):
                            rb.select()
                            self.update_player()
            
            # Draw the UI
            self.draw_ui()
            
            # Cap the frame rate
            pygame.time.Clock().tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run()