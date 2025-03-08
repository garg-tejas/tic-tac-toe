import random
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

# Base screen dimensions - will be adjusted when resizing
BASE_SCREEN_WIDTH = 1200
BASE_SCREEN_HEIGHT = 800

# Initialize the screen with resizable flag
screen = pygame.display.set_mode((BASE_SCREEN_WIDTH, BASE_SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Tic Tac Toe - RL Agent")

# Function to calculate responsive layout dimensions
def calculate_layout(width, height):
    layout = {}
    
    # Set the board size based on the smaller dimension with padding
    layout['board_size'] = min(width, height) * 0.5
    layout['cell_size'] = layout['board_size'] // 3
    
    # Position the board centered on the left side of the screen
    layout['board_x'] = width * 0.1
    layout['board_y'] = (height - layout['board_size']) // 2
    
    # Button dimensions - scale with screen size
    layout['button_width'] = width * 0.15
    layout['button_height'] = height * 0.06
    layout['button_margin'] = width * 0.02
    
    # Training visualization settings
    layout['train_history_width'] = width * 0.35
    layout['train_history_height'] = height * 0.3
    layout['train_graph_x'] = layout['board_x'] + layout['board_size'] + width * 0.05
    layout['train_graph_y'] = layout['board_y']
    
    # Make column 2 (controls) at fixed position relative to board
    layout['controls_x'] = layout['train_graph_x']
    layout['controls_y'] = layout['train_graph_y'] + layout['train_history_height'] + height * 0.05
    
    # Font sizes scaled to screen height
    layout['small_font_size'] = int(height * 0.022)
    layout['normal_font_size'] = int(height * 0.03)
    layout['large_font_size'] = int(height * 0.045)
    layout['title_font_size'] = int(height * 0.06)
    
    return layout

# Global layout dictionary
layout = calculate_layout(BASE_SCREEN_WIDTH, BASE_SCREEN_HEIGHT)

# Fonts - will be updated when screen is resized
font = pygame.font.SysFont("Arial", layout['normal_font_size'])
small_font = pygame.font.SysFont("Arial", layout['small_font_size'])
large_font = pygame.font.SysFont("Arial", layout['large_font_size'])
title_font = pygame.font.SysFont("Arial", layout['title_font_size'], bold=True)

# Maximum number of history points to display
MAX_HISTORY_POINTS = 100


class Button:
    def __init__(self, x, y, width, height, text, color=BLUE, hover_color=DARK_BLUE, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        # Store original values for responsive resizing
        self.original_x = x
        self.original_y = y
        self.original_width = width
        self.original_height = height
    
    def update_position(self, x, y, width, height):
        """Update button position and size when screen is resized"""
        self.rect = pygame.Rect(x, y, width, height)
    
    def draw(self, surface, current_font=None):
        if current_font is None:
            current_font = font
            
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=10)
        
        text_surf = current_font.render(self.text, True, self.text_color)
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
        # Store original values for responsive resizing
        self.original_x = x
        self.original_y = y
        if group is not None:
            group.append(self)
    
    def update_position(self, x, y, radius=10):
        """Update radio button position when screen is resized"""
        self.rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
    
    def draw(self, surface, current_font=None):
        if current_font is None:
            current_font = small_font
            
        # Draw outer circle
        radius = self.rect.width // 2
        pygame.draw.circle(surface, BLACK, self.rect.center, radius, 2)
        
        # Draw filled circle if selected
        if self.selected:
            pygame.draw.circle(surface, BLACK, self.rect.center, radius * 0.6)
        
        # Draw text label
        text_surf = current_font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(midleft=(self.rect.right + radius, self.rect.centery))
        surface.blit(text_surf, text_rect)
    
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            radius = self.rect.width // 2
            # Check if click is within the circle
            if (pos[0] - self.rect.centerx) ** 2 + (pos[1] - self.rect.centery) ** 2 <= radius * radius:
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
        self.rl_agent = QAgent(player=1)  # RL agent plays X
        self.random_agent = RandomAgent(player=-1)  # Random agent plays O
        self.minimax_agent = MiniMaxAgent(player=-1)  # MiniMax agent plays O
        
        # Try to load a pre-trained agent
        self.rl_agent.load()
        
        self.opponent = self.random_agent  # Default opponent
        self.training = False
        self.training_speed = 0.05  # Made training faster for quicker results
        self.episode_count = 0
        self.target_episodes = 5000
        self.training_history = []
        self.win_rate_history = []
        
        self.human_player = -1  # Human plays O by default
        self.current_player = 1  # X starts (RL agent goes first)
        self.game_over = False
        self.result_message = ""
        
        # Current layout (will be updated when window is resized)
        self.current_layout = layout
        
        # Create UI elements
        self.create_buttons()
        
        # Radio button groups
        self.opponent_group = []
        self.player_group = []
        
        # Create radio buttons for opponent selection
        self.rb_random = RadioButton(
            self.current_layout['controls_x'], 
            self.current_layout['controls_y'] + 30, 
            "Random Opponent", 
            self.opponent_group, 
            True
        )
        
        self.rb_minimax = RadioButton(
            self.current_layout['controls_x'], 
            self.current_layout['controls_y'] + 60, 
            "MiniMax Opponent", 
            self.opponent_group
        )
        
        # Create radio buttons for player selection
        self.rb_play_o = RadioButton(
            self.current_layout['controls_x'], 
            self.current_layout['controls_y'] + 120, 
            "Play as O", 
            self.player_group, 
            True
        )
        
        self.rb_play_x = RadioButton(
            self.current_layout['controls_x'], 
            self.current_layout['controls_y'] + 150, 
            "Play as X", 
            self.player_group, 
            False
        )
    
    def create_buttons(self):
        """Create buttons with positions based on current layout"""
        l = self.current_layout
        
        self.buttons = {
            "train": Button(
                l['board_x'], 
                l['board_y'] + l['board_size'] + l['button_margin'], 
                l['button_width'], 
                l['button_height'], 
                "Train Agent", 
                GREEN, 
                DARK_GREEN
            ),
            "reset": Button(
                l['board_x'] + l['button_width'] + l['button_margin'], 
                l['board_y'] + l['board_size'] + l['button_margin'], 
                l['button_width'], 
                l['button_height'], 
                "Reset Game", 
                BLUE
            ),
            "save": Button(
                l['controls_x'], 
                l['controls_y'] + 200, 
                l['button_width'], 
                l['button_height'], 
                "Save Agent", 
                GREEN, 
                DARK_GREEN
            ),
            "load": Button(
                l['controls_x'] + l['button_width'] + l['button_margin'], 
                l['controls_y'] + 200, 
                l['button_width'], 
                l['button_height'], 
                "Load Agent", 
                BLUE
            )
        }
        
    def update_layout(self, width, height):
        """Update the layout when the window is resized"""
        # Calculate new layout based on window dimensions
        self.current_layout = calculate_layout(width, height)
        l = self.current_layout
        
        # Update button positions and sizes
        self.buttons["train"].update_position(
            l['board_x'], 
            l['board_y'] + l['board_size'] + l['button_margin'], 
            l['button_width'], 
            l['button_height']
        )
        
        self.buttons["reset"].update_position(
            l['board_x'] + l['button_width'] + l['button_margin'], 
            l['board_y'] + l['board_size'] + l['button_margin'], 
            l['button_width'], 
            l['button_height']
        )
        
        self.buttons["save"].update_position(
            l['controls_x'], 
            l['controls_y'] + 200, 
            l['button_width'], 
            l['button_height']
        )
        
        self.buttons["load"].update_position(
            l['controls_x'] + l['button_width'] + l['button_margin'], 
            l['controls_y'] + 200, 
            l['button_width'], 
            l['button_height']
        )
        
        # Update radio button positions
        radio_btn_radius = int(min(width, height) * 0.012)
        
        self.rb_random.update_position(
            l['controls_x'] + radio_btn_radius, 
            l['controls_y'] + 30, 
            radio_btn_radius
        )
        
        self.rb_minimax.update_position(
            l['controls_x'] + radio_btn_radius, 
            l['controls_y'] + 60, 
            radio_btn_radius
        )
        
        self.rb_play_o.update_position(
            l['controls_x'] + radio_btn_radius, 
            l['controls_y'] + 120, 
            radio_btn_radius
        )
        
        self.rb_play_x.update_position(
            l['controls_x'] + radio_btn_radius, 
            l['controls_y'] + 150, 
            radio_btn_radius
        )
        
        # Update fonts
        global font, small_font, large_font, title_font
        font = pygame.font.SysFont("Arial", l['normal_font_size'])
        small_font = pygame.font.SysFont("Arial", l['small_font_size'])
        large_font = pygame.font.SysFont("Arial", l['large_font_size'])
        title_font = pygame.font.SysFont("Arial", l['title_font_size'], bold=True)
    
    def draw_board(self):
        l = self.current_layout
        board_x = l['board_x']
        board_y = l['board_y']
        board_size = l['board_size']
        cell_size = l['cell_size']
        
        # Draw the board background
        pygame.draw.rect(screen, WHITE, (board_x, board_y, board_size, board_size))
        line_thickness = max(3, int(board_size * 0.008))
        pygame.draw.rect(screen, BLACK, (board_x, board_y, board_size, board_size), line_thickness)
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(screen, BLACK, 
                            (board_x + i * cell_size, board_y), 
                            (board_x + i * cell_size, board_y + board_size), 
                            line_thickness)
            # Horizontal lines
            pygame.draw.line(screen, BLACK, 
                            (board_x, board_y + i * cell_size), 
                            (board_x + board_size, board_y + i * cell_size), 
                            line_thickness)
        
        # Draw X's and O's
        for i in range(3):
            for j in range(3):
                cell_value = self.game.board[i, j]
                if cell_value == 1:  # X
                    self.draw_x(board_x + j * cell_size, board_y + i * cell_size)
                elif cell_value == -1:  # O
                    self.draw_o(board_x + j * cell_size, board_y + i * cell_size)
    
    def draw_x(self, x, y):
        l = self.current_layout
        cell_size = l['cell_size']
        
        # Scale margin and line thickness with cell size
        margin = int(cell_size * 0.15)
        thickness = max(4, int(cell_size * 0.05))
        
        # Draw X
        pygame.draw.line(screen, RED, 
                         (x + margin, y + margin), 
                         (x + cell_size - margin, y + cell_size - margin), 
                         thickness)
        pygame.draw.line(screen, RED, 
                         (x + cell_size - margin, y + margin), 
                         (x + margin, y + cell_size - margin), 
                         thickness)
    
    def draw_o(self, x, y):
        l = self.current_layout
        cell_size = l['cell_size']
        
        # Scale margin and thickness with cell size
        margin = int(cell_size * 0.15)
        thickness = max(4, int(cell_size * 0.05))
        
        # Draw O
        center = (x + cell_size // 2, y + cell_size // 2)
        radius = cell_size // 2 - margin
        pygame.draw.circle(screen, BLUE, center, radius, thickness)
    
    def draw_training_stats(self):
        l = self.current_layout
        train_x = l['train_graph_x']
        train_y = l['train_graph_y']
        train_width = l['train_history_width']
        train_height = l['train_history_height']
        padding = int(train_width * 0.025)
        
        # Draw training stats background
        pygame.draw.rect(screen, LIGHT_BLUE, 
                        (train_x, train_y, train_width, train_height))
        pygame.draw.rect(screen, BLACK, 
                        (train_x, train_y, train_width, train_height), 2)
        
        # Draw title
        title = small_font.render("Training Progress", True, BLACK)
        screen.blit(title, (train_x + padding, train_y + padding))
        
        # Line height based on font size
        line_height = l['small_font_size'] * 1.5
        
        # Draw win rate
        win_rate = self.rl_agent.get_win_rate()
        win_text = small_font.render(f"Win Rate: {win_rate:.2f}", True, BLACK)
        screen.blit(win_text, (train_x + padding, train_y + padding + line_height))
        
        # Draw episode count
        episode_text = small_font.render(f"Episodes: {self.episode_count}/{self.target_episodes}", True, BLACK)
        screen.blit(episode_text, (train_x + padding, train_y + padding + line_height * 2))
        
        # Draw exploration rate
        explore_text = small_font.render(f"Exploration Rate: {self.rl_agent.exploration_rate:.4f}", True, BLACK)
        screen.blit(explore_text, (train_x + padding, train_y + padding + line_height * 3))
        
        # Draw win/loss/draw counts
        stats = self.rl_agent.training_history
        stats_text = small_font.render(
            f"Wins: {stats['wins']}  Losses: {stats['losses']}  Draws: {stats['draws']}", 
            True, BLACK)
        screen.blit(stats_text, (train_x + padding, train_y + padding + line_height * 4))
        
        # Draw win rate graph
        if len(self.win_rate_history) > 1:
            graph_margin = padding * 2
            graph_width = train_width - graph_margin * 2
            graph_height = train_height * 0.35
            graph_x = train_x + graph_margin
            graph_y = train_y + train_height * 0.6
            
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
        # Get current window size
        width, height = screen.get_size()
        
        # Clear screen
        screen.fill(GRAY)
        
        # Draw title
        title_text = title_font.render("Tic Tac Toe RL", True, BLACK)
        screen.blit(title_text, (width // 2 - title_text.get_width() // 2, height * 0.03))
        
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
        screen.blit(subtitle, (width // 2 - subtitle.get_width() // 2, height * 0.08))
        
        # Draw board
        self.draw_board()
        
        # Draw training stats
        self.draw_training_stats()
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(screen, font)
        
        # Draw section titles
        l = self.current_layout
        
        opponent_title = font.render("Opponent Selection:", True, BLACK)
        screen.blit(opponent_title, (l['controls_x'], l['controls_y']))
        
        player_title = font.render("Player Selection:", True, BLACK)
        screen.blit(player_title, (l['controls_x'], l['controls_y'] + 90))
        
        # Draw radio buttons
        for rb in self.opponent_group:
            rb.draw(screen, small_font)
        for rb in self.player_group:
            rb.draw(screen, small_font)
        
        # Update the screen
        pygame.display.flip()
    
    def handle_click(self, pos):
        if self.training:
            return
        
        l = self.current_layout
        board_x = l['board_x']
        board_y = l['board_y']
        board_size = l['board_size']
        cell_size = l['cell_size']
        
        # Check if clicked on board
        if (board_x <= pos[0] <= board_x + board_size and 
            board_y <= pos[1] <= board_y + board_size and 
            not self.game_over):
            
            # Convert click position to board coordinates
            j = int((pos[0] - board_x) // cell_size)
            i = int((pos[1] - board_y) // cell_size)
            
            # Ensure coordinates are within bounds
            if 0 <= i < 3 and 0 <= j < 3:
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
        
        # Get the agent's action with look-ahead for RL agent only
        if agent == self.rl_agent:
            action = agent.choose_action(self.game.board, training=False, look_ahead=True)
        else:
            action = agent.choose_action(self.game.board, training=False)
        
        if action is not None:
            # Make the move
            self.make_move(action)
    
    def reset_game(self):
        self.game.reset()
        self.current_player = 1  # X starts
        self.game_over = False
        self.result_message = ""
        
        # If it's the AI's turn, make a move
        if self.current_player != self.human_player:
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
        clock = pygame.time.Clock()
        
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle window resize events
                elif event.type == pygame.VIDEORESIZE:
                    # Update screen size and recalculate layout
                    self.update_layout(event.w, event.h)
                
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
                
                # Handle key presses
                elif event.type == pygame.KEYDOWN:
                    # Press F to toggle fullscreen
                    if event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()
            
            # Draw the UI
            self.draw_ui()
            
            # Cap the frame rate
            clock.tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run()