import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# Game settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 800
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 60
PIPE_HEIGHT = 750
GAP_SIZE = 200

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.bird_image = pygame.image.load("./assets/mr-flappy.png")
        self.top_pipe_image = pygame.image.load("./assets/mr-pipe copy.png")
        self.bottom_pipe_image = pygame.image.load("./assets/mr-pipe.png")
        self.bird_image = pygame.transform.scale(self.bird_image, (BIRD_WIDTH, BIRD_HEIGHT))
        self.top_pipe_image = pygame.transform.scale(self.top_pipe_image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.bottom_pipe_image = pygame.transform.scale(self.bottom_pipe_image, (PIPE_WIDTH, PIPE_HEIGHT))

        self.pipes_passed = 0
        
        # Actions: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation space (you can expand this as needed)
        self.observation_space = spaces.Box(low=0, high=SCREEN_HEIGHT, shape=(4,), dtype=np.float32)

        # Initialize Pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):
        # Reset bird
        self.bird = {
            "x": 50,
            "y": SCREEN_HEIGHT // 2,
            "velocity": 0,
            "gravity": 3,
            "lift": -20
        }
        
        # Reset pipes
        self.pipes = [self._create_pipe()]
        
        # Initial score
        self.score = 0
        self.done = False

        # Initial state (e.g., bird position, velocity, next pipe distance)
        return self._get_state(), {}

    def step(self, action):
        # Bird flaps if action is 1
        if action == 1:
            self.bird["velocity"] = self.bird["lift"]

        # Apply gravity to bird
        self.bird["velocity"] += self.bird["gravity"]
        self.bird["y"] += self.bird["velocity"]

        # Update pipes and score
        if self.pipes[-1]["x"] < SCREEN_WIDTH // 1.6:
            self.pipes.append(self._create_pipe())
        for pipe in self.pipes:
            pipe["x"] -= 3
        
        # Check for collision or pass-through
        reward = 0
        if self._check_collision():
            self.done = True
            reward = -10  # Penalty for crashing
        else:
            for pipe in self.pipes:
                if pipe["x"] + PIPE_WIDTH < self.bird["x"] and not pipe["passed"]:
                    pipe["passed"] = True
                    self.score += 1
                    reward = 10  # Reward for passing a pipe
                    self.pipes_passed += 1

        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe["x"] + PIPE_WIDTH > 0]
        
        return self._get_state(), reward, self.done, False, {}

    def _get_state(self):
        # Define state as bird's position, velocity, and distance to next pipe
        next_pipe = next(pipe for pipe in self.pipes if not pipe["passed"])
        return np.array([
            self.bird["y"],
            self.bird["velocity"],
            next_pipe["x"] - self.bird["x"],
            next_pipe["height"] + GAP_SIZE
        ], dtype=np.float32)
    
    def draw_bird(self):
        # Draw the bird image at the bird's position
        self.screen.blit(self.bird_image, (self.bird["x"], self.bird["y"]))

    def draw_pipes(self):
        for pipe in self.pipes:
            # Draw top pipe
            top_pipe_y = pipe["height"] - PIPE_HEIGHT
            self.screen.blit(self.top_pipe_image, (pipe["x"], top_pipe_y))

            # Draw bottom pipe
            bottom_pipe_y = pipe["height"] + GAP_SIZE
            self.screen.blit(self.bottom_pipe_image, (pipe["x"], bottom_pipe_y))

    def render(self, mode="human"):
        self.screen.fill((0, 200, 0))

        # Draw bird
        self.draw_bird()

        # Draw pipes
        self.draw_pipes()

        # Display the number of pipes passed in the top left corner
        font = pygame.font.SysFont(None, 36)  # Define font and size
        score_text = font.render(f"Score: {self.pipes_passed}", True, (0, 0, 0))  # Render text in white
        self.screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()
        self.clock.tick(30)

    def _check_collision(self):
        # Check if bird hits the ground or ceiling
        if self.bird["y"] < 0 or self.bird["y"] + BIRD_HEIGHT > SCREEN_HEIGHT:
            return True
        # Check if bird hits a pipe
        for pipe in self.pipes:
            if pipe["x"] < self.bird["x"] < pipe["x"] + PIPE_WIDTH:
                if self.bird["y"] < pipe["height"] or self.bird["y"] > pipe["height"] + GAP_SIZE:
                    return True
        return False

    def _create_pipe(self):
        gap_borders = 50
        return {
            "x": SCREEN_WIDTH,
            "height": random.randint(gap_borders, SCREEN_HEIGHT - GAP_SIZE - gap_borders),
            "passed": False
        }

    def close(self):
        pygame.quit()