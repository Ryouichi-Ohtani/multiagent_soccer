"""
Renderer for soccer game visualization using pygame
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SoccerEnvironmentConfig

class SoccerRenderer:
    def __init__(self, config: SoccerEnvironmentConfig, window_size: Tuple[int, int] = None):
        self.config = config
        self.window_size = window_size or config.FIELD_SIZE

        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Multi-Agent Soccer Game")

        # Colors
        self.colors = {
            'field': (0, 128, 0),        # Green
            'field_lines': (255, 255, 255),  # White
            'ball': (255, 255, 255),     # White
            'team_0': (0, 0, 255),       # Blue
            'team_1': (255, 0, 0),       # Red
            'goal': (128, 128, 128),     # Gray
            'background': (0, 64, 0),    # Dark green
        }

        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

    def render(self, state: Dict, scores: Tuple[int, int] = (0, 0), step: int = 0) -> bool:
        """
        Render the current state
        Returns True if rendering should continue, False if window was closed
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Clear screen
        self.screen.fill(self.colors['background'])

        # Draw field
        self._draw_field()

        # Draw players
        self._draw_players(state['players'])

        # Draw ball
        self._draw_ball(state['ball'])

        # Draw UI
        self._draw_ui(scores, step)

        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

        return True

    def _draw_field(self):
        """Draw soccer field with goals and center line"""
        field_width, field_height = self.config.FIELD_SIZE

        # Field background
        field_rect = pygame.Rect(0, 0, field_width, field_height)
        pygame.draw.rect(self.screen, self.colors['field'], field_rect)

        # Field border
        pygame.draw.rect(self.screen, self.colors['field_lines'], field_rect, 3)

        # Center line
        center_x = field_width // 2
        pygame.draw.line(self.screen, self.colors['field_lines'],
                        (center_x, 0), (center_x, field_height), 3)

        # Center circle
        pygame.draw.circle(self.screen, self.colors['field_lines'],
                          (center_x, field_height // 2), 100, 3)

        # Goals
        goal_width, goal_height = self.config.GOAL_SIZE
        goal_top = (field_height - goal_height) // 2
        goal_bottom = goal_top + goal_height

        # Left goal
        left_goal = pygame.Rect(-goal_width//2, goal_top, goal_width, goal_height)
        pygame.draw.rect(self.screen, self.colors['goal'], left_goal)
        pygame.draw.rect(self.screen, self.colors['field_lines'], left_goal, 3)

        # Right goal
        right_goal = pygame.Rect(field_width - goal_width//2, goal_top, goal_width, goal_height)
        pygame.draw.rect(self.screen, self.colors['goal'], right_goal)
        pygame.draw.rect(self.screen, self.colors['field_lines'], right_goal, 3)

        # Goal areas (penalty boxes)
        penalty_width, penalty_height = 120, 200
        penalty_top = (field_height - penalty_height) // 2

        # Left penalty box
        left_penalty = pygame.Rect(0, penalty_top, penalty_width, penalty_height)
        pygame.draw.rect(self.screen, self.colors['field_lines'], left_penalty, 2)

        # Right penalty box
        right_penalty = pygame.Rect(field_width - penalty_width, penalty_top,
                                   penalty_width, penalty_height)
        pygame.draw.rect(self.screen, self.colors['field_lines'], right_penalty, 2)

    def _draw_players(self, players: List[Dict]):
        """Draw all players"""
        for i, player in enumerate(players):
            pos = player['pos']
            team = player['team']
            has_ball = player['has_ball']

            color = self.colors[f'team_{team}']

            # Draw player circle
            pygame.draw.circle(self.screen, color, pos.astype(int), self.config.PLAYER_RADIUS)

            # Draw player outline
            outline_color = (255, 255, 255) if has_ball else (0, 0, 0)
            outline_width = 4 if has_ball else 2
            pygame.draw.circle(self.screen, outline_color, pos.astype(int),
                             self.config.PLAYER_RADIUS, outline_width)

            # Draw player number
            player_text = self.font.render(str(player['player_id']), True, (255, 255, 255))
            text_rect = player_text.get_rect(center=pos.astype(int))
            self.screen.blit(player_text, text_rect)

    def _draw_ball(self, ball: Dict):
        """Draw the ball"""
        pos = ball['pos']
        pygame.draw.circle(self.screen, self.colors['ball'], pos.astype(int), self.config.BALL_RADIUS)
        pygame.draw.circle(self.screen, (0, 0, 0), pos.astype(int), self.config.BALL_RADIUS, 2)

        # Draw ball velocity vector (for debugging)
        vel = ball['vel']
        if np.linalg.norm(vel) > 0.1:
            end_pos = pos + vel * 10  # Scale for visibility
            pygame.draw.line(self.screen, (255, 255, 0), pos.astype(int), end_pos.astype(int), 2)

    def _draw_ui(self, scores: Tuple[int, int], step: int):
        """Draw game UI (scores, step counter)"""
        # Score display
        score_text = f"Blue: {scores[0]}  Red: {scores[1]}"
        score_surface = self.font.render(score_text, True, (255, 255, 255))
        self.screen.blit(score_surface, (10, 10))

        # Step counter
        step_text = f"Step: {step}"
        step_surface = self.font.render(step_text, True, (255, 255, 255))
        step_rect = step_surface.get_rect()
        step_rect.topright = (self.window_size[0] - 10, 10)
        self.screen.blit(step_surface, step_rect)

    def close(self):
        """Close the renderer"""
        pygame.quit()

    def save_frame(self, filename: str):
        """Save current frame as image"""
        pygame.image.save(self.screen, filename)

class VideoRecorder:
    """Record gameplay videos"""
    def __init__(self, filename: str, fps: int = 30):
        self.filename = filename
        self.fps = fps
        self.frames = []

    def add_frame(self, surface):
        """Add a frame to the video"""
        frame_array = pygame.surfarray.array3d(surface)
        frame_array = np.transpose(frame_array, (1, 0, 2))  # Correct orientation
        self.frames.append(frame_array)

    def save_video(self):
        """Save recorded frames as video (requires opencv)"""
        if not self.frames:
            return

        try:
            import cv2
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))

            for frame in self.frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"Video saved as {self.filename}")
        except ImportError:
            print("OpenCV not available. Cannot save video.")

    def clear(self):
        """Clear recorded frames"""
        self.frames = []