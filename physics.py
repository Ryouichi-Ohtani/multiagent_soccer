"""
Physics engine for soccer game
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SoccerEnvironmentConfig

class Ball:
    def __init__(self, x: float, y: float, radius: float = 10):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.radius = radius

    def update(self, config: SoccerEnvironmentConfig):
        """Update ball position with physics"""
        # Apply velocity
        self.pos += self.vel

        # Apply ball decay (friction)
        self.vel *= config.BALL_DECAY

        # Boundary collision detection
        field_width, field_height = config.FIELD_SIZE

        # Horizontal boundaries (top/bottom)
        if self.pos[1] <= self.radius or self.pos[1] >= field_height - self.radius:
            self.vel[1] *= -0.8  # Energy loss on collision
            self.pos[1] = max(self.radius, min(field_height - self.radius, self.pos[1]))

        # Vertical boundaries (left/right - goals)
        goal_top = (field_height - config.GOAL_SIZE[1]) // 2
        goal_bottom = goal_top + config.GOAL_SIZE[1]

        # Left side
        if self.pos[0] <= self.radius:
            if goal_top <= self.pos[1] <= goal_bottom:
                # Goal scored
                return "goal_left"
            else:
                self.vel[0] *= -0.8
                self.pos[0] = self.radius

        # Right side
        elif self.pos[0] >= field_width - self.radius:
            if goal_top <= self.pos[1] <= goal_bottom:
                # Goal scored
                return "goal_right"
            else:
                self.vel[0] *= -0.8
                self.pos[0] = field_width - self.radius

        return None

class Player:
    def __init__(self, x: float, y: float, team: int, player_id: int, radius: float = 20):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.team = team
        self.player_id = player_id
        self.radius = radius
        self.has_ball = False

    def update(self, action: np.ndarray, config: SoccerEnvironmentConfig):
        """Update player position based on action"""
        # Extract movement and kick actions
        move_x, move_y = action[0], action[1]
        kick_power = action[2] if len(action) > 2 else 0.0
        kick_dir_x = action[3] if len(action) > 3 else 0.0
        kick_dir_y = action[4] if len(action) > 4 else 0.0

        # Apply movement
        movement = np.array([move_x, move_y]) * config.PLAYER_SPEED
        self.vel = movement
        self.pos += self.vel

        # Apply friction
        self.vel *= config.FRICTION

        # Boundary constraints
        field_width, field_height = config.FIELD_SIZE
        self.pos[0] = max(self.radius, min(field_width - self.radius, self.pos[0]))
        self.pos[1] = max(self.radius, min(field_height - self.radius, self.pos[1]))

        return kick_power, np.array([kick_dir_x, kick_dir_y])

class PhysicsEngine:
    def __init__(self, config: SoccerEnvironmentConfig):
        self.config = config
        self.ball = Ball(
            config.FIELD_SIZE[0] // 2,
            config.FIELD_SIZE[1] // 2,
            config.BALL_RADIUS
        )

        # Initialize players
        self.players = []
        self._init_players()

    def _init_players(self):
        """Initialize player positions"""
        field_width, field_height = self.config.FIELD_SIZE

        # Team 0 (left side - blue)
        self.players.append(Player(field_width * 0.2, field_height * 0.3, 0, 0))
        self.players.append(Player(field_width * 0.2, field_height * 0.7, 0, 1))

        # Team 1 (right side - red)
        self.players.append(Player(field_width * 0.8, field_height * 0.3, 1, 0))
        self.players.append(Player(field_width * 0.8, field_height * 0.7, 1, 1))

    def reset(self):
        """Reset physics state"""
        self.ball.pos = np.array([
            self.config.FIELD_SIZE[0] // 2,
            self.config.FIELD_SIZE[1] // 2
        ], dtype=float)
        self.ball.vel = np.array([0.0, 0.0], dtype=float)

        # Reset player positions
        field_width, field_height = self.config.FIELD_SIZE
        positions = [
            (field_width * 0.2, field_height * 0.3),  # Team 0, Player 0
            (field_width * 0.2, field_height * 0.7),  # Team 0, Player 1
            (field_width * 0.8, field_height * 0.3),  # Team 1, Player 0
            (field_width * 0.8, field_height * 0.7),  # Team 1, Player 1
        ]

        for i, (x, y) in enumerate(positions):
            self.players[i].pos = np.array([x, y], dtype=float)
            self.players[i].vel = np.array([0.0, 0.0], dtype=float)
            self.players[i].has_ball = False

    def step(self, actions: Dict[str, np.ndarray]) -> Optional[str]:
        """Step physics simulation"""
        # Update players
        kicks = {}
        for i, player in enumerate(self.players):
            agent_key = f"player_{i}"
            if agent_key in actions:
                kick_power, kick_dir = player.update(actions[agent_key], self.config)
                if kick_power > 0:
                    kicks[i] = (kick_power, kick_dir)

        # Check player collisions with ball and apply kicks
        ball_touched_by = None
        for i, player in enumerate(self.players):
            dist = np.linalg.norm(player.pos - self.ball.pos)
            if dist <= player.radius + self.ball.radius:
                ball_touched_by = i
                player.has_ball = True

                # Apply kick if player is kicking
                if i in kicks:
                    kick_power, kick_dir = kicks[i]
                    kick_dir = kick_dir / (np.linalg.norm(kick_dir) + 1e-8)  # Normalize
                    self.ball.vel += kick_dir * kick_power * self.config.BALL_SPEED_MULTIPLIER
            else:
                player.has_ball = False

        # Update ball
        goal_result = self.ball.update(self.config)

        # Handle player-player collisions
        self._handle_player_collisions()

        return goal_result, ball_touched_by

    def _handle_player_collisions(self):
        """Handle collisions between players"""
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                p1, p2 = self.players[i], self.players[j]
                dist = np.linalg.norm(p1.pos - p2.pos)

                if dist < p1.radius + p2.radius:
                    # Separate players
                    direction = p1.pos - p2.pos
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    overlap = (p1.radius + p2.radius) - dist

                    p1.pos += direction * overlap * 0.5
                    p2.pos -= direction * overlap * 0.5

    def get_state(self) -> Dict:
        """Get current state of all entities"""
        return {
            'ball': {
                'pos': self.ball.pos.copy(),
                'vel': self.ball.vel.copy()
            },
            'players': [
                {
                    'pos': player.pos.copy(),
                    'vel': player.vel.copy(),
                    'team': player.team,
                    'player_id': player.player_id,
                    'has_ball': player.has_ball
                }
                for player in self.players
            ]
        }