"""
Reward system for soccer environment with multi-objective reward function
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SoccerEnvironmentConfig

class RewardCalculator:
    """
    Multi-objective reward function for soccer agents
    """
    def __init__(self, config: SoccerEnvironmentConfig):
        self.config = config
        self.field_width, self.field_height = config.FIELD_SIZE

        # Reward weights
        self.reward_weights = {
            'goal_scored': 100.0,           # Goal scored by team
            'goal_conceded': -100.0,        # Goal conceded by team
            'ball_touch': 5.0,              # Touching the ball
            'goal_approach': 0.1,           # Moving closer to enemy goal
            'ball_approach': 0.05,          # Moving closer to ball
            'teamwork': 0.02,               # Team coordination
            'out_of_bounds': -10.0,         # Going out of bounds
            'stalemate': -0.1,              # Stalemate penalty
            'ball_possession': 0.01,        # Keeping ball possession
            'defensive_positioning': 0.01,   # Good defensive position
        }

        # Track previous states for delta calculations
        self.prev_states = {}

    def calculate_reward(self, agent_id: int, action: np.ndarray,
                        prev_state: Dict, current_state: Dict,
                        goal_scored: Optional[str] = None,
                        ball_touched_by: Optional[int] = None,
                        scores: Tuple[int, int] = (0, 0),
                        out_of_bounds_agents: List[int] = None) -> float:
        """
        Calculate multi-objective reward for agent
        """
        reward = 0.0
        agent_team = current_state['players'][agent_id]['team']
        out_of_bounds_agents = out_of_bounds_agents or []

        # 1. Goal rewards (most important)
        if goal_scored:
            if goal_scored == "goal_left" and agent_team == 1:  # Red team scored
                reward += self.reward_weights['goal_scored']
            elif goal_scored == "goal_right" and agent_team == 0:  # Blue team scored
                reward += self.reward_weights['goal_scored']
            elif goal_scored == "goal_left" and agent_team == 0:  # Blue conceded
                reward += self.reward_weights['goal_conceded']
            elif goal_scored == "goal_right" and agent_team == 1:  # Red conceded
                reward += self.reward_weights['goal_conceded']

        # 2. Ball contact reward
        if ball_touched_by == agent_id:
            reward += self.reward_weights['ball_touch']

        # 3. Goal approach reward
        goal_approach_reward = self.calculate_goal_approach_reward(
            agent_id, prev_state, current_state
        )
        reward += goal_approach_reward * self.reward_weights['goal_approach']

        # 4. Ball approach reward
        ball_approach_reward = self.calculate_ball_approach_reward(
            agent_id, prev_state, current_state
        )
        reward += ball_approach_reward * self.reward_weights['ball_approach']

        # 5. Teamwork reward
        teamwork_reward = self.calculate_teamwork_reward(agent_id, current_state)
        reward += teamwork_reward * self.reward_weights['teamwork']

        # 6. Penalties
        if agent_id in out_of_bounds_agents:
            reward += self.reward_weights['out_of_bounds']

        if self.is_stalemate(current_state):
            reward += self.reward_weights['stalemate']

        # 7. Ball possession reward
        if current_state['players'][agent_id]['has_ball']:
            reward += self.reward_weights['ball_possession']

        # 8. Defensive positioning reward
        defensive_reward = self.calculate_defensive_positioning_reward(
            agent_id, current_state
        )
        reward += defensive_reward * self.reward_weights['defensive_positioning']

        return reward

    def calculate_goal_approach_reward(self, agent_id: int, prev_state: Dict,
                                     current_state: Dict) -> float:
        """Calculate reward for approaching enemy goal"""
        agent_team = current_state['players'][agent_id]['team']
        current_pos = current_state['players'][agent_id]['pos']
        prev_pos = prev_state['players'][agent_id]['pos']

        # Enemy goal position
        if agent_team == 0:  # Blue team
            enemy_goal_pos = np.array([self.field_width, self.field_height / 2])
        else:  # Red team
            enemy_goal_pos = np.array([0, self.field_height / 2])

        prev_dist = np.linalg.norm(prev_pos - enemy_goal_pos)
        current_dist = np.linalg.norm(current_pos - enemy_goal_pos)

        return prev_dist - current_dist  # Positive if getting closer

    def calculate_ball_approach_reward(self, agent_id: int, prev_state: Dict,
                                     current_state: Dict) -> float:
        """Calculate reward for approaching ball"""
        current_pos = current_state['players'][agent_id]['pos']
        prev_pos = prev_state['players'][agent_id]['pos']
        ball_pos = current_state['ball']['pos']

        prev_dist = np.linalg.norm(prev_pos - ball_pos)
        current_dist = np.linalg.norm(current_pos - ball_pos)

        return prev_dist - current_dist  # Positive if getting closer

    def calculate_teamwork_reward(self, agent_id: int, current_state: Dict) -> float:
        """Calculate teamwork reward based on team coordination"""
        agent_team = current_state['players'][agent_id]['team']
        players = current_state['players']

        # Find teammate
        teammates = [p for i, p in enumerate(players)
                    if p['team'] == agent_team and i != agent_id]

        if not teammates:
            return 0.0

        teammate = teammates[0]
        agent_pos = current_state['players'][agent_id]['pos']
        teammate_pos = teammate['pos']

        # Optimal distance between teammates (100-200 pixels)
        teammate_dist = np.linalg.norm(agent_pos - teammate_pos)
        optimal_dist = 150
        dist_penalty = abs(teammate_dist - optimal_dist) / optimal_dist

        return 1.0 - dist_penalty  # Higher reward for optimal distance

    def calculate_defensive_positioning_reward(self, agent_id: int,
                                             current_state: Dict) -> float:
        """Calculate reward for good defensive positioning"""
        agent_team = current_state['players'][agent_id]['team']
        agent_pos = current_state['players'][agent_id]['pos']
        ball_pos = current_state['ball']['pos']

        # Own goal position
        if agent_team == 0:  # Blue team
            own_goal_pos = np.array([0, self.field_height / 2])
        else:  # Red team
            own_goal_pos = np.array([self.field_width, self.field_height / 2])

        # Reward for being between ball and own goal
        goal_to_ball = ball_pos - own_goal_pos
        goal_to_agent = agent_pos - own_goal_pos

        # Project agent position onto goal-ball line
        if np.linalg.norm(goal_to_ball) > 0:
            projection = np.dot(goal_to_agent, goal_to_ball) / np.linalg.norm(goal_to_ball)
            ball_dist = np.linalg.norm(goal_to_ball)

            # Reward if agent is between goal and ball
            if 0 < projection < ball_dist:
                return 1.0

        return 0.0

    def is_stalemate(self, state: Dict, threshold: float = 1.0) -> bool:
        """Check if the game is in a stalemate (low activity)"""
        ball_speed = np.linalg.norm(state['ball']['vel'])
        player_speeds = [np.linalg.norm(p['vel']) for p in state['players']]
        avg_player_speed = np.mean(player_speeds)

        return ball_speed < threshold and avg_player_speed < threshold

    def get_team_reward(self, team: int, individual_rewards: Dict[int, float]) -> float:
        """Calculate team reward from individual rewards"""
        team_players = [i for i in individual_rewards.keys()
                       if i // 2 == team]  # Assuming 2 players per team
        return sum(individual_rewards[i] for i in team_players) / len(team_players)

class RewardShaper:
    """
    Advanced reward shaping techniques
    """
    def __init__(self, config: SoccerEnvironmentConfig):
        self.config = config
        self.reward_calculator = RewardCalculator(config)

    def shaped_reward(self, agent_id: int, action: np.ndarray,
                     prev_state: Dict, current_state: Dict,
                     **kwargs) -> float:
        """Apply reward shaping for better learning"""
        base_reward = self.reward_calculator.calculate_reward(
            agent_id, action, prev_state, current_state, **kwargs
        )

        # Potential-based reward shaping
        potential_reward = self.calculate_potential_based_reward(
            agent_id, prev_state, current_state
        )

        return base_reward + potential_reward

    def calculate_potential_based_reward(self, agent_id: int,
                                       prev_state: Dict, current_state: Dict) -> float:
        """Calculate potential-based shaped reward"""
        agent_team = current_state['players'][agent_id]['team']

        # Potential functions
        prev_potential = self.calculate_potential(agent_id, prev_state)
        current_potential = self.calculate_potential(agent_id, current_state)

        # Potential-based shaping: F(s,a,s') = γΦ(s') - Φ(s)
        gamma = 0.99  # Discount factor
        return gamma * current_potential - prev_potential

    def calculate_potential(self, agent_id: int, state: Dict) -> float:
        """Calculate potential function value"""
        agent_team = state['players'][agent_id]['team']
        agent_pos = state['players'][agent_id]['pos']
        ball_pos = state['ball']['pos']

        # Potential based on distance to ball and enemy goal
        if agent_team == 0:  # Blue team
            enemy_goal_pos = np.array([self.config.FIELD_SIZE[0], self.config.FIELD_SIZE[1] / 2])
        else:  # Red team
            enemy_goal_pos = np.array([0, self.config.FIELD_SIZE[1] / 2])

        ball_dist = np.linalg.norm(agent_pos - ball_pos)
        goal_dist = np.linalg.norm(ball_pos - enemy_goal_pos)

        # Potential decreases with distance (encouraging approach)
        potential = -0.001 * ball_dist - 0.001 * goal_dist

        return potential