from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, q_function

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
CLOSER_TO_COIN_EVENT = "CLOSER_TO_COIN"
REMOVED_FROM_COIN_EVENT = "REMOVED_FROM_COIN"
INCREASED_NEIGHBORHOOD_COINS_EVENT = "INCREASED_NEIGHBORHOOD_COINS"
DECREASED_NEIGHBORHOOD_COINS_EVENT = "DECREASED_NEIGHBORHOOD_COINS"
INCREASED_REACHABLE_COINS_EVENT = "INCREASED_REACHABLE_COINS_EVENT"
NO_PROGRESS_EVENT = "NO_PROGRESS"

ACTIONS_TO_NUMBERS = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5
}



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.statistics = []
    self.collected_coins = 0

    self.learning_rate = 0.01
    self.min_learning_rate = 0.00001


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # Idea: Add your own events to hand out rewards
    if new_features[0] > old_features[0]:
        events.append(CLOSER_TO_COIN_EVENT)

    if new_features[0] < old_features[0]:               #TODO: Check if a coin was collected, the next one will be further away, but it shouldn't be a penalty (also for neighborhood and reachable coins)
        events.append(REMOVED_FROM_COIN_EVENT)

    if new_features[1] > old_features[1]:
        events.append(INCREASED_NEIGHBORHOOD_COINS_EVENT)

    if new_features[1] < old_features[1]:
        events.append(DECREASED_NEIGHBORHOOD_COINS_EVENT)

    if new_features[2] > old_features[2]:
        events.append(INCREASED_REACHABLE_COINS_EVENT)

    if np.all(np.equal(new_features[:3], old_features[:3])):
        events.append(NO_PROGRESS_EVENT)


    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


    #learning part -------------------------------------------------------------------------------------------------------------------------

    #hyperparameters
    discount_factor = 0.6
    learning_rate = self.learning_rate

    #train the model
    weights = self.model
    #print(events)
    reward = reward_from_events(self, events)
    #print(reward)
    greedy_action = choose_greedy_action(new_game_state, weights)
    action_number = ACTIONS_TO_NUMBERS[self_action]

    #SARSA
    #weights[action_number] = weights[action_number] + learning_rate * state_to_features(new_game_state) * ((reward + discount_factor * q_function(new_game_state, greedy_action, weights)) - q_function(old_game_state, self_action, weights))
    weights[action_number] = weights[action_number] + learning_rate * state_to_features(old_game_state) * ((reward + discount_factor * q_function(new_game_state, greedy_action, weights)) - q_function(old_game_state, self_action, weights))
    #weights[action_number] = weights[action_number] + learning_rate * ((reward + discount_factor * q_function(new_game_state, greedy_action, weights)) - q_function(old_game_state, self_action, weights))

    #print(weights)
    #update the model
    self.model = weights
    #print(weights)

    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1

    #if(learning_rate >= self.min_learning_rate):
    #    self.learning_rate *= 0.9995
    self.learning_rate *= 0.9995

def choose_greedy_action(state, weights):
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    expected_q_values = np.array([q_function(state, action, weights) for action in ACTIONS])

    greedy_action = ACTIONS[np.argmax(expected_q_values)]

    return greedy_action

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    self.statistics.append([last_game_state['round'], last_game_state['step'], self.collected_coins])
    print(self.statistics)
    self.collected_coins = 0

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -5,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        CLOSER_TO_COIN_EVENT: 1,
        REMOVED_FROM_COIN_EVENT: -0.5,
        INCREASED_NEIGHBORHOOD_COINS_EVENT: 3,
        DECREASED_NEIGHBORHOOD_COINS_EVENT: -0.1,
        INCREASED_REACHABLE_COINS_EVENT: 1,
        NO_PROGRESS_EVENT: -3
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
