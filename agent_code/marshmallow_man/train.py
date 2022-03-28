from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, q_function, simulate_step

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
SAFE_STEP_EVENT = "SAFE_STEP"
I_LIKE_TO_DIE_EVENT = "I_LIKE_TO_DIE"
GOOD_CRATE_STEP_EVENT = "GOOD_CRATE_STEP"
GOOD_OFFENSIV_STEP_EVENT = "GOOD_OFFENSIV_STEP"
GOOD_DEFENSIV_STEP_EVENT = "GOOD_DEFENSIV_STEP"
BAD_STEP_EVENT	= "BAD_STEP"
GOOD_WAIT_EVENT = "GOOD_WAIT"
BAD_WAIT_EVENT = "BAD_WAIT"
GOOD_BOMB_DROP_EVENT = "GOOD_BOMB_DROP"
OKAY_BOMB_DROP_EVENT = "OKAY_BOMB_DROP"
BAD_BOMB_DROP_EVENT = "BAD_BOMB_DROP"

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

    self.learning_rate = 0.001
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

    coin_step = old_features[3:5]
    crate_step = old_features[5:7]
    safe_step = old_features[7:9]
    danger = old_features[9]
    offensive_step = old_features[10:12]
    defensive_step = old_features[12:14]
    is_escapable = old_features[14]
    bomb_available = old_features[15]

    #print(crate_step)

    if(danger):
        if performed_step(safe_step, self_action):
            events.append(SAFE_STEP_EVENT)
        else:
            events.append(I_LIKE_TO_DIE_EVENT)
    else:
        # Idea: Add your own events to hand out rewards
        if new_features[0] > old_features[0]:
            events.append(CLOSER_TO_COIN_EVENT)

        if new_features[0] < old_features[0]:
            events.append(REMOVED_FROM_COIN_EVENT)

        if new_features[1] > old_features[1]:
            events.append(INCREASED_NEIGHBORHOOD_COINS_EVENT)

        if new_features[1] < old_features[1]:
            events.append(DECREASED_NEIGHBORHOOD_COINS_EVENT)

        if new_features[2] > old_features[2]:
            events.append(INCREASED_REACHABLE_COINS_EVENT)

        if new_features[3] > old_features[3]:
            events.append(INCREASED_NEIGHBORHOOD_COINS_EVENT)

        if new_features[3] < old_features[3]:
            events.append(DECREASED_NEIGHBORHOOD_COINS_EVENT)

        if new_features[4] > old_features[4]:
            events.append(INCREASED_REACHABLE_COINS_EVENT)

        # if np.all(np.equal(new_features[:3], old_features[:3])):
        #     events.append(NO_PROGRESS_EVENT)

        if performed_step(crate_step, self_action):
            events.append(GOOD_CRATE_STEP_EVENT)
        if (not (performed_step(crate_step, self_action)) and self_action != 'BOMB' and self_action != 'WAIT'):
            events.append(BAD_STEP_EVENT)

        if performed_step(offensive_step, self_action):
            events.append(GOOD_OFFENSIV_STEP_EVENT)
        if (not (performed_step(offensive_step, self_action)) and self_action != 'BOMB' and self_action != 'WAIT'):
            events.append(BAD_STEP_EVENT)

        if performed_step(defensive_step, self_action):
            events.append(GOOD_DEFENSIV_STEP_EVENT)
        if (not (performed_step(defensive_step, self_action)) and self_action != 'BOMB' and self_action != 'WAIT'):
            events.append(BAD_STEP_EVENT)

        if (np.all(coin_step == 0) and np.all(crate_step == 0) and np.all(safe_step == 0) and np.all(offensive_step == 0) and np.all(defensive_step == 0) and self_action == 'WAIT' and bomb_available == False):
            events.append(GOOD_WAIT_EVENT)

        if (not (np.all(coin_step == 0) and np.all(crate_step == 0) and np.all(safe_step == 0) and np.all(offensive_step == 0) and np.all(defensive_step == 0)) and self_action == 'WAIT'):
            events.append(BAD_WAIT_EVENT)

        if (np.all(crate_step == 0) and self_action == 'BOMB' and bomb_available == True): #and is_escapable == True
            events.append(GOOD_BOMB_DROP_EVENT)

        if (np.all(offensive_step == 0) and self_action == 'BOMB' and bomb_available == True and is_escapable == True):
            events.append(GOOD_BOMB_DROP_EVENT)

        if (is_escapable == True and self_action == 'BOMB' and bomb_available == True):
            events.append(OKAY_BOMB_DROP_EVENT)

        if (not ((np.all(crate_step == 0) or np.all(offensive_step == 0)) and is_escapable == True and bomb_available == True) and self_action == 'BOMB'):
            events.append(BAD_BOMB_DROP_EVENT)

        if(is_escapable == False and self_action == 'BOMB'):
            events.append(I_LIKE_TO_DIE_EVENT)


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
    weights[action_number] = weights[action_number] + learning_rate * state_to_features(old_game_state) * ((reward + discount_factor * q_function(new_game_state, greedy_action, weights)) - q_function(old_game_state, self_action, weights))

    #update the model
    self.model = weights

    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1

    if(learning_rate >= self.min_learning_rate):
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


    #learning part -------------------------------------------------------------------------------------------------------------------------

    #hyperparameters
    discount_factor = 0.6
    learning_rate = self.learning_rate

    #train the model
    weights = self.model

    reward = reward_from_events(self, events)
    last_game_state_copy = last_game_state.copy()
    new_game_state = simulate_step(last_game_state_copy, last_game_state_copy['self'][3], last_game_state_copy['self'][2], last_game_state_copy['field'], last_action)


    greedy_action = choose_greedy_action(new_game_state, weights)
    action_number = ACTIONS_TO_NUMBERS[last_action]

    #SARSA
    weights[action_number] = weights[action_number] + learning_rate * state_to_features(last_game_state) * ((reward + discount_factor * q_function(last_game_state, greedy_action, weights)) - q_function(last_game_state, last_action, weights))

    #update the model
    self.model = weights

    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    self.statistics.append([last_game_state['round'], last_game_state['step'], self.collected_coins])
    #print(self.statistics)
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
        e.BOMB_DROPPED: 2,
        e.KILLED_SELF: -15,
        e.GOT_KILLED: -15,
        e.KILLED_OPPONENT: 10,
        e.CRATE_DESTROYED: 0.2,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad
        CLOSER_TO_COIN_EVENT: 1,
        REMOVED_FROM_COIN_EVENT: -0.5,
        INCREASED_NEIGHBORHOOD_COINS_EVENT: 3,
        DECREASED_NEIGHBORHOOD_COINS_EVENT: -0.1,
        INCREASED_REACHABLE_COINS_EVENT: 0.5,
        NO_PROGRESS_EVENT: -3,
        SAFE_STEP_EVENT: 3,
        I_LIKE_TO_DIE_EVENT: -3,
        GOOD_CRATE_STEP_EVENT: 1,
        GOOD_OFFENSIV_STEP_EVENT: 0.5,
        GOOD_DEFENSIV_STEP_EVENT: 0.5,
        BAD_STEP_EVENT: -0.1,
        GOOD_WAIT_EVENT: -1,
        BAD_WAIT_EVENT: -5,
        GOOD_BOMB_DROP_EVENT: 3,
        OKAY_BOMB_DROP_EVENT: 1,
        BAD_BOMB_DROP_EVENT: -0.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def performed_step(step, action):
    step = [step[0], step[1]]
    if((action == 'UP' and step == [0, -1]) or (action == 'DOWN' and step == [0, 1]) or (action == 'LEFT' and step == [-1, 0]) or (action == 'RIGHT' and step == [1, 0]) or (action == 'WAIT' and step == [0, 0])):
        return True
    else:
        return False
