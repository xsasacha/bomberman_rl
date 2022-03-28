from collections import namedtuple, deque

import pickle
#install before tournament??
import dill as pickle
import os
from typing import List

import events as e
from .callbacks import state_to_features, EPS_DECAY, EPS_MIN

import numpy as np

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPS = 0.5
EPS_DECAY = 0.99
EPS_MIN = 0.05

ROUNDS = 10
FNAME_DATA = "model-data.pt"

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

#-------------------------------
# Events
#-------------------------------

CLOSER_TO_COIN_EVENT = "CLOSER_TO_COIN"
REMOVED_FROM_COIN_EVENT = "REMOVED_FROM_COIN"

CLOSER_TO_CRATES_EVENT = "CLOSER_TO_CRATES"
REMOVED_FROM_CRATES_EVENT = "REMOVED_FROM_CRATES"

DEFENSIVE_STEP_EVENT = "DEFENSIVE_STEP"
IN_BLAST_RANGE_EVENT = "IN_BLAST_RANGE"

CLOSER_TO_ENEMY_EVENT = "CLOSER_TO_ENEMY"
REMOVED_FROM_ENEMY_EVENT = "REMOVED_FROM_ENEMY"

NO_PROGRESS_EVENT = "NO_PROGRESS"
STEP_POSSIBLE_EVENT = "STEP_POSSIBLE"




def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    if os.path.isfile(FNAME_DATA):
        with open(FNAME_DATA, "rb") as file:
            self.historic_data = pickle.load(file)
        self.game_nr = max(self.historic_data['games']) + 1

    else:
        self.historic_data = {
            'score'  : [],
            'coins'  : [],
            'games' :  []
                }
        self.game_nr = 1

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.actions = ACTIONS
    self.eps = EPS
    self.score_in_round    = 0
    self.collected_coins   = 0


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

    #---------------------------------------------------
    # Check if distance to coin increased, decreased or remained the same. (Feature 1 & 2)
    #---------------------------------------------------

    if old_game_state:
        old_features = state_to_features(old_game_state)
        coin_dir = np.asarray([old_features[0], old_features[1]])
        crate_dir = np.asarray([old_features[6], old_features[7]])
        safety_dir = np.asarray([old_features[8], old_features[9]])
        #offense_dir = np.asarray([old_features[11], old_features[12]])
        old_player_pos = old_game_state['self'][3]
        new_player_pos = new_game_state['self'][3]

        if check_dir(coin_dir, self_action):
            events.append(CLOSER_TO_COIN_EVENT)
        elif old_player_pos == new_player_pos:
            events.append(NO_PROGRESS_EVENT)
        else:
            events.append(REMOVED_FROM_COIN_EVENT)

     #---------------------------------------------------
     # Check if taken step was a possible one. (Feature 3 - 6)
     #---------------------------------------------------

        if check_action(self_action, old_features):
            events.append(STEP_POSSIBLE_EVENT)

     #---------------------------------------------------
     # Check if distance to crate increased, decreased or remained the same. (Feature 7 & 8)
     #---------------------------------------------------

      #  if check_dir(crate_dir, self_action):
      #      events.append(CLOSER_TO_CRATES_EVENT)
      #  elif not(old_player_pos == new_player_pos):
      #      events.append(REMOVED_FROM_CRATES_EVENT)

     #---------------------------------------------------
     # Check if a defensive action has been taken (Feature 9 - 11)
     #---------------------------------------------------

      #  if (old_features[10] == 1):
      #      if check_dir(safety_dir, self_action):
      #          events.append(DEFENSIVE_STEP_EVENT)
      #      else:
      #          events.append(IN_BLAST_RANGE_EVENT)

     #----------------------------------------------------
     # Check if we moved towards an enemy. (Feature 12 & 13)
     #----------------------------------------------------

     #   if check_dir(offense_dir, self_action):
     #       events.append(CLOSER_TO_ENEMY_EVENT)
     #   elif not (old_player_pos == new_player_pos):
     #       events.append(REMOVED_FROM_ENEMY_EVENT)



    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


    #learning part -------------------------------------------------------------------------------------------------------------------------

    #hyperparameters
    discount_factor = 0.8
    lr_rate = 0.01

    #train the model
    #print(events)
    reward = reward_from_events(self, events)
    #print(reward)

    last_action = ACTIONS.index(self_action)
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    next_action = choose_greedy_action(new_features, self.Q)

    self.Q[old_features][last_action] = self.Q[old_features][last_action] + lr_rate * ((reward + discount_factor * self.Q[new_features][next_action]) - self.Q[old_features][last_action])

    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1

    self.score_in_round += reward_from_events(self, events)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')


def choose_greedy_action(features, Q):
    greedy_action = np.argmax(Q[features])
    return greedy_action

def check_dir(dir, self_action):
    if (np.all(dir == (0, 1)) and self_action == 'DOWN') or (np.all(dir == (0, -1)) and self_action == 'UP') or (np.all(dir == (1, 0)) and self_action == 'RIGHT') or (np.all(dir == (-1, 0)) and self_action == 'LEFT'):
        return True
    else:
        return False

def check_action(self_action, old_features):
    if (self_action == 'UP' and old_features[2] == True) or (self_action == 'DOWN' and old_features[3] == True) or (self_action == 'LEFT' and old_features[4] == True) or (self_action == 'RIGHT' and old_features[5] == True):
        return True
    else:
        return False


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

    if self.eps > EPS_MIN:
        self.eps *= EPS_DECAY


    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    self.score_in_round += reward_from_events(self, events)

    score = np.sum(self.score_in_round)

    self.historic_data['score'].append(score)
    self.historic_data['coins'].append(self.collected_coins)
    self.historic_data['games'].append(self.game_nr)

    self.score_in_round   = 0
    self.collected_coins  = 0
    self.game_nr += 1

    if self.game_nr % ROUNDS == 0:
        self.logger.info(self.historic_data)
        print(self.historic_data)

    with open(FNAME_DATA, "wb") as file:
        pickle.dump(self.historic_data, file)


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 3,
        e.BOMB_DROPPED: 0,
        e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 5,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -10,

        CLOSER_TO_COIN_EVENT: 2,
        REMOVED_FROM_COIN_EVENT: -2,

        CLOSER_TO_CRATES_EVENT: 1,
        REMOVED_FROM_CRATES_EVENT: -1,

        CLOSER_TO_ENEMY_EVENT: 0,
        REMOVED_FROM_ENEMY_EVENT: 0,

        DEFENSIVE_STEP_EVENT: 3,

        NO_PROGRESS_EVENT: -1,
        STEP_POSSIBLE_EVENT: 0.1
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
