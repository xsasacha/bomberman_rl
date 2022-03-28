import os
import pickle
import random

import numpy as np
from queue import Queue


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_TO_NUMBERS = {
    'UP':  0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4,
    'BOMB': 5
}


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model = np.full((len(ACTIONS), 3), 0.01)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

        #self.model = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, 0], [0, 0, 0]])
        #self.model = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])

#TODO: MAybe add a new_game_state argument that is default None, if it is given don't simulate the action, just pick the new game state
def q_function(state, action, weights):
    state_copy = state.copy()
    player_pos = state['self'][3]
    player_bomb = state['self'][2]
    arena = state['field']
    action_number = None

    # Test if the given action is possible for the given state
    for i in range(0, 6):
        if(action == ACTIONS[i]):
            action_number = i
            state_copy = simulate_step(state_copy, player_pos, player_bomb, arena, action)

    if(action_number is None):
        print('Invalid action')

    features = state_to_features(state_copy)

    #matrix multiplication
    assert(len(features) == len(weights[action_number]))
    q_value = np.dot(np.asarray(features), np.asarray(weights[action_number]))

    return q_value


def simulate_step(state, player_pos, player_bomb, arena, action):
    if action == 'UP':
        if arena[player_pos[0]][player_pos[1] - 1] == 0:
            state['self'] = (state['self'][0], state['self'][1], state['self'][2], (player_pos[0], player_pos[1] - 1))
    if action == 'DOWN':
        if arena[player_pos[0]][player_pos[1] + 1] == 0:
            state['self'] = (state['self'][0], state['self'][1], state['self'][2], (player_pos[0], player_pos[1] + 1))
    if action == 'LEFT':
        if arena[player_pos[0] - 1][player_pos[1]] == 0:
            state['self'] = (state['self'][0], state['self'][1], state['self'][2], (player_pos[0] - 1, player_pos[1]))
    if action == 'RIGHT':
        if arena[player_pos[0] + 1][player_pos[1]] == 0:
            state['self'] = (state['self'][0], state['self'][1], state['self'][2], (player_pos[0] + 1, player_pos[1]))
    if action == 'BOMB':
        if state['self'][2] == True:
            state['bombs'].append((player_pos, 4))
            state['self'] = (state['self'][0], state['self'][1], False, player_pos)

    return state

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    #random_prob = .1
    #if self.train and random.random() < random_prob:
    #    self.logger.debug("Choosing action purely at random.")
    #    # 80%: walk in any direction. 10% wait. 10% bomb.
    #    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
#
    #self.logger.debug("Querying model for action.")
    #return np.random.choice(ACTIONS, p=self.model)

    random_prob = .3
    if self.train and random.random() < random_prob:
        #print('Random')
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    #compute the q-values for all possible actions
    q_values = [q_function(game_state, action, self.model) for action in ACTIONS]
    '''print(q_values)
    print(state_to_features(game_state))
    print(np.argmax(q_values))
    print(ACTIONS[np.argmax(q_values)])
    print("-------------------------")
    '''
    if(np.all(q_values[0] == q_values)):
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    #select the best action
    return ACTIONS[np.argmax(q_values)]


#TODO; Maybe also consider coordinates with other players as blocked
def find_distance_original(start_coordinates, goal_coordinates, arena):
    # We work with a copy of the arena so we can change values
    arena_copy = np.copy(arena)

    #Block the start coordinates for paths
    arena_copy[start_coordinates[0], start_coordinates[1]] = 10

    # Use a BFS to find the shortest path towards the goal position, beginning from the starting position
    parent = {}                                  # Save parents of coordinates in paths in a dict
    q = Queue()
    q.put(start_coordinates)
    path_length = 0

    parent[(start_coordinates[0], start_coordinates[1])] = None

    while not q.empty():
        # take first element
        current_element = q.get()

        # check if the path is complete (termination requirement)
        if(current_element[0] == goal_coordinates[0] and current_element[1] == goal_coordinates[1]):
            #return path_length, construct_path(parent, goal_coordinates)
            path = construct_path(parent, goal_coordinates)
            return len(path), construct_path(parent, goal_coordinates)

        # do all possible steps for a position (UP, DOWN, LEFT, RIGHT) -> since the arena is surrounded by walls we don't need to check if a step would exceed the boundaries of the arena error, every path stops before
        # TODO: Maybe vectorize this step
        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for step in steps:
            new_position = current_element + np.asarray(step)
            if arena_copy[new_position[0], new_position[1]] == 0:
                q.put(new_position)

                #save the parent of the new position
                parent[(new_position[0], new_position[1])] = current_element
                # block the new coordinates for other paths in this search
                arena_copy[new_position[0], new_position[1]] = 10

    # Only reach this point if no path to the goal exists
    return 1000

#TODO; Maybe also consider coordinates with other players as blocked
def find_distance(start_coordinates, goal_coordinates, arena):
    # We work with a copy of the arena so we can change values
    arena_copy = np.copy(arena)

    #Block the start coordinates for paths
    arena_copy[start_coordinates[0], start_coordinates[1]] = 10

    # Use a BFS to find the shortest path towards the goal position, beginning from the starting position
    parent = {}                                  # Save parents of coordinates in paths in a dict
    q = Queue()
    q.put(start_coordinates)
    path_length = 0

    parent[(start_coordinates[0], start_coordinates[1])] = None


    #make sure we start at a crossroad
    #check if we can move up or down and check if we can move right or left
    if(((arena[start_coordinates[0]][start_coordinates[1] - 1] == 0) or (arena[start_coordinates[0]][start_coordinates[1] + 1] == 0)) and ((arena[start_coordinates[0] - 1][start_coordinates[1]] == 0) or (arena[start_coordinates[0] + 1][start_coordinates[1]] == 0))):
        #if we are in this if condition, we are at a corssroad and have nothing to do
        pass
    else:
        #we are between 2 crossroads, this means we have to perform one step of the BFS so we can compute it more efficient afterwards
        # take first element
        current_element = q.get()

        # check if the path is complete (termination requirement)
        if(current_element[0] == goal_coordinates[0] and current_element[1] == goal_coordinates[1]):
            #return path_length, construct_path(parent, goal_coordinates)
            path = construct_path(parent, goal_coordinates)
            return len(path), construct_path(parent, goal_coordinates)

        # do all possible steps for the start position (UP, DOWN, LEFT, RIGHT) -> since the arena is surrounded by walls we don't need to check if a step would exceed the boundaries of the arena error, every path stops before
        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for step in steps:
            new_position = current_element + np.asarray(step)
            if arena_copy[new_position[0], new_position[1]] == 0:
                q.put(new_position)

                #save the parent of the new position
                parent[(new_position[0], new_position[1])] = current_element
                # block the new coordinates for other paths in this search
                arena_copy[new_position[0], new_position[1]] = 10



    while not q.empty():
        # take first element
        current_element = q.get()

        # check if the path is complete (termination requirement)
        if(current_element[0] == goal_coordinates[0] and current_element[1] == goal_coordinates[1]):
            #return path_length, construct_path(parent, goal_coordinates)
            path = construct_path(parent, goal_coordinates)
            return len(path), construct_path(parent, goal_coordinates)

        # do all possible steps for a position (UP, DOWN, LEFT, RIGHT) -> since the arena is surrounded by walls we don't need to check if a step would exceed the boundaries of the arena error, every path stops before
        # TODO: Maybe vectorize this step
        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for step in steps:
            new_position = current_element + np.asarray(step)
            if arena_copy[new_position[0], new_position[1]] == 0:
                #q.put(new_position)

                #save the parent of the new position
                parent[(new_position[0], new_position[1])] = current_element
                # block the new coordinates for other paths in this search
                arena_copy[new_position[0], new_position[1]] = 10

                #check if the new position is the goal position
                if(new_position[0] == goal_coordinates[0] and new_position[1] == goal_coordinates[1]):
                    #return path_length, construct_path(parent, goal_coordinates)
                    path = construct_path(parent, goal_coordinates)
                    return len(path), construct_path(parent, goal_coordinates)


                #this is more efficient than the simple BFS, but we need to comment out "q.put(new_position)" (6 lines above). MORE IMPORTANT: Before we do this we must somehow be sure to be at a crossroad!!!
                #go one step further in this direction and to the same step again, because we can to "2-steps" from each crossroad to the next one
                new_new_position = new_position + np.array(step)

                if arena_copy[new_new_position[0], new_new_position[1]] == 0:
                    q.put(new_new_position)

                    #save the parent of the new new position
                    parent[(new_new_position[0], new_new_position[1])] = current_element
                    # block the new new coordinates for other paths in this search
                    arena_copy[new_new_position[0], new_new_position[1]] = 10

    # Only reach this point if no path to the goal exists
    return 1000

def construct_path(parent, goal_coordinates):
    path = []
    current_position = (goal_coordinates[0], goal_coordinates[1])

    while parent[current_position] is not None:
        path.append(parent[current_position])
        current_position = (parent[current_position][0], parent[current_position][1])

    path.reverse()

    #remove the starting position from the path, add the goal position
    path.append(goal_coordinates)
    del path[0]

    return path

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    features = []
    #channels.append(...)




    #--------------------------------------------------------------------------------------------------------------
    # Compute features for coin collecting: (Inverted) Distance to nearest coin, coins in a certain neighborhood, first step of path to nearest coin
    #--------------------------------------------------------------------------------------------------------------
    # Setup
    coins = game_state['coins']             # coordinates of all (visible) coins
    player = game_state['self'][3]          # coordinates of the player
    arena = game_state['field']             # the arena
    best_path = []                          #for paths through the map, needed for feature 4&5 (alternativ)

    #--------------------------------------
    #Feature 1: Distance to nearest coin
    #--------------------------------------
    nearest_coin_distance = 1000
    distances = []
    #if there are no visible coins, assume the agent is at the top of a coin so it does not search for any coin
    if (coins == None) or (coins == []):
        nearest_coin_distance = 0
    else:
        for coin in coins:
            coin_distance, current_path = find_distance(np.asarray(player), np.asarray(coin), arena)
            distances.append(coin_distance)
            if coin_distance < nearest_coin_distance:
                nearest_coin_distance = coin_distance
                best_path = current_path
        #print(best_path)

    # Invert the distance, so that a higher value means the distance is "better"
    inverted_coin_distance = 0
    # if the agent is on a coin don't divide by zero, give a great value (better than 1/1, the next best value if the distance is 1)
    if (nearest_coin_distance == 0):
        inverted_coin_distance = 2
    else:
        inverted_coin_distance = 1/nearest_coin_distance

    features.append(inverted_coin_distance)


    #--------------------------------------
    #Feature 2: Coins in a certain neighborhood (3-neighborhood)
    #--------------------------------------
    '''distances = np.array([])
    close_coins = 0

    for coin in coins:                                                             #IDEA: Use a loop and check if x and y values have a difference smaller than the searched neighborhood, so we do get a square neighborhood instead of a circular
        distance = np.linalg.norm(np.asarray(player) - np.asarray(coin))           #TODO: Maybe vectorize this # We use the linalg norm and not the path distance, because we also want to consider currently unreachable coins (we can bomb our way)
        distances = np.append(distances, distance)

        if(distance <= 3):
            close_coins += 1

    features.append(close_coins)


    #--------------------------------------
    #Feature 3: Coins in a certain neighborhood (6-neighborhood)
    #--------------------------------------
    reachable_coins = 0
    for coin in coins:
        distance = np.linalg.norm(np.asarray(player) - np.asarray(coin))           #TODO: Maybe vectorize this # We use the linalg norm and not the path distance, because we also want to consider currently unreachable coins (we can bomb our way)

        if(distance <= 6):
            reachable_coins += 1

    features.append(reachable_coins)

    '''
    #--------------------------------------
    #Feature 4 & 5: Direction of the nearest coin (float values)
    #--------------------------------------
    float_values_flag = 0

    if(float_values_flag == 1):
        if (coins is not None) and (coins != []):
            direction = np.asarray(coins[np.argmin(distances)]) - np.asarray(player)
            normalized_direction = np.linalg.norm(direction)
            features.append(direction[0] / normalized_direction)
            features.append(direction[1] / normalized_direction)
        else:
            # If no coin is visible, give no direction
            features.append(0)
            features.append(0)

    #--------------------------------------
    #Feature 4 & 5 alternative: Direction of the nearest coin (first direction of shortes path to a coin)
    #--------------------------------------
    if(float_values_flag == 0):
        if ((coins is not None) and (coins != []) and (best_path != [])):
            direction = np.asarray(best_path[0]) - np.asarray(player)
            features.append(direction[0])
            features.append(direction[1])
        else:
            # If no coin is visible, give no direction
            features.append(0)
            features.append(0)

    #--------------------------------------
    #Feature 6 - 9 check if movement in the direction is possible (order: UP, DOWN, LEFT, RIGHT)
    #--------------------------------------
    #features.append(arena[player[0]][player[1] - 1] == 0)
    #features.append(arena[player[0]][player[1] + 1] == 0)
    #features.append(arena[player[0] - 1][player[1]] == 0)
    #features.append(arena[player[0] + 1][player[1]] == 0)



    # End of coin features -----------------------------------------------------------------------------------------------------------------------------

    #print(features)

    for feature in features:
        channels.append(feature)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
