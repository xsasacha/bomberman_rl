import os
import pickle
import dill as pickle
import random

import numpy as np
from queue import Queue
from collections import defaultdict


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
EPS = 0.5
EPS_DECAY = 0.99
EPS_MIN = 0.05


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
    self.actions = ACTIONS
    self.eps = EPS

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        #self.model = np.zeros(3,(len(ACTIONS))
        length = len(ACTIONS)
        self.Q = defaultdict(lambda: np.zeros(length))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train:
        eps = self.eps
    else:
        eps = EPS_MIN
    if random.random() < eps:
        #print('Random action.')
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)
    else:
        self.logger.debug("Querying model for action.")
        feature_lookup = self.Q[(state_to_features(game_state))]
        action_nr = np.argmax(feature_lookup)
        return ACTIONS[action_nr]


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
    best_path_crates = []                          #for paths through the map, needed for feature 4&5 (alternativ)
    active_bombs = get_bombs(game_state)    # list with coordinates of all bombs


    #--------------------------------------------------------------------------------------------------------------
    # Compute distances and best_path
    #--------------------------------------------------------------------------------------------------------------
    nearest_coin_distance = 100
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

    # Invert the distance, so that a higher value means the distance is "better"
    inverted_coin_distance = 0
    # if the agent is on a coin don't divide by zero, give a great value (better than 1/1, the next best value if the distance is 1)
    if (nearest_coin_distance == 0):
        inverted_coin_distance = 2
    else:
        inverted_coin_distance = 1/nearest_coin_distance


    #--------------------------------------
    #Feature 1 & 2: Direction of the nearest coin (float values)
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
    #Feature 1 & 2 alternative: Direction of the nearest coin (first direction of shortes path to a coin)
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
    #Feature 3 - 6: Check if movement in the direction is possible (order: UP, DOWN, LEFT, RIGHT)
    #--------------------------------------
    features.append(arena[player[0]][player[1] - 1] == 0)
    features.append(arena[player[0]][player[1] + 1] == 0)
    features.append(arena[player[0] - 1][player[1]] == 0)
    features.append(arena[player[0] + 1][player[1]] == 0)

    # End of coin features -----------------------------------------------------------------------------------------------------------------------------

    nearest_crate_distance = 100
    distances_crate = []
    crate_coordinates = np.array([np.where(arena == 1)[0], np.where(arena == 1)[1]]).T
    crate_list = crate_coordinates.tolist()
    #if there are no visible coins, assume the agent is at the top of a coin so it does not search for any coin
    if (crate_list == None) or (crate_list == []):
        nearest_crate_distance = 0
    else:
        for crate in crate_list:
            for coordinate in np.array([[0,1],[0,-1],[-1,0],[1,0]]):
                next_to_crate = np.asarray(crate) + coordinate
                crate_distance, current_path = find_distance(np.asarray(player), np.asarray(next_to_crate), arena)
                if crate_distance < nearest_crate_distance:
                    nearest_crate_distance = crate_distance
                    best_path_crates = current_path

    #------------------------------------
    #Feature 7 & 8: Direction to position with max number of destructable crates
    #--------------------------------------

    if(float_values_flag == 0):
        if ((crate_coordinates is not None) and (crate_coordinates != []) and (best_path_crates != [])):
            direction = np.asarray(best_path_crates[0]) - np.asarray(player)
            features.append(direction[0])
            features.append(direction[1])
        else:
            # If no crate is visible, give no direction
            features.append(0)
            features.append(0)


    #--------------------------------------
    #Feature 9 & 10: Direction to next safe field
    #--------------------------------------
    # list with all dangerous fields
    list_dangerous_fields = dangerous_fields(game_state)
    # copy of arena with dangerous fields set to 2
    copy_arena = np.copy(arena)
    for l in list_dangerous_fields:
        copy_arena[l[0], l[1]] = 2

    free_field = find_fields_of_type(player, copy_arena, 0)
    free_field_distance, path = find_distance(np.asarray(player), np.asarray(free_field), arena)

    if (path != [] and path != False):
        direction = np.asarray(path[0]) - np.asarray(player)
        features.append(direction[0])
        features.append(direction[1])
    else:
        # If no free field is visible, give no direction
        features.append(0)
        features.append(0)

    #--------------------------------------
    #Feature 11: Severe danger
    #--------------------------------------
    # danger = 1 if player stands on field within bomb blast
    # danger = 0 if player stands on safe field

    danger = severe_danger(game_state)
    features.append(danger)


    # # list with all possible field coordinates
    # field_coordinates = np.array([np.where(arena == 0)[0], np.where(arena == 0)[1]]).T
    # # list with safe possible fields
    # safe_fields_coordinates = []
    # for f in field_coordinates:
    #     if(f not in dangerous_fields):
    #         safe_fields_coordinates.append(f)

    '''Feature Idee: Integer die angibt "wenn ich jetzt eine Bombe lege, wie weit ist das nächste sichere Feld entfernt (seeehr hoch falls es keins mehr gibt)"?'''


    #------------------------------------------------------------------------------------------
    #Feature 12 & 13: Direction to next offensive tile (that we can escape) in a certain range
    #------------------------------------------------------------------------------------------

    offensive_distance, step = find_best_offensive_step(game_state)

    if(offensive_distance != False):
        features.append(step[0])
        features.append(step[1])
    else:
        features.append(0)
        features.append(0)

    #----------------------------------------------------------------------------------------
    #Feature 14: Time to bomb crate?
    #----------------------------------------------------------------------------------------

    if len(best_path_crates) == 0:
        features.append(1)
    else: features.append(0)


    #-----------------------------------------------------------------------------------------
    #Feature 14&15: Direction with max distance to other players
    #-----------------------------------------------------------------------------------------

    #defensive_step = find_best_defensive_step(game_state)

    #if(defensive_step != False):
    #    features.append(defensive_step[0])
    #    features.append(defensive_step[1])
    #else:
    #    features.append(0)
    #    features.append(0)


    for feature in features:
        channels.append(feature)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return tuple(stacked_channels.reshape(-1))

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
                    parent[(new_new_position[0], new_new_position[1])] = new_position
                    # block the new new coordinates for other paths in this search
                    arena_copy[new_new_position[0], new_new_position[1]] = 10

    # Only reach this point if no path to the goal exists
    return 1000, []



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


def find_fields_of_type(start_coordinates, arena, field_type = 0):
    # We work with a copy of the arena so we can change values
    arena_copy = np.copy(arena)

    #Block the start coordinates for paths
    arena_copy[start_coordinates[0], start_coordinates[1]] = -1

    # Use a BFS to find the nearest free safe field beginning from the starting position
    parent = {}                                  # Save parents of coordinates in paths in a dict
    q = Queue()
    q.put(start_coordinates)
    path_length = 0

    parent[(start_coordinates[0], start_coordinates[1])] = None

    while not q.empty():
        # take first element
        current_element = q.get()
        # # check if the path is complete (termination requirement)
        # if(arena_copy[current_element[0], current_element[1]] == 0):
        #     # set goal coordinates
        #     goal_coordinates = current_element
        #     return goal_coordinates

        # do all possible steps for a position (UP, DOWN, LEFT, RIGHT) -> since the arena is surrounded by walls we don't need to check if a step would exceed the boundaries of the arena error, every path stops before
        # TODO: Maybe vectorize this step
        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for step in steps:
            new_position = current_element + np.asarray(step)
            if arena_copy[new_position[0], new_position[1]] == field_type:
                goal_coordinates = new_position
                return goal_coordinates

            elif arena_copy[new_position[0], new_position[1]] != -1:
                q.put(new_position)

                #save the parent of the new position
                parent[(new_position[0], new_position[1])] = current_element
                # block the new coordinates for other paths in this search
                arena_copy[new_position[0], new_position[1]] = -1
                ''' #this is more efficient than the simple BFS, but we need to comment out "q.put(new_position)" (6 lines above). MORE IMPORTANT: Before we do this we must somehow be sure to be at a crossroad!!!
                #go one step further in this direction and to the same step again, because we can to "2-steps" from each crossroad to the next one
                new_new_position = new_position + np.array(step)

                if arena_copy[new_new_position[0], new_new_position[1]] == 0:
                    q.put(new_new_position)

                    #save the parent of the new new position
                    parent[(new_new_position[0], new_new_position[1])] = current_element
                    # block the new new coordinates for other paths in this search
                    arena_copy[new_new_position[0], new_new_position[1]] = 10
                '''
    # Only reach this point if no path to the goal exists
    return start_coordinates



'''Maybe we have to check if the new coordinates exceed the range of the arena. Also the explosion stops at stone walls, this computation just skips them (but it is still an approximation).'''
def affected_range(bomb_coordinates):
    # list with all coordinates of every field affected by given bomb
    affected_range = []
    # temporary list with all coordinates of every field affected by given bomb
    affected_range_temp=[[bomb_coordinates[0],bomb_coordinates[1]]]

    # calculate coordinates with a radius of 3 from bomb coordinates
    for i in range(1,4):
        affected_range_temp.append([bomb_coordinates[0]+i,bomb_coordinates[1]])
        affected_range_temp.append([bomb_coordinates[0]-i,bomb_coordinates[1]])
        affected_range_temp.append([bomb_coordinates[0],bomb_coordinates[1]+i])
        affected_range_temp.append([bomb_coordinates[0],bomb_coordinates[1]-i])

    # check if coordinates are within arena
    for a in affected_range_temp:
        if(0 < a[0] < 16 and 0 < a[1] < 16):
            affected_range.append(a)
    # return list with coordinates of affected fields
    return affected_range

def dangerous_fields(game_state):
    dangerous_fields = [] # list with coordinates of all potentially affected fields
    bombs = get_bombs(game_state)   # list of coordinates of all bombs
    for b in bombs:
        affected_fields_bomb = affected_range(b)
        for x in affected_fields_bomb:
            dangerous_fields.append(x)
    return dangerous_fields

def get_free_neighbours(coordinates, game_state):
    arena = game_state['field']

    # list of free neighbours
    free_neighbours = []
    # list of directions in form of (x, y) coordinates
    directions = [(coordinates[0], coordinates[1] + 1), (coordinates[0], coordinates[1] - 1), (coordinates[0] + 1, coordinates[1]), (coordinates[0] - 1, coordinates[1])]
    for x, y in directions:
        # check if field is free
        if(arena[x, y] == 0):
            free_neighbours.append([x, y])

    return free_neighbours

def get_bombs(game_state):
    # list of coordinates of all currently active bombs including timer
    bombs = game_state['bombs']

    if(bombs != []):
        # list of only coordinates of active bombs
        bomb_coordinates_list = []
        for b in bombs:
            bomb_coordinates_list.append(list(b[0]))
        return bomb_coordinates_list
    else:
        return bombs

def get_others(game_state):
    # list of other players
    others = game_state['others']
    if(others != []):
        # list of only coordinates of the other agents
        others_coordinates_list = []
        for o in others:
            others_coordinates_list.append(list(o[3]))
        return others_coordinates_list
    else:
        return others

def severe_danger(game_state):
    # returns true if player is standing on field within bomb blast, otherwise false
    # list with all dangerous fields
    list_dangerous_fields = dangerous_fields(game_state)
     # coordinates of the player
    player = game_state['self'][3]
    for f in list_dangerous_fields:
        if player == f:
            return True
        else:
            return False
    return False

def is_escapable(game_state, position):
     # coordinates of the player
    player = position

    # the arena
    arena = game_state['field']
    # list with all dangerous fields
    list_dangerous_fields = dangerous_fields(game_state)
    # copy of arena with dangerous fields set to 2
    copy_arena = np.copy(arena)
    for l in list_dangerous_fields:     # ist das so richtig rum?
        copy_arena[l[0], l[1]] = 2

    free_field = find_fields_of_type(player, copy_arena, 0)

    free_field_distance, path = find_distance(np.asarray(player), np.asarray(free_field), arena)

    if free_field_distance < 4:
        return True
    else:
        return False

def crates_player_range(game_state, position):
     # coordinates of the player
    player = position
    # the arena
    arena = game_state['field']
    # bombs
    active_bombs = get_bombs(game_state)

    # crates have entry = 1 within the arena; crate_coordinates = array with coordiantes of crates
    crate_coordinates = np.array([np.where(arena == 1)[0], np.where(arena == 1)[1]]).T
    crate_list = crate_coordinates.tolist()
    # count of crates which are reachable by player
    reachable_crates = 0
    #create list with all affected fields when player places bomb
    affected_fields = affected_range(player)

    # # count how many crates are in affected range   ##### OLD VERSION
    for field in affected_fields:
     #    print("field: " + str(field))
         if field in crate_list:
             if player in active_bombs:
                 reachable_crates += 1

    # ''' #Alternativ: Vllt effizienter da if(field in crate list) jedes mal die ganze crate list(bis 100 einträge) überprüft -> O(len(affected_fields) * len(crate_list))
    for field in affected_fields:
        if arena[field[0],field[1]] == 1:
            reachable_crates += 1

    if reachable_crates != 0:
        # divide reachable crates with total count of crates
        crates = reachable_crates / 13
        return crates
    else:
        return 0

def maximum_crates(game_state, max_range):
    #find all reachable fields for the player with a BFS
    reachable_positions = []
    game_state_arena = game_state['field']
    arena_copy = np.copy(game_state_arena)
    player = game_state['self'][3]

    #Block the start coordinates for paths
    arena_copy[player[0], player[1]] = 10

    # Initialize the BFS
    parent = {}                                  # Save parents of coordinates in paths in a dict
    q = Queue()
    q.put(player)
    path_length = 0

    parent[(player[0], player[1])] = None
    reachable_positions.append(np.asarray(player))

    #make sure we start at a crossroad
    #check if we can move up or down and check if we can move right or left
    if(((arena_copy[player[0]][player[1] - 1] == 0) or (arena_copy[player[0]][player[1] + 1] == 0)) or ((arena_copy[player[0] - 1][player[1]] == 0) or (arena_copy[player[0] + 1][player[1]] == 0))):
        #if we are in this if condition, we are at a corssroad and have nothing to do
        pass
    else:
        #we are between 2 crossroads, this means we have to perform one step of the BFS so we can compute it more efficient afterwards
        # take first element
        current_element = q.get()

        # do all possible steps for the start position (UP, DOWN, LEFT, RIGHT) -> since the arena is surrounded by walls we don't need to check if a step would exceed the boundaries of the arena error, every path stops before
        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for step in steps:
            new_position = current_element + np.asarray(step)
            if arena_copy[new_position[0], new_position[1]] == 0:
                q.put(new_position)
                reachable_positions.append(new_position)

                #save the parent of the new position
                parent[(new_position[0], new_position[1])] = current_element
                # block the new coordinates for other paths in this search
                arena_copy[new_position[0], new_position[1]] = 10

    #the BFS-loop
    while not q.empty():
        # take first element of the queue
        current_element = q.get()

        # do all possible steps for a position (UP, DOWN, LEFT, RIGHT) -> since the arena is surrounded by walls we don't need to check if a step would exceed the boundaries of the arena error, every path stops before
        # TODO: Maybe vectorize this step
        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for step in steps:
            new_position = current_element + np.asarray(step)
            if arena_copy[new_position[0], new_position[1]] == 0:
                reachable_positions.append(new_position)
                #save the parent of the new position
                parent[(new_position[0], new_position[1])] = current_element
                # block the new coordinates for other paths in this search
                arena_copy[new_position[0], new_position[1]] = 10


                #this is more efficient than the simple BFS, but we need to comment out "q.put(new_position)" (6 lines above). MORE IMPORTANT: Before we do this we must somehow be sure to be at a crossroad!!!
                #go one step further in this direction and to the same step again, because we can to "2-steps" from each crossroad to the next one
                new_new_position = new_position + np.array(step)

                if arena_copy[new_new_position[0], new_new_position[1]] == 0:
                    q.put(new_new_position)
                    reachable_positions.append(new_new_position)

                    #save the parent of the new new position
                    parent[(new_new_position[0], new_new_position[1])] = current_element
                    # block the new new coordinates for other paths in this search
                    arena_copy[new_new_position[0], new_new_position[1]] = 10


    #find best booooom position in the reachable positions in a given range
    max_crates = 0
    max_direction = [0, 0]

    for position in reachable_positions:
        path = construct_path(parent, position)
        if (len(path) < max_range):
            #simulate bomb drop
            arena_copy_copy = game_state_arena.copy()
            game_state_copy = game_state.copy()

            #new_state = simulate_step(game_state_copy, position, game_state_copy['self'][2], arena_copy_copy, 'BOMB')
            new_state = game_state_copy
            new_state['bombs'] = [(position, 4)]

            crates_current_position = crates_player_range(new_state, position)

            if(crates_current_position > max_crates and is_escapable(new_state, position)):
                max_crates = crates_current_position

                if len(path) > 0:
                    max_direction = path[0] - player
                else:
                    max_direction = [0, 0]

        else:
            #in this case every future path will be longer than the max range
            break

    #wir können theoretisch auch mit find_distance die Distanz bestimmen falls wir diese lieber als feature wollen!

    return max_crates, max_direction

def find_best_offensive_step(game_state):
    player = game_state['self'][3]
    others = get_others(game_state)
    arena = game_state['field']

    if(others != []):
        #assume the others are bombs -> The explosion area is also the exact same area in which they can get hit by bombs if a bomb would explode there
        offensive_tiles = []
        for other in others:
            new_offensive_tiles = affected_range(other)
            for new_tile in new_offensive_tiles:
                offensive_tiles.append(new_tile)
        offensive_tiles = np.reshape(np.array(offensive_tiles),(-1, 2))
        offensive_tiles = offensive_tiles.tolist()

        #mark the offensive_tiles in the arena
        arena_copy = arena.copy()

        for tile in offensive_tiles:
            arena_copy[tile[0]][tile[1]] = 5

        #block severe_danger tiles in the arena (so we won't choose a path over them)
        list_dangerous_fields = dangerous_fields(game_state)
        # copy of arena with dangerous fields set to 2
        for l in list_dangerous_fields:     # ist das so richtig rum?
            arena_copy[l[0], l[1]] = 2

        #find the nearest tile to the player with value 10 (that is reachable)
        next_offensive_tile = find_fields_of_type(player, arena_copy, field_type=5)

        distance, path = find_distance(player, next_offensive_tile, arena_copy)

        if(path != []):
            return distance, [path[0][0], path[0][1]]
        else:
            return False, False

    else:
        return False, False

def find_best_defensive_step(game_state):
    player = game_state['self'][3]
    others = get_others(game_state)
    arena = game_state['field']

    if(others != []):
        #Visit all neighbor tiles of the player
        neighbours = get_free_neighbours(player, game_state)

        min_distances = []

        for neighbour in neighbours:
            min_distance = 1000
            for other in others:
                distance, path = find_distance(neighbour, other, arena)
                if distance < min_distance:
                    min_distance = distance
            min_distances.append(min_distance)


        direction = []
        if(max(min_distances) != 1000):
            direction = neighbours[argmax(min_distances)] - player
        else:
            direction = [0, 0]

        return direction

    else:
        return False

