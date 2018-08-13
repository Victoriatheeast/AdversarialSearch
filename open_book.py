from isolation import Isolation
import random, pickle
from collections import defaultdict, Counter
from isolation.isolation import _WIDTH, _HEIGHT

NUM_ROUNDS = 20000

def build_table(num_rounds=NUM_ROUNDS):
    # You should run no more than `num_rounds` simulations -- the
    # goal of this quiz is to understand one possible way to develop
    # an opening book; not to develop a good one

    # NOTE: the GameState object is not hashable, and the python3
    #       runtime includes security features that make object
    #       hashes non-portable. There is a new attribute on
    #       GameState objects in this quiz called `hashable` that
    #       can be used as a dictionary key

    book = defaultdict(Counter)
    for _ in range(num_rounds):
        state = Isolation()
        build_tree(state, book)
    return {k: max(v, key=v.get) for k, v in book.items()}


def build_tree(state, book, depth=4):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    action = alpha_beta_pruning(state)
    wins = build_tree(state.result(action), book, depth - 1)
    book[state][action] += wins
    return -wins

def simulate(state):
    player_id = state.player()
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    return -1 if state.utility(player_id) < 0 else 1

def alpha_beta_pruning(state, depth=3):
    alpha = float("-inf")
    beta = float("inf")
    max_score = float("-inf")
    best_action = None
    for action in state.actions():
        v = min_value(state.result(action), alpha, beta, depth - 1)
        alpha = max(alpha, v)
        if v >= max_score:
            max_score = v
            best_action = action
    return best_action

def min_value(state, alpha, beta, depth):
    if state.terminal_test(): return state.utility(state.player())
    if depth <= 0:
        return advanced_heuristic(state)
    v = float("inf")
    for action in state.actions():
        v = min(v, max_value(state.result(action), alpha, beta, depth - 1))
        if v <= alpha: return v
        beta = min(beta, v)
    return v

def max_value(state, alpha, beta, depth):
    if state.terminal_test(): return state.utility(state.player())
    if depth <= 0:
        return advanced_heuristic(state)
    v = float("-inf")
    for action in state.actions():
        v = max(v, min_value(state.result(action), alpha, beta, depth - 1))
        if v >= beta: return v
        alpha = max(alpha, v)
    return v

def score(state):
    own_loc = state.locs[state.player()]
    opp_loc = state.locs[1 - state.player()]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)

def advanced_heuristic(state):
    own_loc = state.locs[state.player()]
    opp_loc = state.locs[1 - state.player()]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)

    own_distance_from_center = distance_from_center(own_loc)
    opp_distance_from_center = distance_from_center(opp_loc)
    own_distance_to_walls = distance_to_walls(own_loc)
    opp_distance_to_walls = distance_to_walls(opp_loc)

    blank_rate = 1 - state.ply_count / _HEIGHT * _WIDTH
    # if the loc close to center and away from walls, maximize move

    maximizing_own_weight = opp_distance_from_center - own_distance_from_center + own_distance_to_walls - opp_distance_to_walls
    if maximizing_own_weight > 0:
        return maximizing_own_weight * len(own_liberties) - blank_rate * len(opp_liberties)
    else:
        return len(own_liberties) - len(opp_liberties)

def distance_from_center(loc):
    # Use Mahattan Distance
    center = xy_index_loc(57)
    player = xy_index_loc(loc)
    return abs(center[0] - player[0]) + abs(center[1] - player[1])

def distance_to_walls(loc):
    # Get the minimum distance out of four direction
    player = xy_index_loc(loc)
    return min(player[0], player[1], _WIDTH - 1 - player[0], _HEIGHT - 1 - player[1])

def xy_index_loc(loc):
    return (loc % (_WIDTH + 2), loc // (_WIDTH + 2))

if __name__ == "__main__":
    open_book = build_table(NUM_ROUNDS)
    print(open_book)
    # opening book always chooses the middle square on an open board
    with open("data.pickle", 'wb') as f:
        pickle.dump(open_book, f)