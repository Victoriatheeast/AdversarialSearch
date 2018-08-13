import random

from isolation.isolation import _WIDTH, _HEIGHT
from sample_players import DataPlayer

NUM_ROUNDS = 100

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        #
        # Use combination of iterative deepening with alpha-beta-pruning
        #
        if state.ply_count < 4:
            if self.data and state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            best_move = None
            depth_limit = 5
            for depth in range(1, depth_limit + 1):
                best_move = self.alpha_beta_pruning(state, depth)
            self.queue.put(best_move)

    def alpha_beta_pruning(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        max_score = float("-inf")
        best_action = None
        for action in state.actions():
            v = self.min_value(state.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, v)
            if v >= max_score:
                max_score = v
                best_action = action
        return best_action

    def min_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.advanced_heuristic(state)
        v = float("inf")
        for action in state.actions():
            v = min(v, self.max_value(state.result(action), alpha, beta, depth - 1))
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)
        if depth <= 0: return self.advanced_heuristic(state)
        v = float("-inf")
        for action in state.actions():
            v = max(v, self.min_value(state.result(action), alpha, beta, depth - 1))
            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def advanced_heuristic(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        own_distance_from_center = self.distance_from_center(own_loc)
        opp_distance_from_center = self.distance_from_center(opp_loc)
        own_distance_to_borders = self.distance_to_borders(own_loc)
        opp_distance_to_borders = self.distance_to_borders(opp_loc)
        blank_rate = 1 - state.ply_count / (_HEIGHT * _WIDTH)
        # if the loc close to center and away from borders, maximize move
        maximizing_own_weight = (opp_distance_from_center - own_distance_from_center) + (own_distance_to_borders - opp_distance_to_borders)
        if maximizing_own_weight > 0:
            return maximizing_own_weight * len(own_liberties) - blank_rate * len(opp_liberties)
        else:
            return len(own_liberties) - len(opp_liberties)

    def distance_from_center(self, loc):
        # Use Mahattan Distance
        center = self.xy_index_loc(57)
        player = self.xy_index_loc(loc)
        return abs(center[0] - player[0]) + abs(center[1] - player[1])

    def distance_to_borders(self, loc):
        # Get the distance to the closest border
        player = self.xy_index_loc(loc)
        return min(player[0], player[1], _WIDTH - 1 - player[0], _HEIGHT - 1 - player[1])

    def xy_index_loc(self, loc):
        return (loc % (_WIDTH + 2), loc // (_WIDTH + 2))





