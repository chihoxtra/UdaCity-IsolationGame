"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(gameState, player):
  
    # check winning and losing
    if gameState.is_loser(player):
        return float("-inf")

    if gameState.is_winner(player):
        return float("inf")        
    
    # center/location promximity to center
    w, h = int(gameState.width / 2), int(gameState.height / 2)
    y, x = gameState.get_player_location(player)
    #e_pos_scores = math.exp(-1.*(((h - y)**2 + (w - x)**2)/(2*0.05)**2))
    e_pos_scores = 1./(((h - y)**2 + (w - x)**2) + 1e-7)
    
    # no of moves
    player2 = gameState.get_opponent(player)
    mov_score = len(gameState.get_legal_moves(player)) - 5*len(gameState.get_legal_moves(player2))
        
    return float(mov_score + e_pos_scores)

def custom_score_1(gameState, player):
    # check winning and losing
    if gameState.is_loser(player):
        return float("-inf")

    if gameState.is_winner(player):
        return float("inf")        
    
    # center/location promximity to center
    w, h = int(gameState.width / 2), int(gameState.height / 2)
    y, x = gameState.get_player_location(player)
    #e_pos_scores = math.exp(-1.*(((h - y)**2 + (w - x)**2)/(2*0.05)**2))
    e_pos_scores = 1./(((h - y)**2 + (w - x)**2) + 1e-7)
    
    # no of moves
    player2 = gameState.get_opponent(player)
    mov_score = len(gameState.get_legal_moves(player)) - 10*len(gameState.get_legal_moves(player2))
        
    return float(mov_score + e_pos_scores)


def custom_score_2(gameState, player):
    # check winning and losing
    if gameState.is_loser(player):
        return float("-inf")

    if gameState.is_winner(player):
        return float("inf")        
    
    # center/location promximity to center
    w, h = int(gameState.width / 2), int(gameState.height / 2)
    y, x = gameState.get_player_location(player)
    #e_pos_scores = math.exp(-1.*(((h - y)**2 + (w - x)**2)/(2*0.05)**2))
    e_pos_scores = 1./(((h - y)**2 + (w - x)**2) + 1e-7)
    
    # no of moves
    player2 = gameState.get_opponent(player)
    mov_score = len(gameState.get_legal_moves(player)) - 2*len(gameState.get_legal_moves(player2))
        
    return float(mov_score + e_pos_scores)


def custom_score_3(gameState, player):
    # check winning and losing
    if gameState.is_loser(player):
        return float("-inf")

    if gameState.is_winner(player):
        return float("inf")        
    
    # center/location promximity to center
    w, h = int(gameState.width / 2), int(gameState.height / 2)
    y, x = gameState.get_player_location(player)
    e_pos_scores = math.exp(-1.*(((h - y)**2 + (w - x)**2)/(2*0.05)**2))
    
    # no of moves
    player2 = gameState.get_opponent(player)
    mov_score = len(gameState.get_legal_moves(player)) - 1000*len(gameState.get_legal_moves(player2))
        
    return float(mov_score + e_pos_scores)


def custom_score_4(gameState, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # check winning and losing
    if gameState.is_loser(player):
        return float("-inf")

    if gameState.is_winner(player):
        return float("inf")        
    
    # no of moves
    player2 = gameState.get_opponent(player)
    mov_score = len(gameState.get_legal_moves(player)) - 1000*len(gameState.get_legal_moves(player2))

def custom_score_5(gameState, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # check winning and losing
    if gameState.is_loser(player):
        return float("-inf")

    if gameState.is_winner(player):
        return float("inf")        
    
    # center/location promximity to center
    player2 = gameState.get_opponent(player)
    noMyMoves = len(gameState.get_legal_moves(player))
    noOppMoves = len(gameState.get_legal_moves(player2))
    

    return noMyMoves - noOppMoves*1000



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
            

    def get_move(self, game, time_left):

        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        #print(best_move)
        return best_move

    
    def min_value(self, gameState, level, depth):
        """ 
        Input: gameState: instance of the game
        level: current level of iteration, start from 1
        depth: desired level of depth to reach
        return: score
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            #print("MiniMax time out!")
            raise SearchTimeout()
            
        levelScores = [] #scores of this level
        levelMovesList = gameState.get_legal_moves() #list of valid moves on this level
            
        if len(levelMovesList) > 0: 
            if level == 1:

                for move in levelMovesList:
                    levelScores.append(self.score(gameState.forecast_move(move), self))

            elif (level < depth):

                for move in levelMovesList:

                    #For each moves in this level, get the min score the next level
                    nextLevelMaxScore = self.max_value(gameState.forecast_move(move), level-1, depth)

                    levelScores.append(nextLevelMaxScore)
                    
            minScore =  min(levelScores)
            
        else:
            
            minScore = -100  #running out of moves
            #print("running out of moves")

        #print(min(levelScores))
        return minScore

    
    def max_value(self, gameState, level, depth):
        """ 
        Input: gameState: instance of the game
        level: current level of iteration, start from 1
        depth: desired level of depth to reach
        return: score
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            #print("MiniMax time out!")
            raise SearchTimeout()
            
        levelScores = [] #scores of this level
        levelMovesList = gameState.get_legal_moves() #list of valid moves on this level        
        
        if len(levelMovesList) > 0: 
            if level == 1:

                for move in levelMovesList:
                    levelScores.append(self.score(gameState.forecast_move(move), self))

                #maxLevelScore = max(levelScores)

            elif (level < depth):

                for move in levelMovesList:

                    #For each moves in this level, get the min score the next level
                    nextLevelMinScore = self.min_value(gameState.forecast_move(move), level-1, depth)

                    levelScores.append(nextLevelMinScore)
                    
            maxScore =  max(levelScores)
        else:
            maxScore = -100  #running out of moves
            #print("running out of moves")    
            
        return maxScore
    
    
    def minimax(self, gameState, depth):

        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            #print("MiniMax time out!")
            raise SearchTimeout()        
        
        levelScores = [] #scores of this level
        levelMovesList = gameState.get_legal_moves() #list of valid moves on this level 
        
        if len(levelMovesList) > 0:
            
            if depth == 1:

                for move in levelMovesList:

                    levelScores.append(self.score(gameState.forecast_move(move) , self))
                
                #print((levelScores))
                
            else:
                level = depth
                
                for move in levelMovesList:

                    #For each moves in this level, get the min score the next level
                    nextLevelMinScore = self.min_value(gameState.forecast_move(move), level-1, depth)

                    levelScores.append(nextLevelMinScore)    
            
            the_move = levelMovesList[levelScores.index(max(levelScores))]
        
        else:
            the_move = (-1,-1)   #running out of moves
            
            
        return the_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    TIMER_THRESHOLD = 15.

    def get_move(self, game, time_left):

        self.time_left = time_left
                
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_Move_List = []
        best_Move_List.append((-1, -1))
        
        max_depth = game.width * game.height #depth assigned is ignored
        
        for d in range(1,max_depth+1):
            
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_Move_List.append(self.alphabeta(game, d))

            except SearchTimeout: # Handle any actions required after timeout as needed
                #print("AlphaBeta time out!")
                return best_Move_List[-1]

        # Return the best move from the last completed search iteration
        return best_Move_List[-1]

        
    def terminate_check(self, gameState):
        if self.time_left() < self.TIMER_THRESHOLD:
            #print("AlphaBeta time out!")
            raise SearchTimeout()        
        
        if gameState.is_winner(self):
            return "lose"
        elif gameState.is_loser(self):
            return "won"
        else:
            return ""

    def min_value(self, gameState, depth, alpha = float("-inf"), beta = float("inf")):
        """ 
        Input: gameState: instance of the game
        level: current level of iteration, start from 1
        depth: desired level of depth to reach
        alpha: minimal lower bound for max player, initial value is very negative
        return: score
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            #print("AlphaBeta time out!")
            raise SearchTimeout()
        

        if self.terminate_check(gameState) == "won":
            return float("inf")
        
        elif self.terminate_check(gameState) == "lose":
            return float("-inf")
            
        else:

            levelMovesList = gameState.get_legal_moves() #list of valid moves on this level
            
            minLevelScore = float("inf")
            
            if depth <= 1:

                for move in levelMovesList:
                    
                    minLevelScore = min(minLevelScore, self.score(gameState.forecast_move(move), self))
                    
                    if minLevelScore <= alpha:
                        return minLevelScore
            else:

                for move in levelMovesList:

                    #For each moves in this level, get the min score the next level
                    nextLevelMaxScore = self.max_value(gameState.forecast_move(move), depth-1, alpha, beta)
                    
                    minLevelScore = min(minLevelScore, nextLevelMaxScore)

                    if minLevelScore <= alpha:
                        return minLevelScore
                    
                    beta = min(beta, minLevelScore)

        return minLevelScore
    
    
    
    def max_value(self, gameState, depth, alpha = float("-inf"), beta = float("inf")):
        """ 
        Input: gameState: instance of the game
        beta: maximum upper bound of min player: start with a large +ve value
        level: current level of iteration, start from 1
        depth: desired level of depth to reach
        return: score, Beta
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            

        if self.terminate_check(gameState) == "won":
            return float("inf")
        
        elif self.terminate_check(gameState) == "lose":
            return float("-inf")
            
        else: 
            
            levelMovesList = gameState.get_legal_moves() #list of valid moves on this level    
            
            maxLevelScore = float("-inf")
            
            if depth <= 1:

                for move in levelMovesList:
                    
                    maxLevelScore = max(maxLevelScore, self.score(gameState.forecast_move(move), self))
                    
                    if maxLevelScore >= beta: #beta: float("inf") initially, cause upper is a min fx
                        return maxLevelScore

            else:
                
                for move in levelMovesList:

                    #For each moves in this level, get the min score the next level
                    nextLevelMinScore = self.min_value(gameState.forecast_move(move), depth-1, alpha, beta)
                    
                    maxLevelScore = max(maxLevelScore, nextLevelMinScore)                    
                    
                    if maxLevelScore >= beta: #initailly: float("inf"); upper level is a min fx
                        return maxLevelScore
                        
                    alpha = max(alpha, maxLevelScore) #alpha is for max fx
        
        return maxLevelScore
    

    def alphabeta(self, gameState, depth, alpha=float("-inf"), beta=float("inf")):


        if self.time_left() < self.TIMER_THRESHOLD:
            #print("AlphaBeta time out!")
            raise SearchTimeout()    
        
        #implemented a replica of max value to track the position of move
        levelScores = [] #scores of this level
        levelMovesList = gameState.get_legal_moves() #list of valid moves on this level 
        
        if self.terminate_check(gameState) == "won":
            if bool(levelMovesList):
                return levelMovesList[0]
            else:
                return (-1,-1)
        
        elif self.terminate_check(gameState) == "lose":
            return (-1,-1)
            
        else: 
            
            if depth <= 1: # just count this level!

                for move in levelMovesList:

                    levelScores.append(self.score(gameState.forecast_move(move) , self))
                
                #for i in range(len(levelMovesList)):
                    #print(levelScores[i])
                    #print(levelMovesList[i])                
            else:
                
                for move in levelMovesList:

                    #For each moves in this level, get the min score the next level
                    nextLevelMinScore = self.min_value(gameState.forecast_move(move), depth-1, alpha, beta)
                    
                    alpha = max(alpha, nextLevelMinScore)
                    
                    levelScores.append(nextLevelMinScore)    
            
            the_move = levelMovesList[levelScores.index(max(levelScores))]
            
            
        return the_move