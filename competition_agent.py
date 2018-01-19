"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
import random


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
    e_pos_scores = 1./(((h - y)**2 + (w - x)**2) + 1e-10)
    
    # no of moves
    player2 = gameState.get_opponent(player)
    mov_score = len(gameState.get_legal_moves(player)) - 1000*len(gameState.get_legal_moves(player2))
        
    return float(mov_score + e_pos_scores)


class CustomPlayer:
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
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