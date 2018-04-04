# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        MIN_SCORE = -100000000

        if action == 'Stop':
            return MIN_SCORE

        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0 and ghostState.getPosition() == newPos:
                return MIN_SCORE

        foodList = currentGameState.getFood().asList()
        distances = [-1 * util.manhattanDistance(food, newPos) for food in foodList]
        return max(distances)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # Some constants
        MIN_VAL = float("-inf")
        MAX_VAL = float("inf")
        NUM_GHOSTS = gameState.getNumAgents() - 1

        def Maximizer(state, depth):
            # At terminal node
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # At non-ternimal node
            pacmanActions = state.getLegalActions(0)
            maxVal = MIN_VAL
            for action in pacmanActions:
                pacmanState = state.generateSuccessor(0, action)
                maxVal = max(maxVal, Minimizer(pacmanState, depth, 1))
            return maxVal

        def Minimizer(state, depth, ghostIndex):
            # At terminal node
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # At non-terminal node
            ghostActions = state.getLegalActions(ghostIndex)
            minVal = MAX_VAL
            for action in ghostActions:
                ghostState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == NUM_GHOSTS: # all ghosts have iterated, go to next depth
                    minVal = min(minVal, Maximizer(ghostState, depth-1))
                else: # check for next ghosts
                    minVal = min(minVal, Minimizer(ghostState, depth, ghostIndex+1))
            return minVal

        # Minimax search start here
        pacmanActions = gameState.getLegalActions(0)
        bestScore = MIN_VAL

        for action in pacmanActions:
            nextState = gameState.generateSuccessor(0, action)
            score = Minimizer(nextState, self.depth, 1) # start with the 1st ghost

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Some constants
        MIN_VAL = float("-inf")
        MAX_VAL = float("inf")
        NUM_GHOSTS = gameState.getNumAgents() - 1

        def Maximizer(state, depth, a, b):
            # At terminal node
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # At non-ternimal node
            pacmanActions = state.getLegalActions(0)
            for action in pacmanActions:
                pacmanState = state.generateSuccessor(0, action)
                a = max(a, Minimizer(pacmanState, depth, 1, a, b))

                if a >= b:
                    break
            return a

        def Minimizer(state, depth, ghostIndex, a, b):
            # At terminal node
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # At non-terminal node
            ghostActions = state.getLegalActions(ghostIndex)
            for action in ghostActions:
                ghostState = state.generateSuccessor(ghostIndex, action)

                if ghostIndex == NUM_GHOSTS: # all ghosts have iterated, go to next depth
                    b = min(b, Maximizer(ghostState, depth-1, a, b))
                else: # check for next ghosts
                    b = min(b, Minimizer(ghostState, depth, ghostIndex+1, a, b))

                if b <= a:
                    break
            return b

        # AlphaBeta search starts here
        pacmanActions = gameState.getLegalActions(0)
        bestScore = MIN_VAL
        alpha = MIN_VAL
        beta = MAX_VAL

        for action in pacmanActions:
            nextState = gameState.generateSuccessor(0, action)
            score = Minimizer(nextState, self.depth, 1, alpha, beta) # start with the 1st ghost

            if score > bestScore:
                bestScore = score
                bestAction = action
            if bestScore >= beta:
                break
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Some constants
        MIN_VAL = float("-inf")
        MAX_VAL = float("inf")
        NUM_GHOSTS = gameState.getNumAgents() - 1

        def Maximizer(state, depth):
            # At terminal node
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # At non-ternimal node
            pacmanActions = state.getLegalActions(0)
            maxVal = MIN_VAL
            for action in pacmanActions:
                pacmanState = state.generateSuccessor(0, action)
                maxVal = max(maxVal, Minimizer(pacmanState, depth, 1))
            return maxVal

        def Minimizer(state, depth, ghostIndex):
            # At terminal node
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # At non-terminal node
            ghostActions = state.getLegalActions(ghostIndex)
            minVal = 0
            numActions = float(len(ghostActions))
            prob = float(1/numActions)
            for action in ghostActions:
                ghostState = state.generateSuccessor(ghostIndex, action)
                if ghostIndex == NUM_GHOSTS: # all ghosts have iterated, go to next depth
                    minVal += Maximizer(ghostState, depth-1) * prob
                else: # check for next ghosts
                    minVal += Minimizer(ghostState, depth, ghostIndex+1) * prob
            print(minVal)
            return minVal

        # Minimax search start here
        pacmanActions = gameState.getLegalActions(0)
        bestScore = MIN_VAL

        for action in pacmanActions:
            nextState = gameState.generateSuccessor(0, action)
            score = Minimizer(nextState, self.depth, 1) # start with the 1st ghost

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      Algorithm:
      The final score is computed by a linear combination of following features:
        1. total number of food remaining
        2. total number of capsules remaining
        3. distance to the closest food
        4. distance to the closest normal ghost
        5. distance to the closest scared ghost

      Pacman is encouraged to eat food, but it should not chase for the capsules.
      So it is designed to eat capsules whenever it passes them. Normal ghosts
      should be avoided, so further is better. When ghosts turn to scared, it is
      encouraged to chase ghost.

    """
    "*** YOUR CODE HERE ***"

    # distance to food
    # distance to ghost
    # distance to power pellet

    MIN_VAL = float("-inf")
    MAX_VAL = float("inf")
    NUM_GHOSTS = currentGameState.getNumAgents() - 1

    if currentGameState.isWin():
        return MAX_VAL
    if currentGameState.isLose():
        return MIN_VAL

    curScore = currentGameState.getScore()
    curPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()

    # number of foods
    numFoods = len(foods)

    # number of capsules
    numCapsules = len(capsules)

    # distance to closest food
    disFood = min(map(lambda f: util.manhattanDistance(curPosition, f), foods))

    # ghosts
    normalG = []
    scaredG = []
    for ghost in ghosts:
        if ghost.scaredTimer:
            scaredG.append(ghost)
        else:
            normalG.append(ghost)

    if len(normalG):
        disNormalG = min(map(lambda g: util.manhattanDistance(curPosition, g.getPosition()), normalG))
        wNormalG = -2
    else:
        disNormalG = 1
        wNormalG = 0

    if len(scaredG):
        disScaredG = min(map(lambda g: util.manhattanDistance(curPosition, g.getPosition()), scaredG))
        wScaredG = -2
    else:
        disScaredG = 0
        wScaredG = 0

    score = curScore + \
            -4 * numFoods + \
            -20 * numCapsules + \
            -1 * disFood + \
            wNormalG * (1 / disNormalG) + \
            wScaredG * disScaredG

    return score

# Abbreviation
better = betterEvaluationFunction
