#########################################
#                                       #
#                                       #
#  ==  SOKOBAN STUDENT AGENT CODE  ==   #
#                                       #
#      Written by: Shlok Goswami        #
#                                       #
#                                       #
#########################################


# SOLVER CLASSES WHERE AGENT CODES GO
from helper import *
import random
import math


# Base class of agent (DO NOT TOUCH!)
class Agent:
    def getSolution(self, state, maxIterations):

        '''
        EXAMPLE USE FOR TREE SEARCH AGENT:


        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            [ POP NODE OFF OF QUEUE ]

            [ EVALUATE NODE AS WIN STATE]
                [ IF WIN STATE: BREAK AND RETURN NODE'S ACTION SEQUENCE]

            [ GET NODE'S CHILDREN ]

            [ ADD VALID CHILDREN TO QUEUE ]

            [ SAVE CURRENT BEST NODE ]


        '''


        '''
        EXAMPLE USE FOR EVOLUTION BASED AGENT:
        #expand the tree until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            [ MUTATE ]

            [ EVALUATE ]
                [ IF WIN STATE: BREAK AND RETURN ]

            [ SAVE CURRENT BEST ]

        '''


        return []       # set of actions


#####       EXAMPLE AGENTS      #####

# Do Nothing Agent code - the laziest of the agents
class DoNothingAgent(Agent):
    def getSolution(self, state, maxIterations):
        if maxIterations == -1:     # RIP your machine if you remove this block
            return []

        #make idle action set
        nothActionSet = []
        for i in range(20):
            nothActionSet.append({"x":0,"y":0})

        return nothActionSet

# Random Agent code - completes random actions
class RandomAgent(Agent):
    def getSolution(self, state, maxIterations):

        #make random action set
        randActionSet = []
        for i in range(20):
            randActionSet.append(random.choice(directions))

        return randActionSet




#####    ASSIGNMENT 1 AGENTS    #####


# BFS Agent code
class BFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        initializeDeadlocks(state)
        iterations = 0
        bestNode = None
        queue = [Node(state.clone(), None, None)]
        visited = []

        #expand the tree until the iterations run out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(queue) > 0:
            iterations += 1

            # get the node from the front of the queue
            node = queue.pop(0)

            # evaluate the node as a win state
            if node.getState().isWin():
                bestNode = node
                break

            # add the node to visited list
            visited.append(node.getState())

            # get the children of the node
            children = node.getChildren()

            # add valid children to queue
            for child in children:
                if child.getState() not in visited and child.getState() not in [n.getState() for n in queue]:
                    queue.append(child)

            # save the current best node
            if bestNode is None or len(node.getActions()) < len(bestNode.getActions()):
                bestNode = node

        if bestNode is None:
            return []

        return bestNode.getActions()




# DFS Agent Code
class DFSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        initializeDeadlocks(state)
        iterations = 0
        bestNode = None
        stack = [Node(state.clone(), None, None)]
        visited = []

        #expand the tree until the iterations run out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and len(stack) > 0:
            iterations += 1

            # get the node from the top of the stack
            node = stack.pop()

            # evaluate the node as a win state
            if node.getState().isWin():
                bestNode = node
                break

            # add the node to visited list
            visited.append(node.getState())

            # get the children of the node
            children = node.getChildren()

            # add valid children to stack
            for child in reversed(children):
                if child.getState() not in visited and child.getState() not in [n.getState() for n in stack]:
                    stack.append(child)

            # save the current best node
            if bestNode is None or len(node.getActions()) < len(bestNode.getActions()):
                bestNode = node

        if bestNode is None:
            return []

        return bestNode.getActions()




# AStar Agent Code
class AStarAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        initializeDeadlocks(state)
        iterations = 0
        bestNode = None

        #initialize priority queue with the first node
        queue = PriorityQueue()
        queue.put(Node(state.clone(), None, None))

        #initialize visited set
        visited = set()

        #initialize scores
        gScore = {getHash(state): 0}
        hScore = {getHash(state): getHeuristic(state)}
        fScore = {getHash(state): getHeuristic(state)}

        #expand the tree until the iterations run out or a solution sequence is found
        while (iterations < maxIterations or maxIterations <= 0) and not queue.empty():
            iterations += 1

            #get the node with the lowest fScore value from the priority queue
            node = queue.get()

            #check if the node is a win state
            if node.getState().isWin():
                bestNode = node
                break

            #mark the node as visited
            visited.add(getHash(node.getState()))

            #get the children of the node
            children = node.getChildren()

            #process each child
            for child in children:
                childHash = getHash(child.getState())

                #skip if the child has already been visited
                if childHash in visited:
                    continue

                #calculate the tentative gScore for the child
                tentative_gScore = gScore[getHash(node.getState())] + 1

                #add the child to the queue if it hasn't been seen before
                if childHash not in gScore:
                    queue.put(child)
                #skip if the child has a higher gScore than the existing one
                elif tentative_gScore >= gScore[childHash]:
                    continue

                #update scores and pointers
                child.setParent(node)
                child.setAction(child.getAction())
                gScore[childHash] = tentative_gScore
                hScore[childHash] = getHeuristic(child.getState())
                fScore[childHash] = gScore[childHash] + hScore[childHash]

            #update the best node if necessary
            if bestNode is None or fScore[getHash(node.getState())] < fScore[getHash(bestNode.getState())]:
                bestNode = node

        #if no solution was found, return an empty list
        if bestNode is None:
            return []

        #otherwise, return the action sequence leading to the best node
        return bestNode.getActions()



#####    ASSIGNMENT 2 AGENTS    #####


# Hill Climber Agent code
class HillClimberAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        # setup
        intializeDeadlocks(state)
        iterations = 0

        seqLen = 50  # maximum length of the sequences generated
        coinFlip = 0.5  # chance to mutate

        # initialize the first sequence (random movements)
        bestSeq = []
        for i in range(seqLen):
            bestSeq.append(random.choice(directions))

        # mutate the best sequence until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations):
            iterations += 1

            # make a new sequence by copying the current best sequence
            newSeq = bestSeq.copy()

            # mutate the new sequence
            for i in range(seqLen):
                if random.random() < coinFlip:
                    newSeq[i] = random.choice(directions)

            # evaluate the new sequence
            newState = state.clone()
            for direction in newSeq:
                newState.update(direction["x"], direction["y"])
                if newState.isWin():
                    return newSeq

            # if the new sequence is better than the current best sequence, use it
            if len(newSeq) < len(bestSeq):
                bestSeq = newSeq

        # return the best sequence found
        return bestSeq
 



# Genetic Algorithm code
class GeneticAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        # setup
        intializeDeadlocks(state)

        iterations = 0
        seqLen = 50  # maximum length of the sequences generated
        popSize = 10  # size of the population to sample from
        parentRand = 0.5  # chance to select action from parent 1 (50/50)
        mutRand = 0.3  # chance to mutate offspring action

        bestSeq = []  # best sequence to use in case iterations max out

        # initialize the population with sequences of POP_SIZE actions (random movements)
        population = []
        for p in range(popSize):
            bestSeq = []
            for i in range(seqLen):
                bestSeq.append(random.choice(directions))
            population.append(bestSeq)

        # mutate until the iterations runs out or a solution sequence is found
        while (iterations < maxIterations):
            iterations += 1

            # 1. evaluate the population
            fitnesses = []
            for i in range(popSize):
                newState = state.clone()
                for direction in population[i]:
                    newState.update(direction["x"], direction["y"])
                fitnesses.append(getHeuristic(newState))

            # 2. sort the population by fitness (low to high)
            sortedPopulation = [x for _, x in sorted(zip(fitnesses, population))]

            # 2.1 save bestSeq from best evaluated sequence
            bestSeq = sortedPopulation[0]

            # 3. generate probabilities for parent selection based on fitness
            fitnessSum = sum(fitnesses)
            probabilities = [fitnesses[i] / fitnessSum for i in range(popSize)]

            # 4. populate by crossover and mutation
            new_pop = []
            for i in range(int(popSize / 2)):
                # 4.1 select 2 parents sequences based on probabilities generated
                par1 = sortedPopulation[np.random.choice(popSize, p=probabilities)]
                par2 = sortedPopulation[np.random.choice(popSize, p=probabilities)]

                # 4.2 make a child from the crossover of the two parent sequences
                offspring = []
                for j in range(seqLen):
                    if random.random() < parentRand:
                        offspring.append(par1[j])
                    else:
                        offspring.append(par2[j])

                # 4.3 mutate the child's actions
                for j in range(seqLen):
                    if random.random() < mutRand:
                        offspring[j] = random.choice(directions)

                # 4.4 add the child to the new population
                new_pop.append(list(offspring))

            # 5. add top half from last population (mu + lambda)
            for i in range(int(popSize / 2)):
                new_pop.append(sortedPopulation[i])

            # 6. replace the old population with the new one
            population = list(new_pop)

        # return the best found sequence
        return bestSeq



# MCTS Specific node to keep track of rollout and score
class MCTSNode(Node):
    def __init__(self, state, parent, action, maxDist):
        super().__init__(state,parent,action)
        self.children = []  #keep track of child nodes
        self.n = 0          #visits
        self.q = 0          #score
        self.maxDist = maxDist      #starting distance from the goal (heurstic score of initNode)

    #update get children for the MCTS
    def getChildren(self,visited):
        #if the children have already been made use them
        if(len(self.children) > 0):
            return self.children

        children = []

        #check every possible movement direction to create another child
        for d in directions:
            childState = self.state.clone()
            crateMove = childState.update(d["x"], d["y"])

            #if the node is the same spot as the parent, skip
            if childState.player["x"] == self.state.player["x"] and childState.player["y"] == self.state.player["y"]:
                continue

            #if this node causes the game to be unsolvable (i.e. putting crate in a corner), skip
            if crateMove and checkDeadlock(childState):
                continue

            #if this node has already been visited (same placement of player and crates as another seen node), skip
            if getHash(childState) in visited:
                continue

            #otherwise add the node as a child
            children.append(MCTSNode(childState, self, d, self.maxDist))

        self.children = list(children)    #save node children to generated child

        return children

    #calculates the score the distance from the starting point to the ending point (closer = better = larger number)
    def calcEvalScore(self,state):
        return self.maxDist - getHeuristic(state)

    #compares the score of 2 mcts nodes
    def __lt__(self, other):
        return self.q < other.q

    #print the score, node depth, and actions leading to it
    #for use with debugging
    def __str__(self):
        return str(self.q) + ", " + str(self.n) + ' - ' + str(self.getActions())



class MCTSAgent(Agent):
    def getSolution(self, state, maxIterations=-1):
        # setup
        initializeDeadlocks(state)
        iterations = 0
        bestNode = None
        initNode = MCTSNode(state.clone(), None, None, getHeuristic(state))

        while(iterations < maxIterations):
            iterations += 1

            # MCTS algorithm
            rollNode = self.treePolicy(initNode)
            score = self.rollout(rollNode)
            self.backpropagation(rollNode, score)

            # if in a win state, return the sequence
            if rollNode.getState().isWin():
                return rollNode.getActions()

            # set current best node
            bestNode = self.bestChildUCT(initNode)

            # if in a win state, return the sequence
            if bestNode.getState().isWin():
                return bestNode.getActions()

        # return solution of highest scoring descendant for best node
        # if this line was reached, that means the iterations timed out before a solution was found
        return self.bestActions(bestNode)

    # returns the descendant with the best action sequence based on score
    def bestActions(self, node):
        # no node given - return nothing
        if node is None:
            return []

        bestActionSeq = []
        while len(node.children) > 0:
            node = self.bestChildUCT(node)

        return node.getActions()

    #### MCTS SPECIFIC FUNCTIONS BELOW ####

    # determines which node to expand next
    def treePolicy(self, rootNode):
        curNode = rootNode
        visited = set()

        while not curNode.getState().isTerminal():
            children = curNode.getChildren(visited)
            if children:
                curNode = children[0]
                visited.add(getHash(curNode.getState()))
            else:
                break

        return curNode

    # uses the exploitation/exploration algorithm
    def bestChildUCT(self, node):
        c = 1  # c value in the exploration/exploitation equation
        bestChild = None
        bestScore = float("-inf")

        for child in node.children:
            exploit = child.q / child.n
            explore = c * math.sqrt(math.log(node.n) / child.n)
            score = exploit + explore
            if score > bestScore:
                bestChild = child
                bestScore = score

        return bestChild

    # simulates a score based on random actions taken
    def rollout(self, node):
        numRolls = 10  # number of times to rollout to
        state = node.getState().clone()

        while numRolls > 0 and not state.isTerminal():
            actions = state.getPossibleActions()
            if not actions:
                break
            action = random.choice(actions)
            state.update(action["x"], action["y"])
            numRolls -= 1

        score = self.evalScore(node, state)
        return score

    # updates the score all the way up to the root node
    def backpropagation(self, node, score):
        while node is not None:
            node.n += 1
            node.q += score
            node = node.getParent()

        return

    # calculates the score of a node based on the distance from the starting point to the ending point
    def evalScore(self, node, state):
        maxDist = node.maxDist
        return maxDist - getHeuristic(state)

