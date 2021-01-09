import math
import model
import random as rng
import sys

class MFTreeNode:
    __variances = None
    def __init__(self, ID, numActions, numFidelities, models=None, tree=None):
        self.ID = ID
        # Position in the tree
        self.parent = None
        self.children = [None] * numActions
        self.missingChildren = numActions
        # Mean and number of visits, per fidelity
        self.m = [0.] * numFidelities
        self.n = [0.] * numFidelities
        #self.v = [0.] * numFidelities
        # The root of the tree maintains the models
        self.models = models
        if models is not None:
            MFTreeNode.__variances = [[1. for i in range(numActions)] for j in range(numFidelities)]
            #print(MFTreeNode.__variances, file=sys.stderr)
        self.tree = tree

    # Updates the mean and variance values for a node for a given fidelity
    def update(self, value, f, childID = None):
        self.n[f] += 1
        delta = value - self.m[f]
        self.m[f] += delta / self.n[f]
        # Root node
        if self.parent is None:
            # If we update a model, update our children's mean and variance values
            if self.models[f].add(childID, value):
                #print(file=sys.stderr)
                #print("Updating fidelity", f, file=sys.stderr)
                #print("| Visited ", self.n[f], "times", file=sys.stderr)
                #self.models[f].plotPosterior(self.children, ubFunc=self.tree.nodeUB)
                for child in self.children:
                    child.m[f] = model.Model.denormalize(self.models[f].getMean(child.ID).numpy())
                    MFTreeNode.__variances[f][child.ID] = (self.models[f].getStd(child.ID).numpy() ** 2)
                    # Debug info
                    """
                    oldUB = self.tree.nodeUB(child)
                    oldMean = child.mean(f)
                    oldVar = MFTreeNode.__variances[f][child.ID]
                    print("| ACTION", child.ID, file=sys.stderr)
                    print("| | Mean:", oldMean, "to", child.mean(f), file=sys.stderr)
                    print("| | Var:", oldVar, "to", child.variance(f), file=sys.stderr)
                    print("| | UB:", oldUB, "to", self.tree.nodeUB(child), file=sys.stderr)
                    print("| / N: ", child.n[f], file=sys.stderr)
                maxL = list(map(self.tree.nodeUB, self.children))
                maxVal = max(list(map(self.tree.nodeUB, self.children)))
                bestChild = rng.choice( [c for c in self.children if self.tree.nodeUB(c) >= maxVal] )
                print("| UBs:", maxL, file=sys.stderr)
                print("| MaxVal", maxVal, "best child", bestChild.ID, file=sys.stderr)
                seq, _ = self.tree.getActionSequence()
                print("| Suggested sequence:", seq, file=sys.stderr)
                print("/", file=sys.stderr)
                #"""
        # Backpropagate, aka update our parent
        else:
            self.parent.update(value, f, self.ID)

    # Add new child
    def addChild(self, action, fidelity, result):
        #print("inputs: ", action, fidelity , result, file=sys.stderr)
        if self.children[action] is not None:
            #print(self.children, file=sys.stderr)
            raise Exception("Child already exists")
        elif self.missingChildren <= 0:
            raise Exception("Cannot add any more children")
        # Lengths correspond to numActions and numFidelities, respectively
        child = MFTreeNode(action, len(self.children), len(self.m))
        child.parent = self
        child.update(result, fidelity, None)
        self.children[action] = child
        self.missingChildren -= 1

    # Randomly selects a missing child
    def getRandomMissingChild(self):
        missing = [(idx, child) for idx, child in enumerate(self.children) if child is None]
        return rng.choice(missing)[0]

    # Retrieve the normalized mean
    def mean(self, fidelity):
        return model.Model.normalize(self.m[fidelity])

    def variance(self, fidelity):
        return MFTreeNode.__variances[fidelity][self.ID]

    def stdev(self, fidelity):
        return self.variance(fidelity) ** 0.5

class MFTree:
    def __init__(self, numActions, numFidelities, kernel, updateCycle, numTrainingIters, modelPath = ""):
        # Initialize our models
        self.models = []
        self.slackValues = []
        self.thresholds = []
        for i in range(numFidelities):
            m = model.Model(i, modelPath)
            m.setParams(kernel, numActions, updateCycle, numTrainingIters)
            self.thresholds.append(0.001)
            self.slackValues.append( 0.01 * (numFidelities - i - 1) / numFidelities)
            #m.plot = True
            self.models.append(m)
        # Initialize our root node
        self.root = MFTreeNode(None, numActions, numFidelities, models=self.models, tree=self)
        self.t = 0
        self.ucbBias = math.sqrt(2)
        # Save some values for later use
        self.numActions = numActions
        self.numFidelities = numFidelities
        self.lastF = None
        self.lastFctr = 0

    # Manage our slack values and thresholds. Call after prep and before testing
    def setParams(self, costs):
        self.costs = costs

    # Select the next set of samples. Includes a fidelity and a sequence of nodes leading to a leaf node
    def next(self):
        self.t += 1
        seq, node = self.getActionSequence(self.root)
        f = self.__getFidelity(seq[0])
        # If we stay at a fidelity for too long, increase the threshold
        if self.lastF != f:
            self.lastF = f
            self.lastFctr = 0
        elif f < self.numFidelities - 1:
            self.lastFctr += 1
            if self.lastFctr > (self.costs[f+1] * 10. / self.costs[f]):
                self.thresholds[f] *= 2
                #print("---", self.thresholds, file=sys.stderr)
                self.lastFctr = 0
        # Debug info
        """
        if(self.t % 25 == 0):
            print(file=sys.stderr)
            for child in self.root.children:
                print(child.ID, ":", 
                    list(map(lambda x: self.__actionUB(child, x), range(self.numFidelities))), 
                    file=sys.stderr)
            print("selected fidelity", f, "sequence", seq, file=sys.stderr)
        #"""
        return f, seq, node

    # Select the fidelity to sample with
    def __getFidelity(self, action):
       # print("-------", file=sys.stderr, flush=True)
        for f in range(self.numFidelities - 1):
           # print("F:", f, "beta:", self.__beta(), "std:", self.models[f].getStd(action), "threshold",
           #         self.thresholds[f], file=sys.stderr, flush=True)
            if self.__beta() * self.models[f].getStd(action) >= self.thresholds[f]:
                return f
        return self.numFidelities - 1

    # Get the next node to rollout, which requires reporting a sequence of actions
    def getActionSequence(self, random = False, node = None):
        # Root node uses MF-GP-UCB, other nodes use UCT
        nodeFunc = self.__nodeUCT
        if node is None:
            node = self.root
            nodeFunc = self.__nodeUB
        # Expand nodes that aren't fully expanded
        if node.missingChildren > 0:
            return [node.getRandomMissingChild()], node
        bestChild = None
        # Randomly select a child
        if random is True:
            bestChild = rng.choice(node.children)
        # Else select one of the most promising children (ties are possible)
        else:
            maxVal = max(map(nodeFunc, node.children))
            bestChild = rng.choice( [c for c in node.children if nodeFunc(c) >= maxVal] )
        
        seq, node = self.getActionSequence(random, bestChild)
        return [bestChild.ID] + seq, node

    # MF-GP-UCB: Get the tightest upper bound for a given node
    def nodeUB(self, node):
        return self.__nodeUB(node)

    def __nodeUB(self, node):
        return min(map(lambda f: self.__actionUB(node, f), range(self.numFidelities)))

    # MF-GP-UCB: Get the upper bound for a given node and fidelity
    def __actionUB(self, node, f):
        if node.n[f] < 2:
            return 10.
        return node.mean(f) + (self.__beta()*node.stdev(f)) + self.slackValues[f]

    # UCT: Tightest upper bound for a given node
    def __nodeUCT(self, node):
        return min(map(lambda f: self.__actionUCT(node, f), range(self.numFidelities)))

    def __actionUCT(self, node, f):
        return node.mean(f) + (self.ucbBias * math.sqrt(math.log(max(node.parent.n[f], 1)) / max(node.n[f], 1))) 

    # Beta function
    def __beta(self):
        return (0.2 * self.numActions * math.log(2 * self.t))

    # Get the best action, per a metric
    def getBest(self, metric):
        highest = self.numFidelities - 1
        func = None
        # Highest mean
        if metric == "m":
            func = lambda x: x.mean(highest)
        # Most visited by highest fidelity
        elif metric == "v":
            func = lambda x: x.n[highest]
        # Most visited by any fidelity
        elif metric == "w":
            func = lambda x: sum(x.n)
            maxVal = max(map(lambda x: sum(x.n), self.root.children))
        # Unrecognized
        else:
            raise Exception("getBest(): unrecognized metric \"" + metric + "\"")
        # Break ties randomly
        values = list(map(func, self.root.children))
        maxVal = max(values)
        return rng.choice(list(filter(lambda x: func(x) == maxVal, self.root.children))), values
