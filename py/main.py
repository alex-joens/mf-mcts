from enum import Enum
import multiprocessing as mp
import sys

import messaging
import mf_tree as tree

path = "C:\\dev\\mf-mcts"


# Root class for the program. OOP is a hard habit to kick
class Program:

    # Args are supplied from the command line, and will be amended
    def __init__(self, msgQ, readTimeout):
        self.messageHandler = messaging.Handler(msgQ, readTimeout)
        self.mode = self.Mode.START
        self.run = True

    # When changing modes, we only have to acknowledge
    def __changeMode(self, args):
        # Exiting mode...
        if self.mode is self.Mode.PREP:
            for m in self.tree.models:
                m.ignoreUpdate = False
                m.updateModel()

        newMode = self.Mode[args[0]]
        self.mode = newMode
        self.fn = self.loops[self.mode]            

        # Entering mode...
        if newMode is self.Mode.PREP:
            self.node = None
            self.fidelity = 0
            self.budgets = []
            # Don't update our tree's GP models until we're done collecting data
            for m in self.tree.models:
                m.ignoreUpdate = True
        elif newMode is self.Mode.TRAIN:
            self.node = None
            self.fidelity = 0
            self.remainingBudget = self.budget
            self.tree.setParams(self.costs)
        elif newMode is self.Mode.QUIT:
            self.run = False
            return
        
        self.messageHandler.ack()

    # The main function, which loops forever until a kill signal is sent or the program times out
    def start(self):
        # Signals that we're ready to go
        self.messageHandler.ack()
        while self.run:
            message = self.messageHandler.get()
            if message is None:
                break
            
            if message.cmd == "!":
                self.__changeMode(message.args)
            # Run the function for the current state
            elif self.fn is not None:
                self.fn(self, message)
            # All of the above will send a response, so we don't have to

            
    # (Re-)initialize the search tree and related parameters
    def __init(self, msg):
        # Tree info
        if msg.cmd == "T":
            self.kernel = msg.args[0]
            self.numActions, self.numFidelities, self.updateCycle, self.numTrainingIters = list(map(int, msg.args[1:]))
            self.tree = tree.MFTree(self.numActions, self.numFidelities, self.kernel, self.updateCycle, self.numTrainingIters, path)
        # Budget info
        elif msg.cmd == "B":
            self.budget = int(msg.args[0])
            self.costs = list(map(float, msg.args[1:]))
        # Reuse previous parameters
        elif msg.cmd == "R":
            self.numActions = int(msg.args[0])
            self.tree = tree.MFTree(self.numActions, self.numFidelities, self.kernel, self.updateCycle, self.numTrainingIters, path)
        # Unrecognized command
        else:
            self.messageHandler.error("INIT: unrecognized cmd \"" + msg.cmd + "\"")
            return
        # No data will be sent in response
        self.messageHandler.ack()


    # Collect some initial data for the tree
    def __prep(self, msg):
        # Determine training budget (split equally among fidelities)
        if msg.cmd == "S":
            trainingBudget = float(msg.args[0])
            for _ in range(self.numFidelities):
                self.budgets.append(int(trainingBudget / self.numFidelities))
            self.budgets.append(0.)
        # Process training data
        elif msg.cmd == "D":
            result = float(msg.args[0])
            self.node.addChild(self.action, self.fidelity, result)
        # Error
        else:
            self.messageHandler.error("PREP: unrecognized cmd \"" + msg.cmd + "\"")
            return

        # Pick the next fidelity to expand, granting it any leftover budget
        while self.fidelity < self.numFidelities:
            if self.budgets[self.fidelity] < self.costs[self.fidelity]:
                self.budgets[self.fidelity + 1] += self.budgets[self.fidelity]
                self.budgets[self.fidelity] = 0
                self.fidelity += 1
            else:
                break
        # We've exhausted our budget
        if self.fidelity == self.numFidelities:
            self.messageHandler.ack()
            return
        # Else, select a new node to expand
        seq, self.node = self.tree.getActionSequence(random=True)
        self.action = seq[-1]
        self.budgets[self.fidelity] -= self.costs[self.fidelity]
        self.__getSample(self.fidelity, seq)


    # Similar to the above
    def __train(self, msg):
        # Start the training process
        if msg.cmd == "S":
            pass
        # Process training data
        elif msg.cmd == "D":
            result = float(msg.args[0])
            self.node.addChild(self.action, self.fidelity, result)
        # Error
        else:
            self.messageHandler.error("TRAIN: unrecognized cmd \"" + msg.cmd + "\"")
            return            

        # Get the next sample
        self.fidelity, seq, self.node = self.tree.next()
        self.action = seq[-1]
        self.remainingBudget -= self.costs[self.fidelity]
        # Exit if our budget is depleted
        if self.remainingBudget < 0:
            self.messageHandler.ack()
            return
        # Else get this sample
        self.__getSample(self.fidelity, seq)

    # Return which action is most promising
    def __test(self, msg):
        # The only command
        if msg.cmd == "T":
            metric = msg.args[0]
            bestAction, options = self.tree.getBest(metric)
            self.messageHandler.act(bestAction.ID, options)

    # Draws a sample from the simulator and returns the results
    def __getSample(self, fidelity, actionSequence):
        #print(actionSequence, file=sys.stderr)
        sampleMsg = messaging.Message.requestSample(fidelity, actionSequence)
        self.messageHandler.send(sampleMsg)

    # The program is stateful, and for convenience this is made explicit
    Mode = Enum('Mode', ["START", "INIT", "PREP", "TRAIN", "TEST", "QUIT"])
    # Map an enum value to the related function
    loops = {
        Mode.START: None,
        Mode.INIT: __init,
        Mode.PREP: __prep,
        Mode.TRAIN: __train,
        Mode.TEST: __test,
        Mode.QUIT: None
    }

# Program must be run from here
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Must pass one arg for message timeout (in seconds)")
        exit()

    msgQ = mp.Queue()
    fin = sys.stdin.fileno()
    timeout = int(sys.argv[1])
    lineReader = mp.Process(target=messaging.LineReader, args=(msgQ, fin))    
    lineReader.start()

    p = Program(msgQ, timeout)
    try:
        p.start()
    except Exception as e:
        print(e, file=sys.stderr)
    finally:
        lineReader.terminate()
        exit()