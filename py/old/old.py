import numpy as np
import GPy
import sys
from select import select

# Variables that will be set by the program
models: None
numFidelities: None
numActions: None
updateCycle = 25
timeout = 30    # Seconds before the program should automatically terminate

# Sends msg to host program
def send(e = ""):
    print("k " + e, file = sys.stdout)

# Error, currently prints to stdout
def error(e):
    print("e " + e, file = sys.stdout)

# Converts a list to an np.array with the correct structure
def arr(l):
    return np.array(list(map(lambda x: [x], l)))
    #return np.array(list(map(lambda x: [x, 0], l)), ndmin=2)
    #return np.array(l).T
    # i 2 2 SE 0 0.1 0 0.5 1 0.8 1 0.7


# Simple container for models
class Model:
    def __init__(self, kern):
        self.X = []
        self.Y = []
        if kern == "SE":
            self.K = GPy.kern.RBF(1)
        elif kern == "MATERN":
            self.K = GPy.kern.Matern32(1)
        else:
            error("Unrecognized kernel arg \"" + kern + "\"")
        #Justify this later
        self.K.lengthscale = 0.1
        self.M = None

    # Updates data and periodically updates the model
    def update(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        if self.M is None:
            self.M = GPy.models.GPRegression(arr(self.X), arr(self.Y), self.K)
        else:
            self.M.set_XY(arr(self.X), arr(self.Y))
        # Update periodically
        if len(self.X) % updateCycle == 0:
            self.M.optimize()

    # Returns the mean and variance for all arms
    def data(self):
        d = []
        for i in range(numActions):
            mean, covMatrix = self.M.predict_noiseless(arr([i]))
            d.append(mean[0])
            d.append(covMatrix[0])
        return d          


# Initialization function. Requires initial data for seeding
# int numFidelities, int numActions, string kern, int x0, double y0, int x1, double y1...
def init(args):
    global numFidelities, numActions, models
    numFidelities = int(args[0])
    numActions = int(args[1])
    kern = args[2]
    models = [Model(kern) for i in range(numFidelities)]
    send()

# Add a new query-observation-fidelity tuple
def update(args):
    x = int(args[0])
    y = float(args[1])
    fidelity = int(args[2])
    models[fidelity].update(x, y)
    if fidelity != numFidelities - 1:
        models[-1].update(x, y) 
    send()

# Gets the mean and variance for every arm
def get(args):
    fidelity = int(args[0])
    d = models[fidelity].data()
    send(' '.join(map(str, d)))

# Exits the program
keepLooping = True
def quit(x):
    global keepLooping
    keepLooping = False

def test(x):
    xs = [0, 0, 1, 1, 2]
    ys = [0.1, 0.4, 0.8, 0.7, 0.6]
    print(arr(xs))
    print(arr(ys))
    

# Loops over stdin, reads input as text. Will time out after 30 seconds of inactivity
while(keepLooping):
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        line = sys.stdin.readline().rstrip()
        args = line.split(' ')

        # Map the first arg to a function
        options = {'i': init,    # Initialize things
                'u': update,     # Add a new query-observation-fidelity tuple
                'g': get,        # Get the means and vars for each action for a given fidelity
                'q': quit,       # Quit
                't': test,
        }
        func = options.get(args[0], lambda x: error("Unrecognized command"))
        # Run the function, reporting any problems if they are found
        #try:
        func(args[1:])
        #except:
        #    error("Function failed")
    else:
        error("Timeout")
        keepLooping = False