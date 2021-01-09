import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
from multiprocessing import current_process

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.get_logger().disabled = True

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Simple container for a GP regression model for a single fidelity
class Model:
    __minValue = None
    __maxValue = None
    dtype=np.float32
    jitter=dtype(1e-3)
    def __init__(self, fidelity, path=""):
        self.ID = str(fidelity)
        self.path = path
        self.indexPoints = []
        self.observations = []
        self.counter = 0
        self.updatesDeferred = 0
        self.plot = False
        self.verbose = False
        self.ignoreUpdate = False
        # Init model params
        self.constrainPos = tfb.Shift(np.finfo(Model.dtype).tiny)(tfb.Exp())
        self.ampVar = self.__makeTV("amplitude")
        self.lengthVar = self.__makeTV("lengthScale")
        self.noiseVar = self.__makeTV("observationNoiseVariance")
        self.trainableVars = [v.trainable_variables[0] for v in 
                [self.ampVar, self.lengthVar, self.noiseVar]]

    # Create model variable. Float32, constrained to be positive
    def __makeTV(self, varName):
        return tfp.util.TransformedVariable(initial_value=1., 
               bijector=self.constrainPos, name=varName, dtype=Model.dtype)
        
    # Set various parameters
    def setParams(self, kernelName, numActions, updateCycle, numTrainingIters):
        self.kernelName = kernelName
        self.numActions = numActions
        self.updateCycle = updateCycle
        self.numIters = numTrainingIters
        self.post_mean = [0.] * numActions
        self.oldMeans = None
        self.post_std = [1.] * numActions
        Model.__minValue = None
        Model.__maxValue = None

    def __kernel(self, amplitude, lengthScale):
        if self.kernelName == "SE":
            return tfk.ExponentiatedQuadratic(amplitude, lengthScale)
        elif self.kernelName == "Matern":
            return tfk.MaternFiveHalves(amplitude, lengthScale)
        else:
            raise Exception("Model - Invalid kernel param \"" + self.kernelName + "\"")

    # Adds a new action-observation pair. Returns whether or not the model was updated
    def add(self, action, observation):
        self.indexPoints.append(action)
        self.observations.append(observation)
        self.__updateBounds(observation)

        # Used during PREP
        if self.ignoreUpdate is False:
            self.counter += 1
            if self.counter % self.updateCycle == 0:
                # If our means are unlikely to have changed, then we will not update
                similarMeans = False
                if self.oldMeans is not None:
                    similarMeans = sum(x + y for x in self.post_mean for y in self.oldMeans) < 0.001
                # Once the model's mostly converged, updates are less useful
                tinyStdevs = sum(self.post_std) < 0.001
                # Defer no more than 10 updates in a row
                mustUpdate = self.updatesDeferred >= 10
                # Update if forced OR if any of the "defer" conditions are false
                if mustUpdate or not (similarMeans and tinyStdevs):
                    self.oldMeans = self.post_mean
                    self.updateModel()
                    self.updatesDeferred = 0
                    return True
                else:
                    self.updatesDeferred += 1
        return False

    # Updates our normalization bounds
    def __updateBounds(self, observation):
        if Model.__minValue is None or Model.__minValue > observation:
            Model.__minValue = observation
        if Model.__maxValue is None or Model.__maxValue < observation:
            Model.__maxValue = observation

    @staticmethod
    def normalize(value):
        return (value - Model.__minValue) * 1. / (Model.__maxValue - Model.__minValue)

    @staticmethod
    def denormalize(value):
        return ( value * (Model.__maxValue - Model.__minValue)) + Model.__minValue

    # Updates our model. Call this directly at end of PREP
    def updateModel(self):
        self.npIndexPoints = np.fromiter(self.indexPoints, dtype=Model.dtype)[..., np.newaxis]
        # Observations are normalized
        if Model.__maxValue < 1.:
            Model.__maxValue = 1.
        self.npObservations = np.fromiter((Model.normalize(x) for x in self.observations), dtype=Model.dtype)
        self.__buildPrior()
        self.__trainModel()
        # Create the posterior GP regression model
        optimizedKernel = self.__kernel(self.ampVar, self.lengthVar)
        predictPoints = np.arange(self.numActions, dtype=Model.dtype)[..., np.newaxis]
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimizedKernel,
            index_points=predictPoints,
            observation_index_points=self.npIndexPoints,
            observations=self.npObservations,
            observation_noise_variance=self.noiseVar,
            predictive_noise_variance=Model.dtype(0.),
            jitter=Model.jitter)
        # Obtains the mean and stdev 
        self.post_mean = gprm.mean()
        self.post_std = gprm.stddev()

    def getMean(self, arm):
        return self.post_mean[arm]

    def getStd(self, arm):
        return self.post_std[arm]

    # Create our joint model
    def __buildPrior(self):        
        def __priorGP(amplitude, lengthScale, observationNoiseVariance):
            return tfd.GaussianProcess(
                kernel = self.__kernel(amplitude, lengthScale),
                index_points = self.npIndexPoints,
                observation_noise_variance=observationNoiseVariance,
                jitter=Model.jitter)

        self.gp = tfd.JointDistributionNamed({
            'amplitude': tfd.LogNormal(loc=0., scale=Model.dtype(1.)),
            'lengthScale': tfd.LogNormal(loc=0., scale=Model.dtype(1.)),
            'observationNoiseVariance': tfd.LogNormal(loc=0., scale=Model.dtype(1.)),
            'observations': __priorGP,
        })

    # Target function used for optimization
    @tf.function(autograph=False, experimental_compile=False)
    def __targetLogProb(self, amplitude, lengthScale, observationNoiseVariance, observations):
        return self.gp.log_prob({
            'amplitude': amplitude,
            'lengthScale': lengthScale,
            'observationNoiseVariance': observationNoiseVariance,
            'observations': observations
        })

    # Train our model parameters
    def __trainModel(self):
        optimizer = tf.optimizers.Adam(learning_rate=.01)
        # Store the likelihood values during training
        #print("\nTRAINING MODEL FOR FIDELITY", self.ID, "-", self.counter, file=sys.stderr)
        lls_ = np.zeros(self.numIters, Model.dtype)
        for i in range(self.numIters):
            with tf.GradientTape() as tape:
                loss = -self.__targetLogProb(self.ampVar, self.lengthVar, self.noiseVar, self.npObservations)
            grads = tape.gradient(loss, self.trainableVars)
            optimizer.apply_gradients(zip(grads, self.trainableVars))
            lls_[i] = loss
            # If the loss isn't meaningfully changing, terminate
            if i < 10 or i % 10 != 0:
                continue
            recentCumChangeInLoss = lls_[i-10:i].sum() - (10*lls_[i-10:i].min(0))
            if recentCumChangeInLoss < 0.0005:
                if self.plot is True:
                    lls_ = lls_[:i]
                #print("\r\nTerminating at", i, "change is", recentCumChangeInLoss, "list is", lls_[i-5:i], file=sys.stderr, flush=True)
                break
        if self.verbose is True:
            print('Trained parameters:', file=sys.stderr)
            print('| amplitude: {}'.format(self.ampVar._value().numpy()), file=sys.stderr)
            print('| lengthScale: {}'.format(self.lengthVar._value().numpy()), file=sys.stderr)
            print('| observationNoiseVariance: {}'.format(self.noiseVar._value().numpy()), file=sys.stderr)
            print('/ loss:', lls_[-1], file=sys.stderr)

        # Plot the loss values
        if self.plot is True:
            from matplotlib import pyplot as plt
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['grid.color'] = '#666666'

            plt.figure(figsize=(12, 4))
            plt.plot(lls_)
            plt.xlabel("Training iteration")
            plt.ylabel("Log marginal likelihood")
            plt.savefig(self.path + "Loss f=" + self.ID+ "-" + str(self.counter) + ".png")

    # Plots the true function as well as the learned one
    def plotPosterior(self, nodes=None, ubFunc=None, trueFunction=None):
        from matplotlib import pyplot as plt
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['grid.color'] = '#666666'

        predictRange = np.linspace(-0.2, self.numActions - 0.8, 200, dtype=Model.dtype)
        predictRange = predictRange[..., np.newaxis]

        optimized_kernel = self.__kernel(self.ampVar, self.lengthVar)
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=predictRange,
            observation_index_points=self.npIndexPoints,
            observations=self.npObservations,
            observation_noise_variance=self.noiseVar,
            predictive_noise_variance=0.)

        numSamples = 50
        samples = gprm.sample(numSamples)
        mean = gprm.mean().numpy()
        stdev = gprm.stddev().numpy()

        plt.figure(figsize=(12, 4))
        if trueFunction is not None:
            plt.plot(predictRange, trueFunction(predictRange),
                label='True fn')
        plt.scatter(self.npIndexPoints[:, 0], self.npObservations, alpha=.1,
                label='Observations')
        for i in range(numSamples):
            plt.plot(predictRange, samples[i, :], c='r', alpha=.1,
                label='Posterior Sample' if i == 0 else None)
        plt.plot(predictRange, mean, c='g',
                label = 'Mean')
        plt.plot(predictRange, mean + 2*stdev, c='c',
                label = '+2 Std')
        if nodes is not None:
            for node in nodes:
                plt.scatter([node.ID], [ubFunc(node)], c='m', label='Upper Bound' if node.ID == 0 else None)
        leg = plt.legend(loc='upper right')
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        plt.xlabel(r"Index points ($\mathbb{R}^1$)")
        plt.ylabel("Observation space")
        plt.savefig(self.path + "Plot f=" + self.ID+ "-" + str(self.counter) + ".png")

    # Converts a list to an np.array with the correct structure
    def __listToNPArray(self, l):
        return np.array(l)
        #return np.array(list(map(lambda x: [x], l)))
