import time
 
from model import Model
import numpy as np

NUM_TRAINING_POINTS = 500
NUM_ITERS = 500
ONLY_UPDATE_AT_END = True
start = time.time()

def sinusoid(x):
  return np.sin(3 * np.pi * x[..., 0])

def generate_1d_data(num_training_points, observation_noise_variance):
    index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
    index_points_ = index_points_.astype(np.float64)
    # y = f(x) + noise
    observations_ = (sinusoid(index_points_) +
                    np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
    return index_points_, observations_
    
indexPoints, observations = generate_1d_data(NUM_TRAINING_POINTS, 0.1)

# Tests our model system
model = Model()
model.verbose = True
model.plot = True
model.setKernel("SE")
model.setNumActions(2)
model.setNumTrainingIters(NUM_ITERS)
model.setUpdateCycle(100)

# For now, add every point and train at the end
for (action, obs) in zip(indexPoints, observations):
    model.add(action, obs, ignoreUpdate=ONLY_UPDATE_AT_END)

model.updateModel()
model.plotPosterior(sinusoid)

end = time.time()
print("----------")
print("Number of training points:", NUM_TRAINING_POINTS)
print("Training iterations:", NUM_ITERS)
print("Time elapsed in seconds:", (end - start))