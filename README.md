mf-mfts
====

MF-MCTS (Multi-Fidelity Monte Carlo Tree Search) is a general approach for taking advantage of multiple simulation options, of varying cost and quality, with MCTS.  This is a branch of GVGAI's single-player planning track, which is found here: https://github.com/GAIGResearch/GVGAI

The particular algorithm I tested uses GP regression, which is most easily done in TensorFlow. Thus, the models and search tree are stored in a separate Python program, which communicates with GVGAI to request samples.

### Setup

Please refer to the GVGAI documentation for setup. I had the best luck with IntelliJ IDEA, which allowed me to import the entire thing as a project.

The Python code requires Python3, `tensorflow`, and `tensorflow_probability` - please refer to their respective installation instructions. I set up `tensorflow-gpu`, but this can be a righteous pain. I included a test script at `py/old/tftest.py` for verifying successful TensorFlow installation.

Optionally, install `matplotlib` and set `Model.plot = True` in `model.py`. This will plot the learning behavior, which is very useful for making sure everything works. 

To run, create a Run/Debug Configuration targeting `src/main/java/tracks/singlePlayer/Test.java`. If you want to compile a JAR file, target this same file. Note that the program expects certain command-line arguments.

### Configuration

There are several parameters that have to be manually configured.

`singlePlayer/Test.java`: Line 23 is the path to the full list of games.

`Configuration.java`: Some learning parameters and the path to the Python script. Some of these are overridden by command line args.

`main.py`: (Optional) Set `path` to the folder where model plots will be saved.

`model.py`: Controls tensorflow-gpu and model training behavior
- Set `dtype` and `jitter` to meet hardware needs. Consumer-grade GPUs have relatively strong single-point performance and weak double-point, so use np.float32 and 1e-3. CPUs and workstation GPUs can use np.float64 and 1e-6 with little performance penalty, though it's likely overkill.
- Model training is by far the greatest bottleneck, so it is heavily optimized. Each fidelity's model is only checked once every `updateCycle` (default 25) times a new data point is added. It will defer an update if model has mostly converged and the means haven't changed substantially. A maximum of `maxDeferUpdates` (default 10) updates can be deferred this way.

### Operation

You can run `singlePlayer/Test.java` directly from an IDE, or 

### Modification

The relevant Java files are stored in `src/multiFidelityMCTS`. The `main()` function is stored in `src/main/java/tracks/singlePlayer/Test.java`.

The Python files are stored in, you guessed it, `py`. You will most likely want to change `mf_tree.py`.

This system is not optimized for mass testing, due to the significant memory overhead of both Java and TensorFlow (about 1.5GB per program instance). I was partway through implementing parallel experimentation, but it is not fully functional so I uploaded an older, more stable build.

When implementing multithreading, the only oddity was that `ArcadeMachine.runMFMCTSGame()` will break unless you wrap pretty much the entire function body in `synchronized()`. Give each thread an ID number. Update `PyInterface.java` to use static variables to store the program, and modify the message passing system slightly to include an ID number in every message.

