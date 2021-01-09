package multiFidelityMCTS;

public class Configuration {

    //Python script, kept here for convenience
    public String COMMAND = "python3";
    public String SCRIPT = "/home/alex/python/MF-MCTS/src/main.py";

    //Kernel type, either SE or Matern
    public String KERNEL = null;
    //Test metric for selecting the best node
    public String TEST_METRIC = "m";

    //The preparation budget is not deducted from the regular budget
    public int BUDGET = 10000;
    public int PREP_BUDGET = 1000;

    //Rollout depths are used as our different fidelities
    public int[] FIDELITIES = { 5, 10, 25, 50 };
    public double[] COSTS = {5, 10, 25, 50};

    //Should be manually tuned
    public int UPDATE_CYCLE = 50;
    public int NUM_TRAINING_ITERS = 2000;
    public double WIN_REWARD = 100.0;
    public double BASE_REWARD = 0.0;    //To encourage keeping the game going
    public double LOSS_REWARD = -100.0;

    //Default constructor
    public Configuration(String kernel, int budget, int prepBudget, int[] fidelities, double[] costs) {
        KERNEL = kernel;
        BUDGET = budget;
        PREP_BUDGET = prepBudget;
        FIDELITIES = fidelities.clone();
        COSTS = costs.clone();
    }

    //Copy constructor
    public Configuration(Configuration config) {
        new Configuration(config.KERNEL, config.BUDGET, config.PREP_BUDGET, config.FIDELITIES, config.COSTS);
    }
}
