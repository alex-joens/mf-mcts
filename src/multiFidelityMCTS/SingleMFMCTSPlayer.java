package multiFidelityMCTS;

import core.game.StateObservation;
import ontology.Types;

import java.util.Random;

public class SingleMFMCTSPlayer {

    //Random generator
    private Random rnd;

    //Action space
    private StateObservation state;
    private Types.ACTIONS[] actions;

    //Agent configuration
    private Configuration config;

    //MF-MCTS is implemented in Tensorflow in Python
    private PyInterface pyInterface;
    private Boolean initialized = false;

    //Constructor
    public SingleMFMCTSPlayer(Random rnd, Types.ACTIONS[] actions) {
        this.actions = actions;
        this.rnd = rnd;
    }

    //Start the py interface
    public void Configure(Configuration config) {
        this.config = config;
        this.pyInterface = new PyInterface(config.COMMAND, config.SCRIPT);
    }

    //(Re-)Initialize the tree
    public void init(StateObservation so)
    {
        state = so;
        if(!initialized) {
            pyInterface.Initialize(config, actions.length);
        } else {
            pyInterface.Reset(actions.length);
        }
    }

    //Run MF-MCTS and determine which action to take
    public Types.ACTIONS run() {
        //PREP: populate the tree with some initial data
        pyInterface.StartPrep(config.PREP_BUDGET);
        train();
        //TRAIN: self-explanatory
        pyInterface.StartTrain();
        train();
        //TEST: get the best action
        PyInterface.ValuePlusList bestAction = pyInterface.Test(config.TEST_METRIC);
        System.out.println();
        System.out.println("Selecting action " + bestAction.Value + ": "
                           + actions[bestAction.Value].name());
        if(bestAction.List.length > 1) {
            System.out.print("\u001B[34m" + "[");
            for(double value : bestAction.List) {
                System.out.print(" " + value);
            }
            System.out.println(" ]" + "\u001B[0m");
        }
        return actions[bestAction.Value];
    }

    private void train() {
        PyInterface.ValuePlusList request;
        while((request = pyInterface.GetSampleRequest()) != null) {
            SampleResult result = getSample(request, this.state.copy());
            pyInterface.SendSample(result);
        }
    }

    public class SampleResult {
        public double Value;
        public int FirstAction;
        public SampleResult(double value, int firstAction) {
            Value = value;
            FirstAction = firstAction;
        }
    }

    private SampleResult getSample(PyInterface.ValuePlusList request, StateObservation state) {
        //Start by performing the specified action sequence
        for(double action : request.List) {
            state.advance(actions[(int)action]);
        }
        //Perform random rollout from here. Our fidelity is simply the depth of random rollout
        int rolloutDepth = config.FIDELITIES[request.Value];
        for(int i = 0; i < rolloutDepth; i++) {
            //Terminate if the game is over
            if(state.isGameOver())
                break;
            int action = rnd.nextInt(actions.length);
            state.advance(actions[action]);
        }
        //Return the value of our final state
        return new SampleResult(value(state), (int)request.Last());
    }

    //Assign a score to a state
    private double value(StateObservation state) {
        boolean gameOver = state.isGameOver();
        Types.WINNER winner = state.getGameWinner();
        double score = state.getGameScore();
        if(score == 0) {
            score = config.BASE_REWARD;
        }

        //If we won or lost, add an appropriate reward/penalty
        if(gameOver && winner == Types.WINNER.PLAYER_WINS)
            score += config.WIN_REWARD;
        else if(gameOver && winner == Types.WINNER.PLAYER_LOSES)
            score += config.LOSS_REWARD;

        return score;
    }
}
