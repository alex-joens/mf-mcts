/*

package multiFidelityMCTS;

import core.game.StateObservation;
import ontology.Types;
import tools.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class TreeNode {

    //Constants
    protected double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};

    //Configuration
    private Configuration config;

    //Location within tree
    public TreeNode parent;
    public TreeNode[] children;
    public int depth;
    public int childIndex;

    //Value
    public double totalValue;
    public int numVisits;

    //Action space
    Types.ACTIONS[] actions;

    //Etc
    public Random rnd;
    public StateObservation rootState;


    //Root node constructor
    public TreeNode(Random rnd, Types.ACTIONS[] actions, Configuration config ) {
        this(null, -1, rnd, actions, config);
    }

    //Default node constructor
    public TreeNode(TreeNode parent, int childIndex, Random rnd, Types.ACTIONS[] actions, Configuration config) {
        this.parent = parent;
        this.childIndex  = childIndex;
        this.rnd = rnd;
        this.actions = actions;
        this.config = config;
        children = new TreeNode[actions.length];
        totalValue = 0.0;
        if(parent != null)
            depth = parent.depth + 1;
        else
            depth = 0;
    }

    //Search
    public void mctsSearch() {

        //Reset our budget
        config.REMAINING_BUDGET = config.BUDGET;

        try {
            //Search until we can no longer perform any rollouts
            while (config.REMAINING_BUDGET > config.CHEAPEST_FIDELITY) {
                //Select a node to explore, roll out from it, and update the tree
                StateObservation state = rootState.copy();
                TreeNode selected = treePolicy(state);
                double delta = selected.rollOut(state);
                backUp(selected, delta);
            }
        } catch(Exception e) {
            e.printStackTrace();
        }
    }


/*
    .-------------------------.
    | SELECTION and EXPANSION |
    '-------------------------'
*

    //Select a node to explore
    public TreeNode treePolicy(StateObservation state) {

        TreeNode current = this;

        //Find the first unexpanded node that isn't terminal or too deep
        while(!state.isGameOver() && current.depth < config.MAX_TREE_DEPTH)
        {
            if(current.notFullyExpanded())
                return current.expand(state);
            else
                current = current.uct(state);
        }

        return current;
    }

    //Expand a node that hasn't been fully explored
    public TreeNode expand(StateObservation state) {

        int bestAction = 0;
        double randomSelector = -1;

        //Randomly pick an unexplored action
        for(int i = 0; i < children.length; i++)
        {
            double x = rnd.nextDouble();
            if(x > randomSelector && children[i] == null) {
                bestAction = i;
                randomSelector = x;
            }
        }

        //Advance to this state
        state.advance(actions[bestAction]);

        //Create a new node for this and return it
        TreeNode node = new TreeNode(this, bestAction, rnd, actions, config);
        children[bestAction] = node;
        return node;
    }

    //Use the UCT rule to pick the best child node
    public TreeNode uct(StateObservation state) {

        TreeNode selected = null;
        double bestValue = -Double.MAX_VALUE;

        for(TreeNode child : children)
        {
            //Get the child's current normalized value
            double childValue = child.totalValue / (child.numVisits + config.EPSILON);
            childValue = Utils.normalise(childValue, bounds[0], bounds[1]);

            //Get the UCT value, then add a small amount of noise, to help break ties
            double uctValue = childValue +
                    config.EXPLORATION_TRADEOFF * Math.sqrt(Math.log( (numVisits + 1) / (child.numVisits + config.EPSILON) ));
            uctValue = Utils.noise(uctValue, config.EPSILON, rnd.nextDouble());

            //Pick this node if it is the most promising
            if(uctValue > bestValue) {
                selected = child;
                bestValue = uctValue;
            }
        }

        //Should never happen
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Advance the state, then return it
        state.advance(actions[selected.childIndex]);
        return selected;
    }


/*
    .------------.
    | SIMULATION |
    '------------'
*

    //Roll out from a state and return a score
    public double rollOut(StateObservation state) throws Exception {

        //Failsafe
        if(config.CHEAPEST_FIDELITY > config.BUDGET)
            throw new Exception("ERROR: exceeded budget");

        //Pick a fidelity
        int rolloutDepth = config.CHEAPEST_FIDELITY;
        //Multi-fidelity mode: select the best fidelity based on things
        if(config.MULTI_FIDELITY) {
            //TODO: implement
        }
        //Single-fidelity mode: use most expensive fidelity we can
        else {
            if(config.REMAINING_BUDGET >= config.MOST_EXPENSIVE_FIDELITY)
                rolloutDepth = config.MOST_EXPENSIVE_FIDELITY;
            else {
                for(int fidelity : config.FIDELITIES) {
                    if(fidelity <= config.REMAINING_BUDGET && fidelity > rolloutDepth)
                        rolloutDepth = fidelity;
                }
            }
        }

        //Consume our budget
        config.REMAINING_BUDGET -= rolloutDepth;

        //Random rollout
        for(int i = depth; i < depth + rolloutDepth; i++) {

            //Terminate if we're too deep or the game is over
            if((i > config.MAX_TREE_DEPTH )|| (state.isGameOver()))
                break;
            int action = rnd.nextInt(actions.length);
            state.advance(actions[action]);
        }

        //Return the value of our final state
        return value(state);
    }

    //Assign a score to a state
    public double value(StateObservation state) {
        boolean gameOver = state.isGameOver();
        Types.WINNER winner = state.getGameWinner();
        double score = state.getGameScore();

        //If we won or lost, add an appropriate reward/penalty
        if(gameOver && winner == Types.WINNER.PLAYER_WINS)
            score += config.WIN_REWARD;
        else if(gameOver && winner == Types.WINNER.PLAYER_LOSES)
            score += config.LOSS_REWARD;

        //Update our bounds
        updateBounds(score);

        return score;
    }


/*
    .------------------.
    | BACK-PROPAGATION |
    '------------------'
*

    //Back up the score
    public void backUp(TreeNode node, double result) {

        TreeNode n = node;
        while(n != null) {
            n.numVisits++;
            n.totalValue += result;
            n.updateBounds(result);
            n = n.parent;
        }
    }


/*
    .-----------------.
    | TREE EVALUATION |
    '-----------------'
*

    private enum actionTypes { MOST_VISITED, HIGHEST_SCORE }

    //Most visited action
    public Types.ACTIONS mostVisitedAction() {
        return getAction(actionTypes.MOST_VISITED);
    }

    //Best action
    public Types.ACTIONS highestScoringAction() {
        return getAction(actionTypes.HIGHEST_SCORE);
    }

    //Action selection works the same, just with different child values
    private Types.ACTIONS getAction(actionTypes actionType) {

        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for(int i = 0; i < children.length; i++) {

            System.out.println("| Checking action " + actions[i]);
            if(children[i] == null)
                continue;

            //Get the child value
            double childValue = 0;
            switch(actionType) {
                case MOST_VISITED:
                    childValue = children[i].numVisits; break;
                case HIGHEST_SCORE:
                    childValue = children[i].totalValue / (children[i].numVisits + config.EPSILON); break;
            }
            //Add some noise, for breaking ties
            childValue = Utils.noise(childValue, config.EPSILON, rnd.nextDouble());
            System.out.println("| Value is " + childValue);

            //Select this child if it is the best
            if(childValue > bestValue) {
                selected = i;
                bestValue = childValue;
            }
        }

        System.out.println("/ Selected action " + actions[selected] + ", score " + bestValue);

        return actions[selected];
    }


/*
    .-----------.
    | UTILITIES |
    '-----------'
*

    //Check if a node is fully expanded
    public boolean notFullyExpanded() {
        for (TreeNode node : children) {
            if(node == null)
                return true;
        }
        return false;
    }

    //Update bounds
    public void updateBounds(double value) {
        if(value < bounds[0])
            bounds[0] = value;
        if(value > bounds[1])
            bounds[1] = value;
    }
}
*/