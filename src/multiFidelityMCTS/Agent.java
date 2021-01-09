package multiFidelityMCTS;

import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

public class Agent extends AbstractPlayer {

    public Configuration config;

    //Action space
    public Types.ACTIONS[] actions;

    //Player
    protected SingleMFMCTSPlayer mctsPlayer;

    //Constructor
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        //Get the actions in a static array
        ArrayList<Types.ACTIONS> act = so.getAvailableActions();
        actions = new Types.ACTIONS[act.size()+1];
        for(int i = 0; i < act.size(); i++)
        {
            actions[i] = act.get(i);
        }
        actions[actions.length-1] = Types.ACTIONS.ACTION_NIL;

        //Create the player
        mctsPlayer = getPlayer(so, elapsedTimer);
    }

    public void Configure(Configuration config) {
        this.config = new Configuration(config);
        mctsPlayer.Configure(config);
    }

    //Gets a player instance. Starts a game as well
    public SingleMFMCTSPlayer getPlayer(StateObservation so, ElapsedCpuTimer elapsedTimer) {
        SingleMFMCTSPlayer player = new SingleMFMCTSPlayer(new Random(), actions);
        if(config != null) {
            player.Configure(config);
        }
        return player;
    }

    //Picks an action
    public Types.ACTIONS act(StateObservation so, ElapsedCpuTimer elapsedTimer) {
        //Start our search from the current state observation
        mctsPlayer.init(so);

        //Determine what action to perform, then return it
        return mctsPlayer.run();
    }
}
