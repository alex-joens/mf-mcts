package tracks.singlePlayer;

import java.util.Arrays;
import java.util.Random;

import core.logging.Logger;
import multiFidelityMCTS.Agent;
import multiFidelityMCTS.Configuration;
import tools.Utils;
import tracks.ArcadeMachine;

/**
 * Created with IntelliJ IDEA. User: Diego Date: 04/10/13 Time: 16:29 This is a
 * Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Test {

    public static void main(String[] args) {

		String multiFidelityMCTSController = "multiFidelityMCTS.Agent";

		// Load available games
		String spGamesCollection =  "/home/alex/IdeaProjects/GVGAI/examples/all_games_sp.csv";
		String[][] games = Utils.readGames(spGamesCollection);

		// Game settings
		boolean visuals = true;
		int seed = new Random().nextInt();

		// Game and level to play
		int gameIdx = Integer.parseInt(args[0]);
		int levelIdx = 0; // level names from 0 to 4 (game_lvlN.txt).
		String gameName = games[gameIdx][1];
		String game = games[gameIdx][0];
		String level1 = game.replace(gameName, gameName + "_lvl" + levelIdx);
		String recordActionsFile = gameName + "-" + levelIdx + "-{" + args[1] + ","
				+ args[2] + "," + args[3] + "," + args[4] + "}-" + seed + ".log";

		// Agent configuration
		String kernel = "SE";
		int prepBudget = Integer.parseInt(args[1]);
		int trainingBudget = Integer.parseInt(args[2]);
		//int[] fidelities = {5, 10, 25, 50};
		//double[] costs = {5, 10, 25, 50};
		int[] fidelities = Arrays.stream(args[3].split("-")).mapToInt(Integer::parseInt).toArray();
		double[] costs = Arrays.stream(args[4].split("-")).mapToDouble(Double::parseDouble).toArray();
		Configuration config = new Configuration(kernel, trainingBudget, prepBudget, fidelities, costs);

		// Run a single game
		ArcadeMachine.runMFMCTSGame(game, level1, visuals, multiFidelityMCTSController, config, seed, 0, recordActionsFile);

		// 2. This plays a game in a level by the controller.
		//ArcadeMachine.runOneGame(game, level1, visuals, multiFidelityMCTSController, recordActionsFile, seed, 0);


		// 3. This replays a game from an action file previously recorded
		// String readActionsFile = "boulderchase-0-800637819.log";
		// ArcadeMachine.replayGame(game, level1, visuals, readActionsFile);

		// 4. This plays a single game, in N levels, M times :
//		String level2 = new String(game).replace(gameName, gameName + "_lvl" + 1);
//		int M = 10;
//		for(int i=0; i<games.length; i++){
//			game = games[i][0];
//			gameName = games[i][1];
//			level1 = game.replace(gameName, gameName + "_lvl" + levelIdx);
//			ArcadeMachine.runGames(game, new String[]{level1}, M, sampleMCTSController, null);
//		}

		//5. This plays N games, in the first L levels, M times each. Actions to file optional (set saveActions to true).
//		int N = games.length, L = 2, M = 1;
//		boolean saveActions = false;
//		String[] levels = new String[L];
//		String[] actionFiles = new String[L*M];
//		for(int i = 0; i < N; ++i)
//		{
//			int actionIdx = 0;
//			game = games[i][0];
//			gameName = games[i][1];
//			for(int j = 0; j < L; ++j){
//				levels[j] = game.replace(gameName, gameName + "_lvl" + j);
//				if(saveActions) for(int k = 0; k < M; ++k)
//				actionFiles[actionIdx++] = "actions_game_" + i + "_level_" + j + "_" + k + ".txt";
//			}
//			ArcadeMachine.runGames(game, levels, M, sampleRHEAController, saveActions? actionFiles:null);
//		}


    }
}
