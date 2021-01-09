package multiFidelityMCTS;

import java.io.*;

public class PyInterface {

    //Both SampleRequests and Action suggestions use this
    public class ValuePlusList {
        public int Value;
        public double[] List;
        public ValuePlusList(String[] params, int startIndex) {
            Value = Integer.parseInt(params[startIndex]);
            List = new double[params.length - 1 - startIndex];
            for(int i = 0; i < List.length; i++) {
                List[i] = Double.parseDouble(params[i+startIndex+1]);
            }
        }
        public double Last() {
            return List[List.length - 1];
        }
    }

    private static int timeout = 15; //in seconds
    private Process process = null;
    private BufferedReader reader = null;
    private BufferedWriter writer = null;

    private String _kernel;
    private int _numActions, _numFidelities, _updateCycle, _numTrainingIters, _budget;
    private double[] _costs;

    public PyInterface(String command, String script) {
        process = null;
        try {
            ProcessBuilder pb = new ProcessBuilder(command, script, String.valueOf(timeout));
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);
            process = pb.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
        reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
        // Wait until it says it's ready
        ok();
    }

    private void send(String cmd, Object... args) {
        StringBuilder sb = new StringBuilder();
        for(Object o : args) {
            sb.append(String.valueOf(o));
            sb.append(' ');
        }
        String msg = (cmd + ' ' + sb.toString()).trim();
        //System.out.println("Writing: \"" + msg + "\"");
        try {
            writer.write(msg);
            writer.write(System.getProperty("line.separator"));
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String read() {
        try {
            String line = reader.readLine().trim();
            //System.out.println("Read: \"" + line + "\"");
            return line;
        } catch(Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void ok() {
        String str = read();
        if(!str.equals("K")) {
            System.err.println("Error: not ok, instead \"" + str + "\"");
        }
    }

    //Set up the tree, models, etc
    public void Initialize(Configuration config, int numActions) { //String kernel, int numActions, int numFidelities, int updateCycle, int numTrainingIters, int budget, float[] costs) {
        _kernel = config.KERNEL;
        _numActions = numActions;
        _numFidelities = config.FIDELITIES.length;
        _updateCycle = config.UPDATE_CYCLE;
        _numTrainingIters = config.NUM_TRAINING_ITERS;
        _budget = config.BUDGET;
        _costs = config.COSTS;
        send("!", "INIT");
        ok();
        send("T", _kernel, _numActions, _numFidelities, _updateCycle, _numTrainingIters);
        ok();
        StringBuilder sb = new StringBuilder();
        for(double d : _costs) {
            sb.append(String.valueOf(d));
            sb.append(' ');
        }
        send("B", _budget, sb.toString().trim());
        ok();
    }

    //Reset the tree, models, etc
    public void Reset(int numActions) {
        send("!", "INIT");
        ok();
        send("R", numActions);
        ok();
    }

    //MCTS works best with some initial training data
    public void StartPrep(int trainingBudget) {
        System.out.println();
        System.out.println("Start preparation");
        send("!", "PREP");
        ok();
        send("S", trainingBudget);
    }

    //Works very similarly to StartPrep from our end
    public void StartTrain() {
        System.out.println();
        System.out.println("Start training");
        send("!", "TRAIN");
        ok();
        send("S");  //Start the training process
    }

    //MF-MCTS has requested a sample. Returns null if it's done requesting samples (because of an exhausted budget)
    public ValuePlusList GetSampleRequest() {
        String msg = read();
        String[] req = msg.split(" ");
        if(req[0].equals("K")) {
            return null;
        } else if(!req[0].equals("S")) {
            System.err.println("Unexpected message in GetSampleRequest: \"" + msg + "\"");
            return null;
        }
        int fidelity = Integer.parseInt(req[1]);
        char fmap[] = {'.', ':', '-', '='};
        System.out.print(fmap[fidelity]);
        return new ValuePlusList(req, 1);
    }

    //First action == last entry in the action sequence. This is an artifact of an old design
    public void SendSample(SingleMFMCTSPlayer.SampleResult result) {
        send("D", result.Value);
    }

    //Fetch the best action. Metrics:
    //  m   Highest mean
    //  v   Most visited by highest fidelity
    //  w   Most visited by any fidelity
    public ValuePlusList Test(String metric) {
        System.out.println();
        System.out.println("Start testing");
        send("!", "TEST");
        ok();
        send("T", metric);
        String msg = read();
        String[] response = msg.split(" ");
        if(!response[0].equals("A")) {
            System.err.println("Unexpected response in Test: \"" + msg + "\"");
            return null;
        }
        return new ValuePlusList(response, 1);
    }

    //Shut it down like Madagascar
    public void Quit() {
        send("!", "QUIT");
    }
}
