ReadMe
===========================

# Directory Structure
```
.
GitHub
├─.idea
│  └─inspectionProfiles
├── Readme.md                                    // help
├─Code                                           // core code
│  ├─evolutionary_operating-weights_agent.py     // EOW module
│  └─main_Prediction.py                          // LSTM prediction module
├─Different trends                               // Contains prediction results and strategy results of 500ETF and stocks with different trends
│  ├─ETF500                                      // 500ETF
│  │  ├─Prediction                               // Prediction results
│  │  │  ├─figure                                // Contains forecast figure
│  │  │  ├─log                                   // Contains log of code execution
│  │  │  ├─model                                 // Contains forecast model
│  │  │  └─record                                // Contains final result(csv file)
│  │  └─Strategy                                 // Strategy results of 500ETF
│  │      ├─1.turtle_agent                       // Contains result of Turtle Agent
│  │      ├─2.moving_average_agent               // Contains result of Moving Average Agent
│  │      ├─3.policy_gradient_agent              // Contains result of Policy Gradient Agent
│  │      ├─4.q_learning_agent                   // Contains result of Q-Learning Agent
│  │      ├─5.evolutionary_operating-weights_agent // Contains result of Evolutionary Operating-weights Agent
│  │      ├─Different Strategy Compare.png       // Figure of different strategy compare
│  │      └─Different Strategy Evaluation.csv    // File of different strategy evaluation
│  ├─falling-000039.XSHE(34)                     // Falling trend stock (The table of contents is the same as above.)
│  │  ├─Prediction
│  │  │  ├─figure
│  │  │  ├─log
│  │  │  ├─model
│  │  │  └─record
│  │  └─Strategy
│  │      ├─1.turtle_agent
│  │      ├─2.moving_average_agent
│  │      ├─3.policy_gradient_agent
│  │      ├─4.q_learning_agent
│  │      ├─5.evolutionary_operating-weights_agent
│  │      ├─Different Strategy Compare.png
│  │      └─Different Strategy Evaluation.csv
│  ├─rising-600348.XSHG(7)                      // Rising trend stock (The table of contents is the same as above.)
│  │  ├─Prediction
│  │  │  ├─figure
│  │  │  ├─log
│  │  │  ├─model
│  │  │  └─record
│  │  └─Strategy
│  │      ├─1.turtle_agent
│  │      ├─2.moving_average_agent
│  │      ├─3.policy_gradient_agent
│  │      ├─4.q_learning_agent
│  │      ├─5.evolutionary_operating-weights_agent
│  │      ├─Different Strategy Compare.png
│  │      └─Different Strategy Evaluation.csv
│  └─volatile-600339.XSHG(6)                     // Volatile trend stock (The table of contents is the same as above.)
│      ├─Prediction
│      │  ├─figure
│      │  ├─log
│      │  ├─model
│      │  └─record
│      └─Strategy
│          ├─1.turtle_agent
│          ├─2.moving_average_agent
│          ├─3.policy_gradient_agent
│          ├─4.q_learning_agent
│          ├─5.evolutionary_operating-weights_agent
│          ├─Different Strategy Compare.png
│          └─Different Strategy Evaluation.csv
├─Evaluation
│  ├─architecture
│    ├─The architecture of EOW algorithm.png
│    ├─The architecture of evolve operating-weights module.png
│    ├─The architecture of system framework.png
│    └─The architecture of the decision module.png
│  ├─600775.XSHG prediction evaluation.csv
│  ├─different model avg evaluation.csv
│  ├─four types prediction evaluation.csv        // Evaluation results of 500ETF and stocks with different trends
│  └─Stock 600775.XSHG different LSTM layer of predict model with 5 and 10 time steps.png
├─Origin                                         // Raw stock data files (divided by number)
├─Prediction                                     // 100 stock prediction results and 500ETF prediction results
│  ├─x-                                          // x represents the stock number, and each folder has the same structure
│  │  └─bestModel                                // best model
│  │      ├─figure                               // Contains forecast figure
│  │      ├─log                                  // Contains log of code execution
│  │      ├─model                                // Contains forecast model
│  │      └─record                               // Contains final result(csv file)
├─ stock information.csv                         // Stock information corresponding to the stock number
└─Strategy                                       // 100 stock strategy results and 500ETF strategy results (contains different strategy)
    ├─1.turtle_agent                             // Contains result of Turtle Agent
    │  ├─x                                       // x represents the stock number, and each folder has the same structure
    │    ├─Accumulated Return.csv                // Result of accumulated return
    │    ├─Accumulated return.png                // Figure of accumulated return
    ├─2.moving_average_agent                     // Contains result of Moving Average Agent
    │  ├─x
    ├─3.policy_gradient_agent                    // Contains result of Policy Gradient Agent
    │  ├─x
    ├─4.q_learning_agent                         // Contains result of Q-Learning Agent
    │  ├─x
    └─5.evolutionary_operating-weights_agent     // Contains result of Evolutionary Operating-weights Agent
        └─x
