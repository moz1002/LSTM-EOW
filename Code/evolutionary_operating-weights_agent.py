import numpy as np
import math
import types

class Config:
    strategy_name = "Evolutionary Operating-weights Strategy"  # Current model name

    file_save_path = ""
    logger_name = ""
    figure_save_path = file_save_path + "figure/"
    log_save_path = file_save_path + "log/"

    do_log_print_to_screen = True
    do_log_save_to_file = True

    # Strategy-related operation data
    initial_money = 100000    # Initial money
    window_size = 15          # Window size
    sigma = 0.1               # Disturbance coefficient
    population_size = 15      # Population size
    learning_rate = 0.03      # Learning rate
    bin = 1                   # Bin rate

    skip = 1
    input_size = window_size  # Window_size
    layer_size = 500          # Hidden layer
    output_size = 3           # Action_size
    iterations = 300          # Training epoch
    checkpoint = 100          # Interval printing

def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split('.')[0]
        elif isinstance(val, type):
            name = val.__module__.split('.')[0]
        poorly_named_packages = {'PIL': 'Pillow', 'sklearn': 'scikit-learn'}
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]
        yield name

class Deep_Evolution_Strategy:
    inputs = None
    def __init__(
        self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, children):
        weights_population = []
        for index, x_i in enumerate(children):
            jittered = self.sigma * x_i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch = 100):
        for i in range(epoch):
            population = []

            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)

            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )

class Model:
    def __init__(self, config):
        self.input_size = config.input_size
        self.layer_size = config.layer_size
        self.output_size = config.output_size

        self.weights = [
            np.random.randn(self.input_size, self.layer_size),
            np.random.randn(self.layer_size, self.output_size),
            np.random.randn(1, self.layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:
    def __init__(self, config, model, trend):
        self.model = model
        self.window_size = config.window_size
        self.half_window = config.window_size // 2
        self.trend = trend
        self.skip = config.skip
        self.initial_money = config.initial_money
        self.bin = config.bin

        self.layer_size = config.layer_size
        self.output_size = config.output_size
        self.iterations = config.iterations
        self.checkpoint = config.checkpoint

        self.population_size = config.population_size
        self.sigma = config.sigma
        self.learning_rate = config.learning_rate

        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.population_size,
            self.sigma,
            self.learning_rate
        )

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return np.argmax(decision[0])

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d: t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0: t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def get_reward(self, weights):
        self.model.weights = weights
        state = self.get_state(0)

        hold_money = self.initial_money     # Current amount held
        hold_share = 0                      # The number of shares currently held
        holding_money = []                  # Record of the money holding
        total_assets = []                   # Total assets = money held + stock assets
        for t in range(0, len(self.trend)-1, self.skip):
            action = self.act(state)
            if action == 1 and hold_money >= self.trend[t]:
                share = math.floor((hold_money * self.bin) / self.trend[t])
                if share > 0:
                    hold_share += share
                    hold_money -= share * self.trend[t]
                    holding_money.append(hold_money)
            elif action == 2:
                hold_money += self.trend[t] * hold_share
                hold_share = 0
            state = self.get_state(t + 1)

        total_assets.append(hold_money + self.trend[-1] * hold_share)
        invest = ((total_assets[-1] - self.initial_money) / self.initial_money)
        return invest

    def fit(self):
        self.es.train(self.iterations)

    def buy(self, close, logger):
        state = self.get_state(0)

        hold_money = self.initial_money     # Current amount held
        hold_share = 0                      # The number of shares currently held

        states_action = []                  # Record the action
        states_operation = []               # Record the actual operation

        inventory = []                      # Record the price of each purchase
        holding_money = []                  # Record the holding money
        total_assets = []                   # Total assets = money held + stock assets

        holding_money.append(self.initial_money)

        for t in range(0, len(close), self.skip):
            action = self.act(state)
            states_action.append(action)

            if action == 1 and self.initial_money >= close[t]:
                share = math.floor((hold_money * self.bin) / close[t])
                hold_share += share
                hold_money -= share * close[t]
                if share > 0:
                    inventory.append(close[t])
                    states_operation.append(1)
                else:
                    states_operation.append(0)

            elif action == 2 and len(inventory):
                if sum(inventory)/len(inventory) < close[t]:
                    # The current price is higher than the purchase, then liquidate
                    states_operation.append(2)
                    hold_money += close[t] * hold_share
                    invest = ((hold_money - self.initial_money) / self.initial_money)
                    hold_share = 0
                    inventory = []
                else:
                    states_operation.append(0)
            else:
                states_operation.append(0)

            total_assets.append(hold_money + close[t] * hold_share)
            if t != len(close)-1:
                # When it comes to the last day, don't judge the next state
                state = self.get_state(t + 1)

        total_gains = total_assets[-1] - self.initial_money # Total return
        invest = (total_gains / self.initial_money)         # Total return rate
        logger.info('The last day, the total_assets is %f, the invest is %f %%'% (total_assets[-1], invest*100))
        return invest, total_gains, total_assets, states_action, states_operation
