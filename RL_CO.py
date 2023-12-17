import gymnasium
from gymnasium.spaces.box import Box
from gymnasium.spaces.multi_discrete import MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import random
import numpy as np
from CO_Optimizer import Optimizer, grain_flow_path, wheat_supply_path, make_population,\
     make_individual, fitness, vector_from_dataframe_column, precision_round, evaluate_individual


def scaled_sigmoid(score, min_bound=-10, max_bound=10, k_steepness=0.0001, x0_sigmoid_midpoint=0):
    """
    k_steepness: Closer to 0, more gradual approach to min / max is to accommodate for very high values
    """
    sigmoid = max_bound / (1 + np.exp(-k_steepness * (score - x0_sigmoid_midpoint)))
    return (np.abs(max_bound) + np.abs(min_bound)) * (sigmoid / max_bound) + min_bound


def inverse_scaled_sigmoid(sigmoid, min_bound=-10, max_bound=10, k_steepness=0.0001, x0_sigmoid_midpoint=0):
    return x0_sigmoid_midpoint - (1 / k_steepness) * np.log((max_bound - sigmoid) / (sigmoid - min_bound))


def unpack_scaled_sigmoids(sigmoid_vals, min_bound=-10, max_bound=10, k_steepness=0.0001, x0_sigmoid_midpoint=0):
    return [inverse_scaled_sigmoid(sigmoid_val, min_bound=min_bound, max_bound=max_bound, k_steepness=k_steepness,
                                   x0_sigmoid_midpoint=x0_sigmoid_midpoint)
            for sigmoid_val in sigmoid_vals]


def scaled_sigmoid_bin(row_vals, bins):
    normalized_data = [scaled_sigmoid(val, min_bound=-1, max_bound=1) for val in row_vals]
    return np.digitize(normalized_data, np.linspace(-1, 1, bins)), normalized_data


def linear_bin(row_vals, bins):
    return np.digitize(row_vals, np.linspace(np.min(row_vals), np.max(row_vals), bins)),


def bin_data(row_vals, bins=10, bin_algorithm="sigmoid"):
    algorithms = {
        "linear": linear_bin,
        "sigmoid": scaled_sigmoid_bin,
    }

    return algorithms[bin_algorithm](row_vals, bins)


def determine_new_state(individual, max_lots, row_vals, target_val, verbose=False):
    """
    One vector representing the following scores:
     - [combination sums]: Length == # Lots
     - [number of fields per lot]: Length == # Lots
     - # of lots
     - Score of the individual
    """
    unique_lots, lot_counts = np.unique(np.array(individual)[:, 1], return_counts=True)
    combo_sums = []
    for combo_ind in unique_lots:
        combo_sum = 0
        for a in range(len(individual)):
            if individual[a][1] == combo_ind:
                combo_sum += row_vals[a]

        combo_sums.append(combo_sum)

    combo_sums.extend([np.inf for _ in range(max_lots - len(combo_sums))])

    combo_sums.extend(list(lot_counts))
    combo_sums.extend([0 for _ in range(2 * max_lots - len(combo_sums))])

    combo_sums.append(len(unique_lots))
    combo_sums.append(fitness(row_vals, individual, target_val))

    if len(combo_sums) != 2 * max_lots + 2:
        print(f"Combo Sums is not right length: {combo_sums}")
        exit(-1)

    return np.array(combo_sums, dtype=np.float64)


class RLAlgorithm(gymnasium.Env):
    def __init__(self, row_vals, max_lots, target_val=0, bins=10, verbose=False, bin_algorithm="linear"):
        super(RLAlgorithm, self).__init__()

        self.max_lots = max_lots
        self.row_vals = row_vals
        self.target_val = target_val
        self.bins = bins
        self.verbose = verbose

        self.action_space = MultiDiscrete([self.bins, self.max_lots])
        # Combined vectors and scalars into one vector as following:
        # [combo sums] + [combo sizes] + # combos + score
        observation_size = 2 * self.max_lots + 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float64)

        self.individual = make_individual(row_vals, max_lots)
        self.state = determine_new_state(self.individual, self.max_lots, self.row_vals, self.target_val, self.verbose)
        self.binned_data, self.sigmoid_vals = bin_data(row_vals, bin_algorithm=bin_algorithm)
        self.previous_score = 0
        self.last_action = [0, 0]

    def step(self, action):
        bin_num, new_combo_index = action

        valid_inds = [a for a in range(len(self.binned_data)) if self.binned_data[a] == bin_num]
        if len(valid_inds) <= 1: sigmoid_ind = 0
        else: sigmoid_ind = valid_inds[random.randint(0, len(valid_inds) - 1)]

        field_val = inverse_scaled_sigmoid(self.sigmoid_vals[sigmoid_ind], min_bound=-1, max_bound=1)
        field_val_ind = np.argmin([field_val - np.array(self.row_vals)])
        try:
            individual_to_vals = [self.row_vals[field[0]] for field in self.individual]
            field_ind = list(map(precision_round, individual_to_vals)).index(precision_round(self.row_vals[field_val_ind]))

        except ValueError:
            print(f"Field Val Individual: {np.array(self.individual)[:, 0]}")
            print(f"Value to Find:        {self.row_vals[field_val_ind]}")
            exit(-1)

        self.individual[field_ind] = (self.individual[field_ind][0], new_combo_index)

        curr_fitness = fitness(self.row_vals, self.individual, self.target_val)
        if np.array_equal(action, self.last_action): curr_fitness -= 1
        self.last_action = action

        self.previous_score = curr_fitness - self.previous_score
        self.state = determine_new_state(self.individual, self.max_lots, self.row_vals, self.target_val, self.verbose)

        done = False
        truncated = False
        info = {}

        if self.verbose:
            print(f"Chosen Action:      {action}")
            print(f"Current Fitness:    {curr_fitness}")
            print(f"Previous Score:     {self.previous_score}")
            # print(f"Current State:      {self.state}")
            print(f"Current Individual: {self.individual}")

        return self.state, self.previous_score, done, truncated, info

    def reset(self, seed=42):
        random.seed(seed)
        self.individual = make_individual(self.row_vals, self.max_lots)
        self.state = determine_new_state(self.individual, self.max_lots, self.row_vals, self.target_val)
        self.binned_data, self.sigmoid_vals = bin_data(self.row_vals, bin_algorithm="sigmoid")
        self.previous_score = 0

        info = {"seed": seed}

        return self.state, info

    def render(self, mode='console'):
        pass  # Optional


class RLCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RLCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals == 'episode':
            self.episode_count += 1
            print(f"Episode: {self.episode_count}")

        return True


class RLOptimizer(Optimizer):
    def __init__(self, init_dataframe=None, header_row=1, max_column=10):
        super().__init__(algorithm="rl", init_dataframe=None, header_row=1, max_column=10)
        self.default_parameters = {
            "category": "carbon_emissions",
            "row_vals": None,
            "max_lots": 5,
            "target_val": 0,
            "number_bins": 10,
            "learning_rate": 0.00025,
            "max_steps": 100000,
            "multithreaded": True,
            "bin_method": "sigmoid",
            "verbose": False,
            "trained": False,
            "file_name": "rl_co_model",
            "return_default_params": False,
        }

    def __manage_test_data(self):
        try:
            return self.dataframe.T
        except AttributeError:
            return self.dataframe

    def __setup_params(self, hyper_parameters=None):
        if type(hyper_parameters) == dict:
            for key, item in hyper_parameters.items(): self.default_parameters[key] = item

        if self.dataframe is None:
            print("Please call \"import_data()\" before \"optimize_for()\" to load a dataset into the optimizer")
            return

        valid_categories = {
            "carbon_emissions": vector_from_dataframe_column(self.dataframe, col=37),
            "lot_cost": vector_from_dataframe_column(self.dataframe, col=41),
            "test_data": self.__manage_test_data()
        }

        try:
            data = valid_categories[self.default_parameters["category"]]
        except KeyError:
            print(f"Category \"{self.default_parameters['category']}\" is invalid, quitting training...")
            return

        self.default_parameters["row_vals"] = data

        rl_alg = RLAlgorithm(data, self.default_parameters["max_lots"], target_val=self.default_parameters["target_val"],
                             bins=self.default_parameters["number_bins"], bin_algorithm=self.default_parameters["bin_method"],
                             verbose=self.default_parameters["verbose"])

        return rl_alg, RLCallback()
        
    def predict(self, num_steps=100, hyper_parameters=None):
        rl_alg, callback = self.__setup_params(hyper_parameters=hyper_parameters)
        model = PPO.load(self.default_parameters["file_name"])
        observation, info = rl_alg.reset()

        for _ in range(num_steps):
            action, _states = model.predict(observation, deterministic=True)
            new_state, reward, done, truncated, info = rl_alg.step(action)
            rl_alg.state = new_state
            observation = new_state

        if self.default_parameters["verbose"]:
            print(f"Predicted individual: {rl_alg.individual}")

        if self.default_parameters["return_default_params"]: return rl_alg.individual, self.default_parameters
        return rl_alg.individual,

    def optimize_for(self, hyper_parameters=None):
        rl_alg, callback = self.__setup_params(hyper_parameters=hyper_parameters)

        model = PPO("MlpPolicy", rl_alg, learning_rate=self.default_parameters["learning_rate"],
                    verbose=1 if self.default_parameters["verbose"] else 0)
        model.learn(total_timesteps=self.default_parameters["max_steps"], callback=callback)
        model.save(self.default_parameters["file_name"])

        if self.default_parameters["return_default_params"]: return rl_alg.individual, self.default_parameters
        return rl_alg.individual,


def load_predict():
    rl_opt = RLOptimizer(max_column=41)
    # rl_opt.load_test_data(test_size=1000)
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)
    hyper_parameters = {
        "category": "carbon_emissions",
        "max_steps": 2 ** 13,
        "verbose": True,
    }
    best_individual = rl_opt.predict(hyper_parameters=hyper_parameters)
    print(f"**  Best Individual: {evaluate_individual(best_individual, rl_opt.vectorize_column(col=37))}")


def train_six_lots():
    rl_opt = RLOptimizer()
    rl_opt.load_test_data(test_size=200)

    hyper_parameters = {
        "max_lots": 6,
        "max_steps": 2**12,
        "verbose": True,
        "file_name": "rl_six_lots",
        "return_default_params": True,
        "learning_rate": 0.00025,
    }

    best_individual, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    print(f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}")


if __name__ == "__main__":
    # load_predict()
    train_six_lots()
    
