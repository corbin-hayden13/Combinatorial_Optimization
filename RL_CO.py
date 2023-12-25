import gymnasium
from gymnasium.spaces.box import Box
from gymnasium.spaces.multi_discrete import MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from scipy.stats import skew
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from CO_Optimizer import Optimizer, grain_flow_path, wheat_supply_path,\
     make_individual, fitness, vector_from_dataframe_column, precision_round, evaluate_individual, scaled_sigmoid


def inverse_scaled_sigmoid(sigmoid, min_bound=-10, max_bound=10, k_steepness=0.0001, x0_sigmoid_midpoint=0):
    return x0_sigmoid_midpoint - (1 / k_steepness) * np.log((max_bound - sigmoid) / (sigmoid - min_bound))


def unpack_scaled_sigmoids(sigmoid_vals, min_bound=-10, max_bound=10, k_steepness=0.0001, x0_sigmoid_midpoint=0):
    return [inverse_scaled_sigmoid(sigmoid_val, min_bound=min_bound, max_bound=max_bound, k_steepness=k_steepness,
                                   x0_sigmoid_midpoint=x0_sigmoid_midpoint)
            for sigmoid_val in sigmoid_vals]


def scaled_sigmoid_bin(row_vals, bins):
    normalized_data = [scaled_sigmoid(val, min_bound=-1, max_bound=1) for val in row_vals]
    return np.digitize(normalized_data, np.linspace(-1, 1, bins - 1)) - 1, normalized_data


def linear_bin(row_vals, bins):
    return np.digitize(row_vals, np.linspace(np.min(row_vals), np.max(row_vals), bins - 1)) - 1,


def bin_data(row_vals, bins=10, bin_algorithm="sigmoid"):
    algorithms = {
        "linear": linear_bin,
        "sigmoid": scaled_sigmoid_bin,
    }

    return algorithms[bin_algorithm](row_vals, bins)


def determine_new_state(individual, row_vals, target_val, target_size, verbose=False):
    """
    One vector representing the following scores:
     - # of lots
     - Score of the individual
     - Variance of combo sums
     - Standard Deviation of combo sums
     - Average distance of sums from target value
     - Skewness of combination sums
     - Maximum sum
     - Minimum sum
     - Count of positive sums
     - Count of negative sums
     - (Deprecated) [number of fields per lot]: Length == # Lots
     - (Deprecated) [combination sums]: Length == # Lots
    """
    lot_sums = [sums for _, sums in evaluate_individual(individual, row_vals).items()]
    unique_lots, lot_counts = np.unique(np.array(individual)[:, 1], return_counts=True)
    normalized_state = np.array([len(unique_lots), fitness(row_vals, individual, target_val),
                                 np.var(lot_sums), np.std(lot_sums),
                                 np.mean([target_val - lot_sum for lot_sum in lot_sums]),
                                 skew(lot_sums), np.max(lot_sums), np.min(lot_sums),
                                 sum([1 for lot_sum in lot_sums if lot_sum > 0]),
                                 sum([1 for lot_sum in lot_sums if lot_sum <= 0])])

    normalized_state = (normalized_state - np.mean(normalized_state)) / np.var(normalized_state)

    # min_data = min(normalized_state)
    # max_data = max(normalized_state)
    # normalized_state = (np.array(normalized_state) - min_data) / (max_data - min_data + 1e-6)  # Avoid x / 0

    if len(normalized_state) != target_size:
        print(f"Combo Sums {len(normalized_state)} != {target_size}: {normalized_state}")
        exit(-1)

    return np.array(normalized_state, dtype=np.float64)


class CustomNeuralNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomNeuralNetwork, self).__init__(observation_space, features_dim)

        # Example of a more complex network
        self.layer1 = nn.Linear(observation_space.shape[0], 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, features_dim)

    def forward(self, observations):
        x = F.relu(self.layer1(observations))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class RLAlgorithm(gymnasium.Env):
    def __init__(self, row_vals, max_lots, target_val=0, bins=10, verbose=False, ga_optimizer=None,
                 bin_algorithm="linear"):
        super(RLAlgorithm, self).__init__()

        self.max_lots = max_lots
        self.row_vals = row_vals
        self.target_val = target_val
        self.bins = bins
        self.verbose = verbose
        self.individual = make_individual(row_vals, max_lots)
        if ga_optimizer is None:
            self.ga_best_individual = None

        else:
            self.ga_best_individual = ga_optimizer.optimize_from(individual=self.individual, hyper_parameters={
                "row_vals": np.array(row_vals),
                "max_lots": max_lots,
                "target_val": target_val,
                "number_generations": 350,
                "individual_mutation_chance": 0.3,
                "gene_mutation_chance": 0.2,
                "verbose": True,
            })

        self.action_space = MultiDiscrete([self.bins, self.max_lots])
        # See determine_new_state() comments for feature description
        self.observation_size = 10
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float64)

        self.state = determine_new_state(self.individual, self.row_vals, self.target_val, self.observation_size,
                                         verbose=self.verbose)
        self.binned_data, self.sigmoid_vals = bin_data(row_vals, bin_algorithm=bin_algorithm)
        self.previous_score = 0
        self.last_action = [0, 0]

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        return [seed]

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

        if self.ga_best_individual is None: curr_fitness = fitness(self.row_vals, self.individual, self.target_val)
        else:
            curr_fitness = fitness(self.row_vals, self.individual, self.target_val) - \
                           fitness(self.row_vals, self.ga_best_individual, self.target_val)

        if np.array_equal(action, self.last_action): curr_fitness -= 30
        self.last_action = action

        self.state = determine_new_state(self.individual, self.row_vals, self.target_val, self.observation_size,
                                         verbose=self.verbose)

        done = False
        truncated = False
        info = {}

        if self.verbose:
            # print(f"Chosen Action:      {action}")
            print(f"Current Fitness:    {curr_fitness}")
            # print(f"Previous Score:     {self.previous_score}")
            # print(f"Current State:      {self.state}")
            # print(f"Current Individual: {self.individual}")

        return self.state, curr_fitness, done, truncated, info

    def reset(self, seed=42):
        random.seed(seed)
        self.individual = make_individual(self.row_vals, self.max_lots)
        self.state = determine_new_state(self.individual, self.row_vals, self.target_val, self.observation_size,
                                         verbose=self.verbose)
        self.binned_data, self.sigmoid_vals = bin_data(self.row_vals, bin_algorithm="sigmoid")
        self.previous_score = 0

        info = {"seed": seed}

        return self.state, info

    def render(self, mode='console'):
        pass  # Optional


def make_env(env_rank, hyper_parameters, env_seed=42):
    hyper_parameters["policy_kwargs"] = dict(
        features_extractor_class=CustomNeuralNetwork,
        features_extractor_kwargs=dict(features_dim=128)  # This should match the last layer of your network
    )

    def _init():
        env = RLAlgorithm(hyper_parameters["row_vals"], hyper_parameters["max_lots"],
                          target_val=hyper_parameters["target_val"], bins=hyper_parameters["number_bins"],
                          bin_algorithm=hyper_parameters["bin_method"], verbose=hyper_parameters["verbose"],
                          ga_optimizer=hyper_parameters["ga_optimizer"])
        env.seed(env_seed + env_rank)
        return env

    return _init


def vectorize_envs(hyper_parameters, cores=8):
    env_inits = [make_env(a, hyper_parameters) for a in range(cores)]
    envs = [_init() for _init in env_inits]
    return SubprocVecEnv(env_inits), envs


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
            "num_environments": 8,
            "bin_method": "sigmoid",
            "verbose": False,
            "print_policy": False,
            "trained": False,
            "file_name": "rl_co_model",
            "tensorboard_log": "./rl_co_model_tb_log/",
            "return_default_params": False,
            "ga_optimizer": None,
            # PPO best practices: https://github.com/EmbersArc/PPO/blob/master/best-practices-ppo.md
            "batch_size": 128,  # Typical Discrete [32, 512]
            "learning_rate": 2.5e-4,  # Typical [1e-5, 1e-3]
            "num_epochs": 7,  # Typical [3, 10]
            "max_steps": 5e6,  # Typical [5e5, 1e7]
            "entropy": 0.0,  # ent_coef, increase -> incentivizes diverse actions
            "epsilon": 0.2,  # clip_range, increase -> more drastic changes per update
            "gae_lambda": 0.15,  # 0 - 1 as current or future reward prioritized
            "normalize_advantage": False,
            "policy_kwargs": None,
        }

    def __compare_to_ga_reward(self):
        pass

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

        if self.default_parameters["num_environments"] > 1:
            vectorized_envs, env_objs = vectorize_envs(self.default_parameters,
                                                       cores=self.default_parameters["num_environments"])
        else:
            vectorized_envs = make_env(0, self.default_parameters)
            env_objs = vectorized_envs()

        return vectorized_envs, env_objs, RLCallback()

    def predict(self, num_steps=100, hyper_parameters=None):
        hyper_parameters["num_environments"] = 1
        _, rl_alg, callback = self.__setup_params(hyper_parameters=hyper_parameters)
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
        vectorized_envs, env_objs, callback = self.__setup_params(hyper_parameters=hyper_parameters)

        model = PPO("MlpPolicy", vectorized_envs, verbose=1,
                    batch_size=self.default_parameters["batch_size"], n_epochs=self.default_parameters["num_epochs"],
                    ent_coef=self.default_parameters["entropy"], clip_range=self.default_parameters["epsilon"],
                    learning_rate=self.default_parameters["learning_rate"], gae_lambda=self.default_parameters["gae_lambda"],
                    normalize_advantage=self.default_parameters["normalize_advantage"],
                    policy_kwargs=self.default_parameters["policy_kwargs"],
                    tensorboard_log=self.default_parameters["tensorboard_log"])
        if self.default_parameters["print_policy"]: print(model.policy)
        model.learn(total_timesteps=self.default_parameters["max_steps"], callback=callback)
        model.save(self.default_parameters["file_name"])

        if self.default_parameters["return_default_params"]:
            return [env.individual for env in env_objs], self.default_parameters
        else: [env.individual for env in env_objs],


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
    
