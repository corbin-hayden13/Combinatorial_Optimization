import numpy as np

from RL_CO import RLOptimizer, RLAlgorithm
from CO_Optimizer import evaluate_individual, fitness, greatest_closest_power, wheat_supply_path


def train_six_lots():
    rl_opt = RLOptimizer()
    # rl_opt.load_test_data(test_size=100, min_bound=-20000, max_bound=11000, percent_negative=0.35)
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)

    hyper_parameters = {
        "max_lots": 6,
        "max_steps": 6.5e5,
        "entropy": 0.021,
        "epsilon": 0.3,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "verbose": False,
        "num_environments": 16,
        "file_name": "rl_six_lots_actual",
        "return_default_params": True,
    }

    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    best_individual = best_individuals[np.argmax([fitness(params["row_vals"], individual, params["target_val"])
                                                  for individual in best_individuals])]
    print(f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}")


def train_n_lots():
    test_size = 50

    rl_opt = RLOptimizer()
    rl_opt.load_test_data(test_size=test_size, min_bound=-22000, max_bound=11000, percent_negative=0.3)
    # rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)

    hyper_parameters = {
        "max_lots": 11,
        "max_steps": 1e7,
        "entropy": 0.03,
        "epsilon": 0.225,
        "batch_size": greatest_closest_power(test_size),
        "learning_rate": 2.5e-4,
        "verbose": False,
        "num_environments": 16,
        "file_name": "rl_n_lots_realistic_distribution",
        "return_default_params": True,
    }

    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    best_individual = best_individuals[np.argmax([fitness(params["row_vals"], individual, params["target_val"])
                                                  for individual in best_individuals])]
    print(f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}")


if __name__ == "__main__":
    train_n_lots()

