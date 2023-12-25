import numpy as np

from RL_CO import RLOptimizer, RLAlgorithm
from GA_CO import GAOptimizer
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
    test_size = 16

    rl_opt = RLOptimizer()
    # rl_opt.load_test_data(test_size=test_size, min_bound=-25000, max_bound=25000, percent_negative=0.5)
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)

    hyper_parameters = {
        "max_lots": 11,
        "max_steps": 5e5,
        "entropy": 0.965,
        "epsilon": 0.375,  # This is clip_range
        "batch_size": greatest_closest_power(test_size),
        "learning_rate": 6.25e-3,
        "gae_lambda": 0.8,  # 0 - 1 as current or future reward prioritized
        "normalize_advantage": True,
        "verbose": False,
        "num_environments": 16,
        "file_name": "rl_n_lots_binned_state",
        "tensorboard_log": "./rl_n_lots_binned_state/",
        "return_default_params": True,
        "use_binned_state": True,
    }

    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    best_individual = best_individuals[np.argmax([fitness(params["row_vals"], individual, params["target_val"])
                                                  for individual in best_individuals])]
    print(f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}")
    print(f"** Final Fitness: {fitness(params['row_vals'], best_individual, params['target_val'])}")


def best_individual_str(best_individuals, params):
    best_individual = best_individuals[np.argmax([fitness(params["row_vals"], individual, params["target_val"])
                                                  for individual in best_individuals])]

    return f"Model Name: {params['file_name']}\n" + \
           f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}\n" + \
           f"** Final Fitness: {fitness(params['row_vals'], best_individual, params['target_val'])}"


def train_on_schedule(output_file_name="scheduled_runs_output.txt"):
    test_size = 16
    rl_opt = RLOptimizer()
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)

    hyper_parameters = {
        "max_lots": 11,
        "max_steps": 5e5,
        "entropy": 0.965,
        "epsilon": 0.375,  # This is clip_range
        "batch_size": greatest_closest_power(test_size),
        "learning_rate": 6.25e-3,
        "gae_lambda": 0.8,  # 0 - 1 as current or future reward prioritized
        "normalize_advantage": True,
        "verbose": False,
        "num_environments": 16,
        "file_name": "rl_n_lots_real_data",
        "tensorboard_log": "./rl_n_lots_real_data_tb_log/",
        "return_default_params": True,
    }

    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    total_output = best_individual_str(best_individuals, params) + "\n"

    test_size = 300
    rl_opt.load_test_data(test_size=test_size, min_bound=-25000, max_bound=25000, percent_negative=0.5)
    hyper_parameters["max_steps"] = 1e6
    hyper_parameters["gae_lambda"] = 0.5
    hyper_parameters["batch_size"] = greatest_closest_power(test_size)
    hyper_parameters["file_name"] = "rl_300_test_-25000+25000_0.5"
    hyper_parameters["tensorboard_log"] = "rl_300_test_-25000+25000_0.5_tb_log/"
    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    total_output += best_individual_str(best_individuals, params) + "\n"

    hyper_parameters["ga_optimizer"] = GAOptimizer()
    hyper_parameters["file_name"] = "rl_300_test_-25000+25000_0.5_ga_compared"
    hyper_parameters["tensorboard_log"] = "rl_300_test_-25000+25000_0.5_ga_compared_tb_log/"
    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    total_output += best_individual_str(best_individuals, params) + "\n"

    with open(output_file_name, "w") as f:
        f.write(total_output)


def train_ga_rl_combo():
    test_size = 16
    rl_opt = RLOptimizer()
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)

    hyper_parameters = {
        "max_lots": 11,
        "max_steps": 5e5,
        "entropy": 0.965,
        "epsilon": 0.375,
        "batch_size": greatest_closest_power(test_size),
        "learning_rate": 6.25e-3,
        "gae_lambda": 0.8,
        "normalize_advantage": True,
        "verbose": False,
        "num_environments": 16,
        "file_name": "rl_300_test_-25000+25000_0.5_ga_compared",
        "tensorboard_log": "rl_300_test_-25000+25000_0.5_ga_compared_tb_log/",
        "return_default_params": True,
        "ga_optimizer": GAOptimizer(),
    }

    best_individuals, params = rl_opt.optimize_for(hyper_parameters=hyper_parameters)
    total_output = best_individual_str(best_individuals, params) + "\n"

    print(total_output)


if __name__ == "__main__":
    train_n_lots()
    # train_on_schedule()
    # train_ga_rl_combo()

