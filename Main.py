from GA_CO import GAOptimizer
from RL_CO import RLOptimizer, scaled_sigmoid
from CO_Optimizer import grain_flow_path, wheat_supply_path, evaluate_individual
import numpy as np


def parse_column_unique_labels(dataframe, unique_column, header_row=2, include_columns=None):
    column_vals = {}
    for a in range(header_row + 1, dataframe.max_row + 1):
        row_dict = {}
        for b in range(1, dataframe.max_column + 1):
            if include_columns is None: row_dict[dataframe.cell(header_row, b).value] = dataframe.cell(a, b).value
            else:
                if b in include_columns: row_dict[dataframe.cell(header_row, b).value] = dataframe.cell(a, b).value

        try:
            column_vals[dataframe.cell(a, unique_column).value].append(row_dict)
        except KeyError:
            column_vals[dataframe.cell(a, unique_column).value] = [row_dict]

    del column_vals[None]

    return column_vals


def score_data(data, weights=None, min_max=None, weighted_average=False, scaled=True, precision=10, verbose=False):
    """
    Assigns a score to each key of the data dictionary
    Pure Summation:
     - Given N dictionaries as rows for each element of data, sum every value in each row to create the feature vector.
    Euclidean Norm:
     - Given N dictionaries as rows for each element of data, square every value in every row, take the sum of all
       squared elements in each column, then take the square root of that sum of squares to create the final feature
       vector.

    Score is assigned by taking a scaled sum of weighted features. Higher scores are better than lower scores.
    """
    default_weights = weights if weights is not None else np.ones(len(data[list(data.keys())[0]][0].keys()))
    ret_dict = {}

    for key, item in data.items():
        matrix = np.array([list(d.values()) for d in item])
        if verbose: print(f"Key: {key}, Matrix: {matrix}")

        if not weighted_average:
            data_vector = np.sum(matrix, axis=0)
            k = 0.0000001
        else:
            data_vector = np.mean(matrix, axis=0)
            k = 0.0000004

        if verbose: print(f"Key: {key}, Data Vector: {data_vector}")
        if scaled:
            scaled_score = scaled_sigmoid(np.sum(data_vector * default_weights), k_steepness=k)
            if min_max is None: ret_dict[key] = float(f"{scaled_score:.{precision}f}")
            else:
                min_val, max_val = min_max
                scaled_score = scaled_sigmoid(np.sum(data_vector * default_weights), min=min_val, max=max_val, k_steepness=k)
                ret_dict[key] = float(f"{scaled_score:.{precision}f}")
        else: ret_dict[key] = float(f"{np.sum(data_vector * default_weights):.{precision}f}")

    return ret_dict


def initial_main():
    optimizer = GAOptimizer()
    optimizer.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)
    optimizer.vectorize_column(col=37)

    """include_columns =  [14, 20, 23, 24, 25, 26, 27, 28, 33, 36, 37]
    weights = np.array([ 1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1])  # Negative weights for negative columns

    dataframe = load_data(wheat_supply_path, "EnviroSpec Vision data Table")  # Row 2 is column header
    lot_data = parse_column_unique_labels(dataframe, 3, include_columns=include_columns)
    final_scores = score_data(lot_data, weights=weights, precision=5, weighted_average=False)

    for key, item in final_scores.items():
        print(f"Lot {key}, Score: {item}")"""


def combo_optimization():
    rl_opt = RLOptimizer(max_column=41)
    ga_opt = GAOptimizer(max_column=41)
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)
    best_rl_individual, params = rl_opt.predict(num_steps=100,
                                                hyper_parameters={"return_default_params": True})

    hyper_parameters = {
        "max_lots": params["max_lots"],
        "row_vals": params["row_vals"],
        "target_vals": params["target_val"],
        "number_generations": 50,
        "verbose": params["verbose"],
    }

    best_individual = ga_opt.optimize_from(best_rl_individual, hyper_parameters=hyper_parameters)
    print(f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}")


def test_combo_optimization():
    rl_opt = RLOptimizer(max_column=41)
    ga_opt = GAOptimizer(max_column=41)
    # rl_opt.load_test_data(test_size=1000, min_bound=-22000, max_bound=11000)
    rl_opt.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)
    hyper_parameters = {
        "return_default_params": True,
        "max_lots": 6,
        "file_name": "rl_n_lots_realistic_distribution",
    }
    best_rl_individual, params = rl_opt.predict(num_steps=16,
                                                hyper_parameters=hyper_parameters)

    print(f"**  RL Individual: {best_rl_individual}")
    print(f"**  RL Solution:   {evaluate_individual(best_rl_individual, params['row_vals'])}")

    hyper_parameters = {
        "max_lots": params["max_lots"],
        "row_vals": params["row_vals"],
        "target_vals": params["target_val"],
        "number_generations": 25,
        "verbose": params["verbose"],
        "individual_mutation_chance": 0.3,
        "gene_mutation_chance": 0.15,
    }

    best_individual = ga_opt.optimize_from(best_rl_individual, hyper_parameters=hyper_parameters)
    print(f"**  Best Individual: {evaluate_individual(best_individual, params['row_vals'])}")


# file format as: "EnviroSpec vision EIM yyyy-mm-dd.csv"
# conversion of data (cannot protect fancy math)
# solving a problem (the AI combinatorial optimization problem) can be protected
if __name__ == "__main__":  # Indexed at 1 for col and row
    # combo_optimization()
    test_combo_optimization()

