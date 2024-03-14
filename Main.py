from GA_CO import GAOptimizer
from CO_Optimizer import grain_flow_path, wheat_supply_path, evaluate_individual, optimize_init_solution, scaled_sigmoid
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


def test_optimize_init_state():
    # test_individual = [-6, 6, 7, -7, 8, -8, 10, -10, 1, -1]
    test_individual = [3414.3241176470615, 10741.626925490193, 10341.393921568633, 2302.466015686273,
                       -4057.0649647058804, -7462.052215686282, -11301.34425882353, -7044.932803921577,
                       -3753.3592470588246, -22032.892000000007, 3147.5504766536988, 11677.476999999997,
                       -1594.277999999995, -1046.7939999999955, 1628.2349999999958, -2084.1659999999974]
    optimize_init_solution(test_individual, 7, 0)


if __name__ == "__main__":  # Indexed at 1 for col and row
    # combo_optimization()
    # test_combo_optimization()
    test_optimize_init_state()

