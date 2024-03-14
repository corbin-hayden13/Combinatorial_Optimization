import numpy as np
import random
from copy import deepcopy
from time import time
from threading import Thread
from multiprocessing import Process
from CO_Optimizer import Optimizer, grain_flow_path, wheat_supply_path,\
     vector_from_dataframe_column, make_population, fitness, evaluate_individual, make_individual,\
     clone_individual


def non_threaded_tournament_selection(row_vals, population, target_val, t_size=4):
    selected_parents = []
    while len(selected_parents) < len(population):
        indices = np.arange(len(population))
        tournament_inds = np.random.choice(indices, size=t_size)
        tournament = [population[a] for a in tournament_inds]
        winner = max(tournament, key=lambda x: fitness(row_vals, x, target_val))
        selected_parents.append(winner)

    return selected_parents


def threaded_tournament_selection(row_vals, population, target_val, ret_dict, tournament_num, t_size=4):
    indices = np.arange(len(population))
    tournament_inds = np.random.choice(indices, size=t_size)
    tournament = [population[a] for a in tournament_inds]
    fitness_vals = np.array([fitness(row_vals, individual, target_val) for individual in tournament])
    ret_dict[tournament_num] = tournament[np.argmax(fitness_vals)]


def tournament_selection(row_vals, population, target_val, t_size=4, multithreaded=True):
    if multithreaded:
        tournaments = []
        ret_dict = {}
        for a in range(len(population)):
            tournaments.append(Thread(target=threaded_tournament_selection,
                                      args=(row_vals, population, target_val, ret_dict, a, t_size)))
            tournaments[-1].start()

        for tournament in tournaments:
            tournament.join()

        return [parent for _, parent in ret_dict.items()]

    else: return non_threaded_tournament_selection(row_vals, population, target_val, t_size=t_size)


"""
Enforcing minimum field count:
 - Crossover only if parent_one:lot_number == parent_two:lot_number
   OR field count remains >= minimum field count for both lot numbers from each child
"""
def min_field_count_uniform_crossover(parent_one, parent_two, crossover_chance, min_field_count):
    """
    List comprehension comparison logic:
     - Count the number of fields in a lot given one of the fields from that lot is removed:
       - Create list of fields removing the one field to be swapped
       - Count fields remaining in lot number of swapped field to verify the count is >= min_field_count
       - Do this for both children
    """
    child_one = sorted(parent_one)
    child_two = sorted(parent_two)
    for a in range(len(parent_one)):
        if random.random() < crossover_chance:
            if child_one[a][1] == child_two[a][1] or (len([True for index, lot_num in [field for field in child_one
                                                                                       if field != child_one[a]]
                                                           if lot_num == child_one[a][1]]) >= min_field_count
                                                      and len([True for index, lot_num in [field for field in child_two
                                                                                           if field != child_two[a]]
                                                               if lot_num == child_two[a][1]]) >= min_field_count):
                child_one_lot_num = child_one[a][1]
                child_one[a] = (child_one[a][0], child_two[a][1])
                child_two[a] = (child_two[a][0], child_one_lot_num)

    return child_one, child_two


def uniform_crossover(parent_one, parent_two, crossover_chance):
    child_one = deepcopy(parent_one)
    child_two = deepcopy(parent_two)
    for a in range(len(parent_one)):
        if random.random() < crossover_chance:
            child_one[a] = parent_two[a]
            child_two[a] = parent_one[a]

    return child_one, child_two


def crossover_selection(winning_population, crossover_chance=0.5, min_field_count=-1):
    new_generation = []
    for a in range(0, len(winning_population), 2):
        try:
            if min_field_count == -1:
                child_one, child_two = uniform_crossover(winning_population[a], winning_population[a + 1],
                                                         crossover_chance)
            else:
                child_one, child_two = min_field_count_uniform_crossover(winning_population[a], winning_population[a + 1],
                                                                         crossover_chance, min_field_count)

            new_generation.append(child_one)
            new_generation.append(child_two)
        except IndexError:
            new_generation.append(winning_population[a])

    return new_generation


def legacy_min_field_count_mutate(individual, max_lots, min_field_count, individual_mutation_chance, gene_mutation_chance):
    ancestor = deepcopy(individual)
    new_individual = []

    if random.random() > individual_mutation_chance:
        return individual

    while len(ancestor) > 1:
        unique_lots, lot_counts = np.unique(np.array(ancestor)[:, 1], return_counts=True)
        if random.random() < 0.5 and len(unique_lots) > 1:  # Combine lots
            lots_to_combine = np.random.choice(unique_lots, 2, replace=False)
            to_combine = [index_lot_num for index_lot_num in ancestor if index_lot_num[1] in lots_to_combine]
            ancestor = [index_lot_num for index_lot_num in ancestor if index_lot_num[1] not in lots_to_combine]
            for index_lot_num in to_combine:
                if random.random() < gene_mutation_chance:  # actually combine the lots
                    new_individual.append((index_lot_num[0], lots_to_combine[0]))
                else:  # Do not mutate genes / combine, simply add to the individual
                    new_individual.append(index_lot_num)

        else:  # Split
            random_lot_ind = random.randint(0, len(unique_lots) - 1)
            if random.random() < gene_mutation_chance and lot_counts[random_lot_ind] >= 2 * min_field_count:
                fields = [index_lot_num for index_lot_num in ancestor if index_lot_num[1] == unique_lots[random_lot_ind]]
                ancestor = [index_lot_num for index_lot_num in ancestor if index_lot_num not in fields]
                new_lot_num = max(unique_lots) + 1
                for index_lot_num in fields[:len(fields) // 2]:
                    new_individual.append((index_lot_num[0], new_lot_num))

                for index_lot_num in fields[len(fields) // 2:]:
                    new_individual.append(index_lot_num)

            else:
                not_split = [index_lot_num for index_lot_num in ancestor if index_lot_num[1] == unique_lots[random_lot_ind]]
                ancestor = [index_lot_num for index_lot_num in ancestor if index_lot_num[1] != unique_lots[random_lot_ind]]
                for index_lot_num in not_split:
                    new_individual.append(index_lot_num)

        if len(ancestor) + len(new_individual) != len(individual):
            print(f"Something's not adding up...")
            exit(1)

        if len(new_individual) != len(individual):
            print(f"New Individual: {len(new_individual)} is not same size as Individual: {len(individual)}")
            exit(1)

    return new_individual


def mutate_cross_split(individual, min_field_count, individual_mutation_chance, gene_mutation_chance,
                       split_chance=0.75):
    ancestor = deepcopy(individual)
    new_individual = []

    if random.random() > individual_mutation_chance:
        return individual

    while len(ancestor) > 0:
        unique_lots, lot_counts = np.unique(np.array(ancestor)[:, 1], return_counts=True)
        items_to_remove = []

        if random.random() > split_chance and len(unique_lots) > 1:  # Combine lots
            lots_to_combine = np.random.choice(unique_lots, 2, replace=False)
            found_fields = [index_lot_num for index_lot_num in ancestor if index_lot_num[1] in lots_to_combine]
            if random.random() < gene_mutation_chance:
                new_individual.extend([(index_lot_num[0], lots_to_combine[0]) for index_lot_num in found_fields])
            else:
                new_individual.extend(found_fields)

            items_to_remove.extend(found_fields)

        else:  # Split
            random_lot_ind = random.randint(0, len(unique_lots) - 1)
            if random.random() < gene_mutation_chance and lot_counts[random_lot_ind] >= 2 * min_field_count:
                fields = [index_lot_num for index_lot_num in ancestor if
                          index_lot_num[1] == unique_lots[random_lot_ind]]
                new_lot_num = max(unique_lots) + 1
                for index_lot_num in fields[:len(fields) // 2]:
                    new_individual.append((index_lot_num[0], new_lot_num))

                for index_lot_num in fields[len(fields) // 2:]:
                    new_individual.append(index_lot_num)

                items_to_remove.extend(fields)
            else:
                for index_lot_num in ancestor:
                    if index_lot_num[1] == unique_lots[random_lot_ind]:
                        new_individual.append(index_lot_num)
                        items_to_remove.append(index_lot_num)

        # Remove items after iterating through the loop
        for item in items_to_remove: ancestor.remove(item)

        if len(new_individual) + len(ancestor) != len(individual):
            print(f"Lengths not matching up...\nNew Individual: {len(new_individual)}, Ancestor: {len(ancestor)}")
            exit(1)

    if len(new_individual) != len(individual):
        print(f"Lengths not matching up...\nNew Individual: {len(new_individual)}")
        exit(1)

    return new_individual


def swap_fields_per_lot(individual, swap_chance=0.5):
    for a in range(len(individual)):
        if random.random() < swap_chance:
            potential_fields = individual[:a] + individual[a + 1:]
            field_to_swap = random.randint(0, len(potential_fields) - 1)
            field_to_swap = field_to_swap if field_to_swap < a else field_to_swap + 1
            org_field = individual[a][0]
            individual[a] = (individual[field_to_swap][0], individual[a][1])
            individual[field_to_swap] = (org_field, individual[field_to_swap][1])


def min_field_count_mutate(individual, min_field_count, individual_mutation_chance, gene_mutation_chance,
                           split_chance=0.75):
    """
    Implemented as an internal lot crossover performing one or more of the following operations:
     - Combine 2 lots together
     - Split a lot into two separate lots
     - Swap fields between lots
    """
    new_individual = mutate_cross_split(individual, min_field_count, individual_mutation_chance, gene_mutation_chance,
                                        split_chance=split_chance)
    # swap_fields_per_lot(new_individual, gene_mutation_chance)

    return new_individual


def mutate(individual, max_lots, individual_mutation_chance, gene_mutation_chance):
    if random.random() > individual_mutation_chance:
        return individual

    arr = np.array(individual)
    random_chances = np.random.random(size=len(arr))
    mutation_indices = np.where(random_chances < gene_mutation_chance)[0]
    arr[mutation_indices, 1] = np.random.randint(0, max_lots, size=len(mutation_indices))

    return list(map(tuple, arr))


def mutate_population(population, max_lots, min_field_count=-1, individual_mutation_chance=0.25,
                      gene_mutation_chance=0.15):
    if min_field_count == -1:
        return [mutate(individual, max_lots, individual_mutation_chance, gene_mutation_chance)
                for individual in population]
    else:
        return [min_field_count_mutate(individual, min_field_count, individual_mutation_chance, gene_mutation_chance,
                                       split_chance=0.65)
                for individual in population]


def run_genetic_algorithm(vals_to_optimize, max_lots, target_val=0, min_field_count=-1,
                          population_size=100, min_generations=10, num_generations=1000, allow_early_convergence=True,
                          individual_mutation_chance=1, gene_mutation_chance=0.5,
                          verbose=False, multithreaded=False, override_population=None):
    if override_population is None: init_population = make_population(vals_to_optimize, max_lots,
                                                                      min_field_count=min_field_count,
                                                                      pop_size=population_size)
    else: init_population = override_population

    for a in range(num_generations):
        if verbose: print(f"Running generation {a + 1} / {num_generations}")
        winning_population = tournament_selection(vals_to_optimize, init_population, target_val)
        new_generation = crossover_selection(winning_population, min_field_count=min_field_count)
        init_population = mutate_population(new_generation, max_lots, min_field_count=min_field_count,
                                            individual_mutation_chance=individual_mutation_chance,
                                            gene_mutation_chance=gene_mutation_chance)
        best_individual = max(init_population, key=lambda individual: fitness(vals_to_optimize, individual, target_val))
        if (np.array([sums for _, sums in evaluate_individual(best_individual, vals_to_optimize).items()]) < 0).all()\
                and a >= min_generations and allow_early_convergence:
            if verbose: print(f"Algorithm converged before hitting max generation")

            return best_individual

        if verbose:
            print(f"Highest individual score = {fitness(vals_to_optimize, best_individual, target_val)}")

    return max(init_population, key=lambda individual: fitness(vals_to_optimize, individual, target_val))


def parse_constraints(data, constraints, valid_constraints):
    use_min_fields = False
    if constraints is None:
        max_lots = data.shape[0] - 1  # Number of rows in numpy ndarray
        min_fields = 1
    else:
        for key in constraints.keys():
            if key not in valid_constraints:
                print(f"Invalid key \"{key}\" passed as a constraint, quitting training...")
                return

        try:
            max_lots = constraints["max_lots"]
            if max_lots >= data.shape[0] - 1: max_lots = data.shape[0] - 1
            min_fields = 1
        except KeyError:
            try:
                min_fields = constraints["min_fields"]
                use_min_fields = True
                if min_fields > data.shape[0]:
                    min_fields = data.shape[0]
                    use_min_fields = False
                max_lots = data.shape[0] // min_fields
            except KeyError:  # Should never occur given previous checks
                print(f"Wow, idk how you made it here, but I'll print constraints so you can see where you went wrong\n{constraints}")

    return max_lots, min_fields, use_min_fields


class GAOptimizer(Optimizer):
    def __init__(self, init_dataframe=None, header_row=1, max_column=10):
        super().__init__(init_dataframe, header_row, max_column)

    def __manage_test_data(self):
        try:
            return self.dataframe.T
        except AttributeError:
            return self.dataframe

    def optimize_for(self, hyper_parameters=None):
        """
        Using a genetic algorithm to solve a combinatorial optimization problem based on specific category and constraint

        Accepted Parameter Values:
            * category:
                * "carbon_emissions" - Finds combinations of fields as lots such that every lot's carbon emissions are closest to zero
                * "lot_cost" - Finds combinations of fields as lots such that the cost of every lot is minimized (closest to zero)
            * constraints:
                * {"max_lots": default=row_count - 1} - constrains number of combinations of fields as lots.
                                                        Default max lots is equal to number of rows - 1.
                * {"min_fields": default=1} - constrains number of fields per lot.
                                              Default minimum fields per lot is 1.
        """

        default_parameters = {
            "category": "carbon_emissions",
            "max_lots": 5,
            "target_val": 0,
            "population_size": 100,
            "min_generations": 10,
            "number_generations": 500,
            "allow_early_convergence": True,
            "individual_mutation_chance": 1,
            "gene_mutation_chance": 0.5,
            "multithreaded": True,
            "return_params_dict": False,
            "verbose": False,
        }

        if type(hyper_parameters) == dict:
            for key, item in hyper_parameters.items(): default_parameters[key] = item

        if self.dataframe is None:
            print("Please call \"import_data()\" before \"optimize_for()\" to load a dataset into the optimizer")
            return

        valid_categories = {
            "carbon_emissions": vector_from_dataframe_column(self.dataframe, col=37),
            "lot_cost": vector_from_dataframe_column(self.dataframe, col=41),
            "test_data": self.__manage_test_data()
        }

        try:
            data = valid_categories[default_parameters["category"]]
        except KeyError:
            print(f"Category \"{default_parameters['category']}\" is invalid, quitting training...")
            return

        self.evaluated_column = data

        valid_constraints = ["max_lots"]  # planned +"min_fields"
        try:
            max_lots, min_fields, use_min_fields = parse_constraints(data, {valid_constraints[0]: default_parameters[valid_constraints[0]]}, valid_constraints)

        except KeyError:
            try:
                max_lots, min_fields, use_min_fields = parse_constraints(data, {valid_constraints[1]: default_parameters[valid_constraints[1]]}, valid_constraints)

            except KeyError:
                print(f"No valid constraints from {valid_constraints} were passed, exiting...")
                exit(1)

        default_parameters["final_params"] = {"max_lots": max_lots, "min_fields": min_fields, "training_data": data}

        best_individual = run_genetic_algorithm(data, max_lots,
                                                target_val=default_parameters["target_val"],
                                                min_field_count=-1 if not use_min_fields else min_fields,
                                                population_size=default_parameters["population_size"],
                                                min_generations=default_parameters["min_generations"],
                                                num_generations=default_parameters["number_generations"],
                                                allow_early_convergence=default_parameters["allow_early_convergence"],
                                                individual_mutation_chance=default_parameters["individual_mutation_chance"],
                                                gene_mutation_chance=default_parameters["gene_mutation_chance"],
                                                verbose=default_parameters["verbose"],
                                                multithreaded=default_parameters["multithreaded"])
        self.best_individual = best_individual

        if not default_parameters["return_params_dict"]: return self.best_individual
        else: return self.best_individual, default_parameters

    def optimize_from(self, individual=None, hyper_parameters=None):
        default_parameters = {
            "row_vals": np.ones(10),
            "target_val": 0,
            "max_lots": 5,
            "population_size": 100,
            "number_generations": 500,
            "individual_mutation_chance": 1,
            "gene_mutation_chance": 0.5,
            "multithreaded": True,
            "return_params_dict": False,
            "verbose": False,
        }

        if type(hyper_parameters) == dict:
            for key, item in hyper_parameters.items(): default_parameters[key] = item

        if individual is None:
            individual = make_individual(default_parameters["row_vals"], default_parameters["max_lots"])
        population = clone_individual(individual, pop_size=default_parameters["population_size"])

        best_individual = run_genetic_algorithm(default_parameters["row_vals"], default_parameters["max_lots"],
                                                target_val=default_parameters["target_val"],
                                                min_field_count=-1,
                                                population_size=default_parameters["population_size"],
                                                num_generations=default_parameters["number_generations"],
                                                individual_mutation_chance=default_parameters["individual_mutation_chance"],
                                                gene_mutation_chance=default_parameters["gene_mutation_chance"],
                                                verbose=default_parameters["verbose"],
                                                multithreaded=default_parameters["multithreaded"],
                                                override_population=population)

        self.best_individual = best_individual

        if not default_parameters["return_params_dict"]:
            return self.best_individual
        else:
            return self.best_individual, default_parameters


def main():
    params_dict = {
        "max_lots": 5,
        "target_val": 0,
        "min_generations": 3,
        "number_generations": 30,
        "allow_early_convergence": False,
        "individual_mutation_chance": 0.3,
        "gene_mutation_chance": 0.15,
        "multithreaded": False,
        "verbose": True,
    }
    optimizer = GAOptimizer(max_column=41)
    optimizer.import_data(wheat_supply_path, workbook="EnviroSpec Vision data Table", header_row=2)
    best_individual = optimizer.optimize_for(hyper_parameters=params_dict)

    print(best_individual)
    print(evaluate_individual(best_individual, optimizer.evaluated_column))
    print(optimizer.individual_to_dict(best_individual, sort_keys=True))


if __name__ == "__main__":
    main()

