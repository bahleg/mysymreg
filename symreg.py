from typing import Callable
import numpy as np
import operator
import math
import random
from copy import deepcopy
from deap import base, creator, tools
from deap import tools

import deap
from deap import algorithms


class Node:
    """
    This class represents a node in an expression tree

    Tree is a node root with children
    """

    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"{self.value}(" + ", ".join(map(str, self.children)) + ")"

    def copy(self):
        return deepcopy(self)


class Composition:
    """
    A composition stores multiple trees

    We use notation:
        f = g(h(x))
    """

    def __init__(self, children_h, children_g):

        self.children_h = children_h
        self.children_g = children_g

    def __str__(self):
        buf = [f'h_{i} = ' + str(h) for i, h in enumerate(self.children_h)]
        buf2 = [f'g_{i} = ' + str(h) for i, h in enumerate(self.children_g)]
        return '\n' + '\n'.join(buf) + '\n' + '\n'.join(buf2)

    def copy(self):
        return deepcopy(self)


# default operations with argument number
DEFAULT_OPERATIONS = {
    'add': (operator.add, 2),
    'sub': (operator.sub, 2),
    'mult': (operator.mul, 2),
    #'/': (lambda x, y: x / y, 2),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'exp': (np.exp, 1),
    'log': (lambda x: np.log(abs(x) + 1e-5), 1),
    'max': (np.maximum, 2)
}


def evaluate_tree(node, variables: dict[str, np.ndarray],
                  ops: dict[str, Callable]) -> np.ndarray:
    """
    Evaluates tree with variables and operations
    """
    if node.is_leaf():
        if isinstance(node.value, str):  # variable
            return variables[node.value]
        else:
            return np.full_like(next(iter(variables.values())),
                                fill_value=node.value,
                                dtype=float)
    else:
        func, _ = ops[node.value]
        args = [
            evaluate_tree(child, variables, ops) for child in node.children
        ]
        return func(*args)


def evaluate_composition(
    composition: Composition,
    variables: dict[str, np.ndarray],
    pool: dict[str, np.ndarray | float],
    ops: dict[str, Callable],
    g_id: int = 0,
    return_hidden: bool = False
) -> np.ndarray | tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Evaluates composition

    Args:
        composition (Composition): composition to eval
        variables (dict[str, np.ndarray]): dictionary of variables
        pool (dict[str, np.ndarray | float]): pool of additional parameters
        ops (dict[str, Callable]): operations
        g_id (int, optional): index of "g" to eval. Defaults to 0.
        return_hidden (bool, optional): if True, will return both value and hidden variables from h. Defaults to False.
    """
    new_vars = {}
    for h_id, h in enumerate(composition.children_h):
        new_vars['h' + str(h_id)] = evaluate_tree(h, variables | pool, ops)
    result = evaluate_tree(composition.children_g[g_id], new_vars | pool, ops)
    if return_hidden:
        return new_vars, result
    else:
        return result


def collect_nodes(node):
    """
    Gathers all the nodes from the tree
    """
    if not isinstance(node, Node):
        return []

    result = [node]
    for child in node.children:
        result.extend(collect_nodes(child))
    if not node.is_leaf():
        result.extend(collect_nodes(child.value))
    return result


def init_param():
    """
    Utility function to initialize parameters
    """
    return round(random.uniform(-5, 5), 2)


def get_node_depth(node):
    """
    Depth of the node
    """
    d = 1
    buf = []
    buf.extend([(1, n) for n in node.children])
    while len(buf) > 0:
        d2, n = buf.pop()
        d = max(d2, d)
        buf.extend([(d2 + 1, n2) for n2 in n.children])
    return d


class SymReg:

    def __init__(
        self,
        loss_fn: Callable,
        x: np.ndarray,
        y: np.ndarray,
        loss_component_num: int,
        h_hidden_num: int,
        g_head_num: int,
        param_num: int,
        init_population=50,
        p_mutation=1.0,
        loss_weights: tuple[float] = (-1.0, ),
        operations: dict[str, Callable] = None,
        elite_part=0.1,
        mutate_params: bool = True,
        weight_err_eps=1e-10,
    ):
        """ 
        A symbolic regression class

        Args:
            loss_fn (Callable): loss function: returns multiple values. The first is to optimize, others only for logging
            x (np.ndarray): dataset objects
            y (np.ndarray): dataset labels
            loss_component_num (int): number of components in loss functions
            h_hidden_num (int): number of hidden "h" functions in compositions
            g_head_num (int): number of heads, "g" functions in compositions
            param_num (int): number of parameters to use
            init_population (int, optional): generation size. Defaults to 50.
            p_mutation (float, optional): probability of mutation. Defaults to 1.0.
            loss_weights (tuple, optional): initial weihght in loss. For minimization set negative. Defaults to (-1.0, ).
            operations (dict[str, Callable], optional): operations to use. If None, will use default. Defaults to None.
            elite_part (float, optional): percentage of population that will be greedily taken from top. Defaults to 0.1.
            mutate_params (bool, optional): if set, will also change values of parameters. Defaults to True.
            weight_err_eps (float, optional): near-zero value. Required for logging mainly. Defaults to 1e-10.
        """
        self.loss_fn = loss_fn
        self.loss_component_num = loss_component_num
        self.h_hidden_num = h_hidden_num
        self.g_head_num = g_head_num
        self.param_num = param_num
        self.init_population = init_population
        self.p_mutation = p_mutation
        self.loss_weights = loss_weights
        self.operations = operations or DEFAULT_OPERATIONS
        self.elite_part = elite_part
        self.mutate_params = mutate_params
        self.weight_err_erps = weight_err_eps
       
        self.x = x
        self.y = y

        self.pool = {f'p{i}': init_param() for i in range(self.param_num)}
        self.variables = [f'x{i}' for i in range(self.x.shape[1])]
        self.hidden_variables = [f'h{i}' for i in range(self.h_hidden_num)]

        creator.create(
            "FitnessMin",
            base.Fitness,
            weights=tuple([-1.0] + [weight_err_eps] * self.loss_component_num))
        creator.create("Individual", Composition, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            lambda: creator.Individual(*self.generate_composition(3)))
        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)

        toolbox.register("evaluate", self.eval_individual)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", lambda a, b:
                         (a.copy(), b.copy()))  # пока без кроссовера
        self.toolbox = toolbox
        self.pop = toolbox.population(n=self.init_population)

    def generate_composition(self, max_depth):
        h = [
            self.generate_random_tree(max_depth, self.variables)
            for _ in range(self.h_hidden_num)
        ]

        g = [
            self.generate_random_tree(max_depth, self.hidden_variables)
            for _ in range(self.g_head_num)
        ]
        return h, g

    def generate_random_tree(self, max_depth, variables):
        if max_depth == 0 or (max_depth > 1 and random.random() < 0.3):
            if random.random() < 0.5:
                return Node(random.choice(variables))
            else:
                return Node(random.choice(list(self.pool.keys())))  # const
        else:
            op = random.choice(list(self.operations.keys()))
            func, arity = self.operations[op]
            return Node(op, [
                self.generate_random_tree(max_depth - 1, variables)
                for _ in range(arity)
            ])

    def mutate(self, ind):
        ind_copy = ind.copy()

        elements_to_sample = ind_copy.children_h + ind_copy.children_g
        element = random.choice(elements_to_sample)
        if element in ind_copy.children_h:
            variables = self.variables
        else:
            variables = self.hidden_variables

        # change subtree at random
        nodes = collect_nodes(element)
        if nodes:
            node_to_replace = random.choice(nodes)
            if node_to_replace.is_leaf() and node_to_replace.value.startswith(
                    'p') and random.uniform(0, 1) < 0.5 and self.mutate_params:
                self.pool[node_to_replace.value] = init_param()
            else:

                new_subtree = self.generate_random_tree(3, variables)
                node_to_replace.value = new_subtree.value
                node_to_replace.children = new_subtree.children

        return ind_copy,

    def eval_individual(self, ind):
        try:
            variables = {f'x{i}': self.x[:, i] for i in range(self.x.shape[1])}
            result = self.loss_fn(self.x, self.y, ind, variables, self.pool,
                                  self.operations)
            return result

        except Exception as e:
            print(f'error: {e}')
            return tuple([float('inf')] * len(self.weights))

    def fit_epoch(self):
        offspring = algorithms.varAnd(self.pop,
                                      self.toolbox,
                                      cxpb=0.0,
                                      mutpb=self.p_mutation)

        for ind in offspring:
            ind.fitness.values = self.toolbox.evaluate(ind)

        elite_size = int(len(self.pop) * self.elite_part)
        elite = tools.selBest(self.pop, k=elite_size)
        self.pop = self.toolbox.select(
            offspring + self.pop +
            self.toolbox.population(n=self.init_population),
            k=len(self.pop) - elite_size) + elite

    def fit(self,
            epoch_num,
            log_models_every: int = -1,
            log_perf_every: int = -1):
        for e in range(epoch_num):
            self.fit_epoch()
            best = tools.selBest(self.pop, 1)[0]
            err = best.fitness.values
            if log_models_every > 0 and e % log_models_every == 0:
                print(f"[Gen {e}] Expr: {best}")
            if log_perf_every > 0 and e % log_perf_every == 0:
                print(f"[Gen {e}] Best errors: {err}")
        best = tools.selBest(self.pop, 1)[0]
        return best
