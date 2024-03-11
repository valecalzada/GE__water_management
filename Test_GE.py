import bnf_grammar
from SearchEngines import DE
from SearchEngines.FitnessFunction import FitnessFunction, list_to_dictionary
from mapping_process import mappers
import json

instance = json.load(open('test_instances_Scenario_2/instance_v5_u2_k2_c1.json'))
v_n = len(instance['V'])
u_n = len(instance['U'])
c_n = len(instance['carriers'])
x_inf = 0
x_sup = 50

BNF_grammar = bnf_grammar.create(v_n, u_n, c_n, x_inf, x_sup)
start = ["<expr>"]
print(BNF_grammar)

population_size = 200
dimensions = 500
lower_limit = 0
upper_limit = 255
fitness_function = FitnessFunction(BNF_grammar, start, instance, wrapping=0)
stopping_criterion = 1E-6
iterations = 1000
bias = 0.
cut_points = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

print("Parameters:")
print("Population size:", population_size)
print("Dimensions:", dimensions)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Stopping criterion:", stopping_criterion)
print("Iterations:", iterations)
print("Cut points:", cut_points)

# Evolutionary process
Cr = 0.9
F = 0.5
algorithm = DE.Rand_Bin(population_size, dimensions, Cr, F,
                        lower_limit, upper_limit, fitness_function,
                        stopping_criterion + bias, cut_points,
                        print_evolution=True)
individuals = algorithm.evolution(iterations)
genotype = individuals[0][-1].chromosome
phenotype = mappers.DepthFirst(BNF_grammar, start, genotype, wrapping=0).apply_mapping()
print("Phenotype:", list_to_dictionary(phenotype))
print("Fitness:", individuals[0][-1].fitness_value)
