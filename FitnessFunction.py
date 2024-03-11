from mapping_process import mappers
import numpy as np


def compute_fitness(phenotype, instance):
    x_values = []
    for subdict in phenotype.values():
        for value in subdict.values():
            x_values.append(int(value))

    c_values = []
    for key in phenotype.keys():
        subdict = phenotype[key]
        dict_keys = subdict.keys()
        modified_keys = [key.replace("_", ",") for key in dict_keys]
        for subkey in modified_keys:
            subdict = instance['costs'][key][subkey]
            c_values.append(subdict)

    return np.sum(np.array(c_values) * np.array(x_values))


def evaluate_conditions(phenotype, instance):
    flag = [True, True, True]
    # Revisar restricción 1: Tipos de agua
    for k, v in phenotype.items():
        for key, value in v.items():
            v_key, u_key = key.split('_')  # Descomponer la clave compuesta
            tipeU = instance['T'][u_key]
            tipeV = instance['T'][v_key]
            if tipeU != tipeV:
                flag[0] = False

    # Revisar restricción 2: Capacidades de los Carriers.
    # Solo se cuenta un uso por cada carrier de cada v_i
    for k, v in phenotype.items():
        # Guarda los Valores unicos de v_i para cada carrier
        difV = set()
        for subclave in v:
            v_key, _ = subclave.split('_')
            if v_key not in difV:
                difV.add(v_key)
        if instance['L'][k] < len(difV):
            flag[1] = False

    # Revisar restricción 3
    sumasV = {}  # guardar las sumas por cada carrier lx para V
    sumasU = {}  # guardar las sumas por cada carrier lx para U
    for k, v in phenotype.items():
        for subclave, valor in v.items():
            v_key, u_key = subclave.split('_')
            if v_key not in sumasV:
                sumasV[v_key] = 0
            sumasV[v_key] += int(valor)
            if u_key not in sumasU:
                sumasU[u_key] = 0
            sumasU[u_key] += int(valor)
    for ik, i in sumasV.items():
        if (instance['S'][ik] < i):
            flag[2] = False

    for ik, i in sumasU.items():
        if (instance['D'][ik] > i):
            flag[2] = False

    #flag[0] = True
    #flag[1] = True
    #flag[2] = True
    #print(phenotype)
    #print(flag)

    if all(flag):
        return True
    else:
        return False


def check_nodes_u(phenotype, instance):
    # Revisar que la solución tenga todos los u's
    U = instance['U']
    check = 0
    flag = False
    for u in U:
        for symbol in phenotype:
            if (u == symbol):
                check = check + 1
                break
    if check >= (len(U) * 1):
        flag = True
    return flag


def list_to_dictionary(phenotype):
    # Inicializar el diccionario vacío
    dictionary = {}

    # Variables auxiliares para realizar el seguimiento del análisis
    current_label = None
    current_values = []

    # Recorremos los elementos de la lista phenotype
    for item in phenotype:
        if item.startswith('l'):
            current_label = item
        elif item == '[':
            current_values = []
        elif item == ']':
            sub_dict = {}
            for i in range(0, len(current_values), 3):
                sub_dict[f"{current_values[i]}_{current_values[i + 1]}"] = current_values[i + 2]
            if current_label in dictionary:
                dictionary[current_label].update(sub_dict)
            else:
                dictionary[current_label] = sub_dict
        else:
            current_values.append(item)
    return dictionary


def check_non_terminals(lst):
    for sub_list in lst:
        for item in sub_list:
            if "<" in item or ">" in item:
                return False
    return True


class FitnessFunction:
    def __init__(self, BNF_grammar, start, instance, wrapping):
        self.BNF_grammar = BNF_grammar
        self.start = start
        self.instance = instance
        self.wrapping = wrapping

    def evaluate(self, x):
        phenotype = mappers.DepthFirst(self.BNF_grammar, self.start, x, self.wrapping).apply_mapping()
        if check_non_terminals(phenotype):
            #print("Non-terminals")
            if check_nodes_u(phenotype, self.instance):
                #print("All u")
                phenotype = list_to_dictionary(phenotype)
                if evaluate_conditions(phenotype, self.instance):
                    #print("All conditions")
                    return compute_fitness(phenotype, self.instance)
                else:
                    return 1E9
            else:
                return 1E9
        else:
            return 1E9