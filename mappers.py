import numpy as np

class MappingProcess:
    BNF_grammar = {}
    genotype = []
    phenotype = []

    def __init__(self, BNF_grammar, start, genotype, wrapping):
        genotype = genotype.tolist()
        self.BNF_grammar = BNF_grammar
        self.phenotype = start
        for i in range(-1, wrapping):
            self.genotype = self.genotype + genotype


class DepthFirst(MappingProcess):
    def apply_mapping(self):
        self.phenotype = ["<expr>"]
        for codon_value in self.genotype:
            i = 0
            for symbol in self.phenotype:
                try:
                    rules = self.BNF_grammar[symbol]
                    num_rules = len(rules)
                    rule = codon_value % num_rules
                    #print(codon_value,"%",num_rules,"=",rule)
                    self.phenotype.pop(i)
                    j = i
                    for s in rules[int(rule)]:
                        self.phenotype.insert(j, s)
                        j = j + 1
                    break
                except Exception as e:
                    i = i + 1
                    pass
            #print(self.phenotype)
        return self.phenotype


class piGE(MappingProcess):
    def apply_mapping(self):
        return self.phenotype