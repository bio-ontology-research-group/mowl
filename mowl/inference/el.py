from mowl.inference.axiom_scoring import AxiomScoring
import re


class GCI0Score(AxiomScoring):
    def __init__(self, gci0_method, class_list):
        patterns = ["c?? SubClassOf c??",
                    "not c?? or c?? SubClassOf owl:Nothing",
                    "not c??"]
        super().__init__(patterns, gci0_method, class_list)

    def standardize_pattern(self, pattern):
        if "SubClassOf" in pattern:
            return super().standardize_pattern(pattern)
        else:
            pattern = pattern.strip()
            pattern_decomp = re.split("\s+", pattern)
            objects = [pattern_decomp[1]]
            objects.append("c?http://www.w3.org/2002/07/owl#Nothing?")
            return objects


class GCI1Score(AxiomScoring):
    def __init__(self, gci0_method, class_list):
        patterns = ["c?? and c?? SubClassOf c??", "c?? DisjointWith c??"]
        super().__init__(patterns, gci0_method, class_list)

        def standardize_pattern(self, pattern):
            if "DisjointWith" in pattern:
                pattern = pattern.strip()
                pattern_decomp = re.split("\s+", pattern)
                objects = [pattern_decomp[0], pattern_decomp[2]]
                objects.append("c?http://www.w3.org/2002/07/owl#Nothing?")
                return objects
            else:
                return super().standardize_pattern(pattern)


class GCI2Score(AxiomScoring):
    def __init__(self, gci0_method, class_list, property_list):
        patterns = ["c?? SubClassOf p?? some c??"]
        super().__init__(patterns, gci0_method, class_list, property_list)


class GCI3Score(AxiomScoring):
    def __init__(self, gci0_method, class_list, property_list):
        patterns = ["p?? some c?? SubClassOf c??"]
        super().__init__(patterns, gci0_method, class_list, property_list)
