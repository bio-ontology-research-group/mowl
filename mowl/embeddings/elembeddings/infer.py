from mowl.inference.axiom_scoring import AxiomScoring
import re

class GCI0Score(AxiomScoring):
    def __init__(self, gci0_method, class_list):
        patterns = ["c?? subclassOf c??", "not c?? or c?? subclassOf bottom"]
        super().__init__(patterns, gci0_method, class_list)

        
