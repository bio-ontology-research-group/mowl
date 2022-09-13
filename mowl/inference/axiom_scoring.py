from tqdm import tqdm
import re
import itertools as it
import numpy as np


class AxiomScoring():

    """This is an abstract class for methods that can score axioms. The input is a set of axioms \
    patterns that can be processed by the method. One example of axiom pattern could be: \
    `c? subclassOf c?` meaning that subclass axioms are accepted. Since the previous pattern can \
    be represented as `not c? or c? subclassOf bottom`, this pattern could also be included.

    :param patterns: Collection of patterns accepted by the scoring method
    :type patterns: list
    """

    def __init__(self, patterns, method, class_list, property_list=None, canonical_pattern=0):
        self.patterns = set(patterns)
        self.canonical_pattern = patterns[canonical_pattern]
        self.method = method
        self.class_list = class_list
        self.property_list = [] if property_list is None else property_list

    def is_pattern_correct(self, pattern):
        pattern = self.canonical_expression(pattern)
        print(pattern)
        if pattern in self.patterns:
            return True
        else:
            raise ValueError("Intended pattern does not match any pattern defined for the scoring \
                method.")

    def canonical_expression(self, pattern):
        """Transformsa pattern to its canonical form. This is a fixed point method.
        """
        pattern = pattern.strip()
        pattern_decomp = re.split("\s+", pattern)
        regex = "\?.*?\?$"
        canon_pattern = list(map(lambda x: re.sub(regex, "??", x), pattern_decomp))
        return " ".join(canon_pattern)

    def standardize_pattern(self, pattern):
        """Transforms a pattern to its canonical form. This is a fixed point method.
        """
        pattern = pattern.strip()
        pattern_decomp = re.split("\s+", pattern)
        return pattern_decomp

    def pattern_to_data_points(self, pattern):
        """This method will receive any accepted pattern and transform it into data points to be \
            accepted by the method.
        """
        pattern_decomp = self.standardize_pattern(pattern)

        objects = []
        regex = re.compile("[cp]\?.*?\?")
        for pat in pattern_decomp:
            match = regex.fullmatch(pat)
            if match:
                objects.append(match.group(0))

        # objects =re.findall("[cr]\?.*?\?", pattern)
        objects = [x[:-1] for x in objects]

        objects_sub_lists = []

        for obj in objects:
            regex = obj[2:]
            print(f"regex {regex}")
            regex = re.compile(regex)

            if obj.startswith("c"):
                iter_list = self.class_list
            elif obj.startswith("p"):
                iter_list = self.property_list

            curr_list = []
            for name in iter_list:

                match = regex.fullmatch(name)
                if match:
                    curr_list.append(match.group(0))
            objects_sub_lists.append(curr_list)
        return it.product(*objects_sub_lists)

    def inverse(self, point, pattern):
        output = []
        pattern_decomp = pattern.split(" ")

        for item in pattern_decomp:
            if "??" in item:
                output.append(point[0])
                point = point[1:]
            else:
                output.append(item)
        assert len(point) == 0
        return " ".join(output)

    def score(self, pattern):
        can_pattern = self.canonical_expression(pattern)
        self.is_pattern_correct(can_pattern)
        data_points = self.pattern_to_data_points(pattern)

        preds = dict()
        for point in tqdm(data_points):

            preds[self.inverse(point, can_pattern)] = self.method(point).cpu().detach().item()

        return preds
