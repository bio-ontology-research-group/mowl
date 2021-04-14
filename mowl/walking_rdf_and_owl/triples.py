from random import random

class Triplet():
    
    def __init__(self, entity_1, relation, entity_2, score = random()):
        self.__entity_1 = entity_1
        self.__relation = relation
        self.__entity_2 = entity_2
        self.__score    = float(score) # for developing purposes

    def __repr__(self):
        return '\t'.join((self.__entity_1, self.__relation, self.__entity_2, str(self.__score)))

    def __eq__(self, other):
        on_entity_1 = self.__entity_1 == other.entity_1
        on_relation = self.__relation == other.relation
        on_entity_2 = self.__entity_2 == other.entity_2

        return on_entity_1 and on_relation and on_entity_2


    def __key(self):
        return (self.__entity_1, self.__relation, self.__entity_2)

    def __hash__(self):
        return hash(self.__key())

    @property
    def entity_1(self):
        return self.__entity_1

    @property
    def entity_2(self):
        return self.__entity_2

    @property
    def relation(self):
        return self.__relation
    # @relation.setter
    # def relation(self, value)
    #     self.__relation = value
    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, value):
        self.__score = value

    @classmethod
    def create_valid_triplet(cls, string):
        # will try to parse axioms of the form: "ENTITY_1 SubClassOf RELATION some ENTITY_2"
        words = string.rstrip('\n').split(' ')

        if len(words) == 5 and words[1] == "SubClassOf" and words[3] == "some":
            entity_1 = words[0]
            relation = words[2]
            entity_2 = words[4]

            return cls(entity_1, relation, entity_2)
        
        else:
            return None


    @classmethod
    def read_triplet(cls, string):
        entity_1, relation, entity_2, score = tuple(string.rstrip('\n').split('\t'))
        return cls(entity_1, relation, entity_2, score)

    @classmethod
    def groupBy_relation(cls, triplets):
        dict = {}

        for triplet in triplets:
            rel = triplet.relation

            if not rel in dict:
                dict[rel] = [triplet]
            else:
                dict[rel].append(triplet)
        return dict


    @classmethod
    def groupBy_entity_relation(cls, triplets):
        """
        Given a list of triplets [(s_1,r_1,t_2),...,(s_n,r_n,t_n)], the function groups the triplets by checking equality on s_l and r_l. The result has the following form:

        [[s_1,r_1,t_i],...,[s_1,r_k,t_i],...,[s_j,r_1,t_i],...,[s_j,r_k,t_i]]. j is the number of source nodes. k is the number of relations and i is a value from 1 to the number of target nodes.
        """
        dict = {}

        for triplet in triplets:
            entity_1 = triplet.entity_1
            rel = triplet.relation

            if not (entity_1, rel) in dict:
                dict[(entity_1, rel)] = [triplet]
            else:
                dict[(entity_1, rel)].append(triplet)
        return dict


    @classmethod
    def to_file(cls, triplets, out_file):
        with open(out_file, 'w') as f:
            for triplet in triplets:
                f.write(str(triplet)+'\n')

    @classmethod
    def to_zero(cls, triplets):
        triplets = list(triplets)[:]
        for i in range(len(triplets)):
            triplets[i].score =0
        
        return set(triplets)