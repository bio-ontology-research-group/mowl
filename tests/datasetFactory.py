import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.datasets.base import OWLClasses


class FamilyDataset(PathDataset):
    def __init__(self):
        super().__init__("family/ontology.owl")


class GDADataset(PathDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def evaluation_classes(self):

        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if owl_name[7:].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://is_associated_with"


class GDAHumanDataset(GDADataset):
    def __init__(self):
        super().__init__("gda_human/ontology.owl", validation_path="gda_human/valid.owl",
                         testing_path="gda_human/test.owl")


class GDAHumanELDataset(GDADataset):
    def __init__(self):
        super().__init__("gda_human_el/ontology.owl", validation_path="gda_human_el/valid.owl",
                         testing_path="gda_human_el/test.owl")


class GDAMouseDataset(GDADataset):
    def __init__(self):
        super().__init__("gda_mouse/ontology.owl", validation_path="gda_mouse/valid.owl",
                         testing_path="gda_mouse/test.owl")


class GDAMouseELDataset(GDADataset):
    def __init__(self):
        super().__init__("gda_mouse_el/ontology.owl", validation_path="gda_mouse_el/valid.owl",
                         testing_path="gda_mouse_el/test.owl")


class PPIYeastDataset(PathDataset):
    def __init__(self):
        super().__init__("ppi_yeast/ontology.owl", validation_path="ppi_yeast/valid.owl",
                         testing_path="ppi_yeast/test.owl")


class PPIYeastSlimDataset(PathDataset):
    def __init__(self):
        super().__init__("ppi_yeast_slim/ontology.owl", validation_path="ppi_yeast_slim/valid.owl",
                         testing_path="ppi_yeast_slim/test.owl")
