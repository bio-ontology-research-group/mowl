from mowl.datasets import PathDataset
import mowl
mowl.init_jvm("10g")


class FamilyDataset(PathDataset):
    def __init__(self):
        super().__init__("family/ontology.owl")


class GDAHumanDataset(PathDataset):
    def __init__(self):
        super().__init__("gda_human/ontology.owl", validation_path="gda_human/valid.owl",
                         testing_path="gda_human/test.owl")


class GDAHumanELDataset(PathDataset):
    def __init__(self):
        super().__init__("gda_human_el/ontology.owl", validation_path="gda_human_el/valid.owl",
                         testing_path="gda_human_el/test.owl")


class GDAMouseDataset(PathDataset):
    def __init__(self):
        super().__init__("gda_mouse/ontology.owl", validation_path="gda_mouse/valid.owl",
                         testing_path="gda_mouse/test.owl")


class GDAMouseELDataset(PathDataset):
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
