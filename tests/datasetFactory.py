import os

import mowl
# mowl.init_jvm("10g")
from mowl.datasets import PathDataset, default_data_root
from mowl.datasets.base import OWLClasses

# The builtin datasets land in the user-level cache, not in the working
# directory, so resolve every path against the same root instead of assuming
# pytest runs from the repository root. When a test requests a cached file that
# has not yet been downloaded/extracted into DATA_ROOT, fetch it lazily on
# demand rather than eagerly instantiating every dataset in conftest.py.
DATA_ROOT = default_data_root()

_BUILTIN_LOADERS = {
    "family": "FamilyDataset",
    "gda_human": "GDAHumanDataset",
    "gda_human_el": "GDAHumanELDataset",
    "gda_mouse": "GDAMouseDataset",
    "gda_mouse_el": "GDAMouseELDataset",
    "ppi_yeast": "PPIYeastDataset",
    "ppi_yeast_slim": "PPIYeastSlimDataset",
    "ppi_human": "PPIHumanDataset",
    "hpi": "HPIDataset",
    "go_subsumption": "GOSubsumptionDataset",
    "foodon_subsumption": "FoodOnSubsumptionDataset",
    "gda2": "GDADatasetV2",
}


def data_path(*parts):
    """Path of a file inside the dataset cache, downloading lazily if missing."""
    path = os.path.join(DATA_ROOT, *parts)
    if not os.path.exists(path) and parts:
        top_dir = parts[0].split("/")[0]
        cls_name = _BUILTIN_LOADERS.get(top_dir)
        if cls_name is not None:
            from mowl.datasets import builtin
            getattr(builtin, cls_name)()
    return path


class FamilyDataset(PathDataset):
    def __init__(self):
        super().__init__(data_path("family/ontology.owl"))


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
        super().__init__(data_path("gda_human/ontology.owl"),
                         validation_path=data_path("gda_human/valid.owl"),
                         testing_path=data_path("gda_human/test.owl"))


class GDAHumanELDataset(GDADataset):
    def __init__(self):
        super().__init__(data_path("gda_human_el/ontology.owl"),
                         validation_path=data_path("gda_human_el/valid.owl"),
                         testing_path=data_path("gda_human_el/test.owl"))


class GDAMouseDataset(GDADataset):
    def __init__(self):
        super().__init__(data_path("gda_mouse/ontology.owl"),
                         validation_path=data_path("gda_mouse/valid.owl"),
                         testing_path=data_path("gda_mouse/test.owl"))


class GDAMouseELDataset(GDADataset):
    def __init__(self):
        super().__init__(data_path("gda_mouse_el/ontology.owl"),
                         validation_path=data_path("gda_mouse_el/valid.owl"),
                         testing_path=data_path("gda_mouse_el/test.owl"))


class PPIYeastDataset(PathDataset):
    def __init__(self):
        super().__init__(data_path("ppi_yeast/ontology.owl"),
                         validation_path=data_path("ppi_yeast/valid.owl"),
                         testing_path=data_path("ppi_yeast/test.owl"))

    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://4932" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

class PPIYeastSlimDataset(PathDataset):
    def __init__(self):
        super().__init__(data_path("ppi_yeast_slim/ontology.owl"),
                         validation_path=data_path("ppi_yeast_slim/valid.owl"),
                         testing_path=data_path("ppi_yeast_slim/test.owl"))
    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://4932" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes


class GOSubsumptionDataset(PathDataset):
    def __init__(self):
        super().__init__(data_path("go_subsumption/ontology.owl"),
                         validation_path=data_path("go_subsumption/valid.owl"),
                         testing_path=data_path("go_subsumption/test.owl"))

class FoodOnSubsumptionDataset(PathDataset):
    def __init__(self):
        super().__init__(data_path("foodon_subsumption/ontology.owl"),
                         validation_path=data_path("foodon_subsumption/valid.owl"),
                         testing_path=data_path("foodon_subsumption/test.owl"))


class GDADatasetV2(PathDataset):
    def __init__(self):
        super().__init__(data_path("gda2/ontology.owl"),
                         validation_path=data_path("gda2/valid.owl"),
                         testing_path=data_path("gda2/test.owl"))

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            diseases = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name and owl_name.split("/")[-1].isnumeric():
                    genes.add(owl_cls)
                if "OMIM_" in owl_name:
                    diseases.add(owl_cls)

            genes = OWLClasses(genes)
            diseases = OWLClasses(diseases)
            self._evaluation_classes = (genes, diseases)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"


class PPIHumanDataset(PathDataset):
    def __init__(self):
        super().__init__(data_path("ppi_human/ontology.owl"),
                         validation_path=data_path("ppi_human/valid.owl"),
                         testing_path=data_path("ppi_human/test.owl"))

    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://9606" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

class HPIDataset(PathDataset):

    def __init__(self):
        super().__init__(data_path("hpi/ontology.owl"),
                         validation_path=data_path("hpi/valid.owl"),
                         testing_path=data_path("hpi/test.owl"))

    @property
    def evaluation_classes(self):
        
        if self._evaluation_classes is None:
            genes = set()
            viruses = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                
                if "mowl.borg" in owl_name:
                    genes.add(owl_cls)
                if "NCBITaxon_" in owl_name:
                    viruses.add(owl_cls)

            genes = OWLClasses(genes)
            viruses = OWLClasses(viruses)
            self._evaluation_classes = (genes, viruses)

        return self._evaluation_classes

    @property
    def evaluation_object_property(self):
        return "http://mowl.borg/associated_with"
