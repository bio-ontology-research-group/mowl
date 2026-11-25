"""The seven ontologies of the masked ABox revision experiment of DF-ALC."""

import os
import shutil
import urllib.request

from mowl.datasets.base import PathDataset, default_data_root

#: Where the files are read from: the (anonymised) repository of the code of the paper,
#: https://anonymous.4open.science/r/DF-ALC (training/ontologies, licence MPL-2.0). It is a
#: mirror of an anonymous submission and may disappear; ``base_url`` overrides it.
BASE_URL = "https://anonymous.4open.science/api/repo/DF-ALC/file/training/ontologies/"

#: name -> (file name, description)
ONTOLOGIES = {
    "family": ("Family.owl", "Family (instantiated families, incomplete knowledge)"),
    "family2": ("Family2.owl", "Family2 (Family with bridging definitions, domain and range)"),
    "glycordf": ("glycordf.glycordf.14.owl.xml", "GlycoRDF (BioPortal GLYCORDF, version 14)"),
    "nifdys": ("nifdys.neuroscience-information-framework-nif-dysfunction-ontlogy.14.owl.xml",
               "NIF Dysfunction (BioPortal NIFDYS, version 14)"),
    "nihss": ("nihss.national-institutes-of-health-stroke-scale-ontology.11.owl.xml",
              "NIH Stroke Scale (BioPortal NIHSS, version 11)"),
    "ontodm": ("ontodm-core.ontology-of-core-data-mining-entities.6.owl.xml",
               "OntoDM-core (BioPortal ONTODM-CORE, version 6)"),
    "sso": ("sso.syndromic-surveillance-ontology.1.owl.xml",
            "Syndromic Surveillance Ontology (BioPortal SSO, version 1)"),
}
NAMES = tuple(ONTOLOGIES)


class DFALCRevisionDataset(PathDataset):
    """One of the ontologies of the masked ABox revision experiment of Wu et al. (2022,
    arXiv:2211.12006, Table 1): Family, Family2, GlycoRDF, NIF-Dysfunction (Nifdys), NIHSS,
    OntoDM-core and SSO. The paper took the BioPortal ontologies with negation, disjunction
    or a universal quantifier and at least 100 ABox assertions; the files are those of the
    released code.

    The ontology is read as it is released (a single file, ABox and TBox together); the
    :class:`~mowl.evaluation.abox_revision.MaskedABoxRevisionRegime` removes the ABox from the
    training data and uses it as the grounding to revise. The files are downloaded at run
    time from the repository of the code (MPL-2.0; the BioPortal ontologies keep their own
    licences, see their BioPortal pages) and cached unchanged.

    :param name: One of :data:`NAMES`.
    :param data_root: Where to cache the files (default: :func:`default_data_root`).
    :param base_url: Override the download location.
    """

    def __init__(self, name="family", data_root=None, base_url=None):
        if name not in ONTOLOGIES:
            raise ValueError(f"name must be one of {NAMES}.")
        self.name = name
        root = default_data_root() if data_root is None else data_root
        self.directory = os.path.join(root, "dfalc_revision")
        filename = ONTOLOGIES[name][0]
        path = os.path.join(self.directory, filename)
        if not os.path.exists(path):
            os.makedirs(self.directory, exist_ok=True)
            partial = f"{path}.{os.getpid()}.part"
            # The host refuses the default urllib client name (HTTP 403).
            request = urllib.request.Request(
                (base_url or BASE_URL) + filename, headers={"User-Agent": "mowl"})
            with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out)
            os.replace(partial, path)
        super().__init__(path)
