"""References of the training and evaluation regime cards written for the classification in
``docs/source/methods/training_and_evaluation.rst`` (the leading underscore keeps this
module out of the regime registry)."""

from mowl.catalog.card import Reference
from mowl.catalog.entries._refs import (  # noqa: F401
    KULMANOV2019, PENG2020, XIONG2022, JACKERMEIER2024, YANG2025, FALCON2022, CHEN2025,
    MASHKOVA2026, AHARONI2024, CHEN2020B, SMAILI2018, SMAILI2019, MIKOLOV2013)
from mowl.catalog.entries.elkem import SINGH2026  # noqa: F401
from mowl.catalog.entries.regd import YANG2025_REGD  # noqa: F401
from mowl.catalog.entries.bcbox import ZHANG2022, DASGUPTA2020  # noqa: F401
from mowl.benchmarks.entries._refs import ZHAPA2023, MONDAL2021  # noqa: F401

MOHAPATRA2021 = Reference(
    "mohapatra2021",
    "Mohapatra, B., Bhatia, S., Mutharaju, R., Srinivasaraghavan, G. (2021). Why Settle for "
    "Just One? Extending EL++ Ontology Embeddings with Many-to-Many Relationships. "
    "arXiv:2110.10555.",
    "https://arxiv.org/abs/2110.10555")

MASHKOVA2024NESY = Reference(
    "mashkova2024nesy",
    "Mashkova, O., Zhapa-Camacho, F., Hoehndorf, R. (2024). Enhancing Geometric Ontology "
    "Embeddings for EL++ with Negative Sampling and Deductive Closure Filtering. In "
    "Neural-Symbolic Learning and Reasoning (NeSy 2024). arXiv:2405.04868.",
    "https://arxiv.org/abs/2405.04868")

MASHKOVA2024DELE = Reference(
    "mashkova2024dele",
    "Mashkova, O., Zhapa-Camacho, F., Hoehndorf, R. (2024). DELE: Deductive EL++ Embeddings "
    "for Knowledge Base Completion. Neurosymbolic Artificial Intelligence, "
    "arXiv:2411.01574.",
    "https://arxiv.org/abs/2411.01574")

HE2024HIT = Reference(
    "he2024hit",
    "He, Y., Yuan, Z., Chen, J., Horrocks, I. (2024). Language Models as Hierarchy "
    "Encoders. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024). "
    "arXiv:2401.11374.",
    "https://arxiv.org/abs/2401.11374")

YANG2025ONT = Reference(
    "yang2025ont",
    "Yang, H., Chen, J., He, Y., Gao, Y., Horrocks, I. (2025). Language Models as Ontology "
    "Encoders. In The Semantic Web - ISWC 2025. arXiv:2507.14334.",
    "https://arxiv.org/abs/2507.14334")

LOURENCO2026 = Reference(
    "lourenco2026",
    "Lourenço, B. F., Morgan, H., Ozaki, A., Pavlović, A., Sallinger, E. (2026). BoxLitE: A "
    "Faithful Knowledge Base Embedding Based on Convex Optimization. In Proceedings of KR "
    "2026. arXiv:2605.23937.",
    "https://arxiv.org/abs/2605.23937")

GARG2019 = Reference(
    "garg2019",
    "Garg, D., Ikbal, S., Srivastava, S. K., Vishwakarma, H., Karanam, H., Subramaniam, "
    "L. V. (2019). Quantum Embedding of Knowledge for Reasoning. In Advances in Neural "
    "Information Processing Systems 32 (NeurIPS 2019).")

HE2024ACONE = Reference(
    "he2024acone",
    "He, Y., Hernández, D., Nayyeri, M., Xiong, B., Zhu, Y., Kharlamov, E., Staab, S. "
    "(2024). Generating SROI- Ontologies via Knowledge Graph Query Embedding Learning. In "
    "Proceedings of ECAI 2024. arXiv:2407.09212.",
    "https://arxiv.org/abs/2407.09212")

REN2020 = Reference(
    "ren2020",
    "Ren, H., Hu, W., Leskovec, J. (2020). Query2box: Reasoning over Knowledge Graphs in "
    "Vector Space using Box Embeddings. In Proceedings of ICLR 2020.",
    "https://arxiv.org/abs/2002.05969")

ZHANG2021CONE = Reference(
    "zhang2021cone",
    "Zhang, Z., Wang, J., Chen, J., Ji, S., Wu, F. (2021). ConE: Cone Embeddings for "
    "Multi-Hop Reasoning over Knowledge Graphs. In Advances in Neural Information "
    "Processing Systems 34 (NeurIPS 2021).",
    "https://arxiv.org/abs/2110.13715")

CHARPENAY2024 = Reference(
    "charpenay2024",
    "Charpenay, V., Schockaert, S. (2024). Capturing Knowledge Graphs and Rules with "
    "Octagon Embeddings. In Proceedings of IJCAI 2024. arXiv:2401.16270.",
    "https://arxiv.org/abs/2401.16270")

VENDROV2016 = Reference(
    "vendrov2016",
    "Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016). Order-Embeddings of Images and "
    "Language. In Proceedings of ICLR 2016.",
    "https://arxiv.org/abs/1511.06361")
