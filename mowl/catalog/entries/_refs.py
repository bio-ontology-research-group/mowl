"""Bibliographic references shared by several catalogue cards.

Keys follow ``docs/source/appendix/references.rst`` where the reference is listed there.
This module starts with an underscore, so the registry does not treat it as a card module.
"""

from mowl.catalog.card import Reference

# --- Analyses of several methods -------------------------------------------------------

BOURGAUX2024 = Reference(
    "bourgaux2024",
    "Bourgaux, C., Guimarães, R., Koudijs, R., Lacerda, V., Ozaki, A. (2024). Knowledge Base "
    "Embeddings: Semantics and Theoretical Properties. In Proceedings of KR 2024, 823-833.",
    "https://doi.org/10.24963/kr.2024/77")

LEEMHUIS2025 = Reference(
    "leemhuis2025",
    "Leemhuis, M., Kutz, O. (2025). Understanding the Expressive Capabilities of Knowledge "
    "Base Embeddings under Box Semantics. In Proceedings of NeSy 2025, PMLR 284:303-321.",
    "https://proceedings.mlr.press/v284/leemhuis25a.html")

LEEMHUIS2026 = Reference(
    "leemhuis2026",
    "Leemhuis, M., Kutz, O. (2026). Trustworthy Knowledge Base Embeddings: A Foundational "
    "Study of Box Semantics. Neurosymbolic Artificial Intelligence (preprint).",
    "https://neurosymbolic-ai-journal.com/node/905")

ECKHOFF1988 = Reference(
    "eckhoff1988",
    "Eckhoff, J. (1988). Intersection properties of boxes. Part I: An upper-bound theorem. "
    "Israel Journal of Mathematics 62(3):283-301.",
    "https://doi.org/10.1007/BF02783298")

CHEN2025 = Reference(
    "chen2025",
    "Chen, J., Mashkova, O., Zhapa-Camacho, F., Hoehndorf, R., He, Y., Horrocks, I. (2025). "
    "Ontology Embedding: A Survey of Methods, Applications and Resources. IEEE TKDE "
    "37(7):4193-4212.",
    "https://doi.org/10.1109/TKDE.2025.3559023")

MASHKOVA2026 = Reference(
    "mashkova2026",
    "Mashkova, O., Hoehndorf, R. (2026). A hierarchy of faithfulness criteria for knowledge "
    "base completion. arXiv:2609.27863.",
    "https://arxiv.org/abs/2609.27863")

GUTIERREZ2018 = Reference(
    "gutierrez2018",
    "Gutiérrez-Basulto, V., Schockaert, S. (2018). From Knowledge Graph Embedding to "
    "Ontology Embedding? An Analysis of the Compatibility between Vector Space "
    "Representations and Rules. In Proceedings of KR 2018.",
    "https://arxiv.org/abs/1805.10461")

LACERDA2024 = Reference(
    "lacerda2024",
    "Lacerda, V., Ozaki, A., Guimarães, R. (2024). Strong Faithfulness for ELH Ontology "
    "Embeddings. Transactions on Graph Data and Knowledge 2(3):2:1-2:29.",
    "https://doi.org/10.4230/TGDK.2.3.2")

FEDERICO2026_MULTIBOX = Reference(
    "federico2026a",
    "Federico, G., Leemhuis, M., Kutz, O. (2026). Introducing Multiboxes: Increasing the "
    "Trustworthiness of Ontology Embeddings. In Proceedings of JOWO 2026 (OSETAI workshop).",
    "https://foisconference.org/assets/pdfs/jowo/osetai-2.pdf")

FEDERICO2026_PRISM = Reference(
    "federico2026b",
    "Federico, G., Leemhuis, M., Kutz, O. (2026). Introducing Prism Embeddings: A New Family "
    "of Geometric Ontology Embeddings. In Proceedings of DL 2026, CEUR-WS Vol-4230.",
    "https://ceur-ws.org/Vol-4230/dl-paper32.pdf")

AHARONI2024 = Reference(
    "aharoni2024",
    "Aharoni, A. (2024). Pushing the BoxEL Envelope. MSc thesis, ILLC, University of "
    "Amsterdam.",
    "https://eprints.illc.uva.nl/id/document/13037")

KAZAKOV2014 = Reference(
    "kazakov2014",
    "Kazakov, Y., Krötzsch, M., Simančík, F. (2014). The Incredible ELK: From Polynomial "
    "Procedures to Efficient Reasoning with EL Ontologies. Journal of Automated Reasoning "
    "53(1):1-61.",
    "https://doi.org/10.1007/s10817-013-9296-3")

BAADER2005 = Reference(
    "baader2005",
    "Baader, F., Brandt, S., Lutz, C. (2005). Pushing the EL Envelope. In Proceedings of "
    "IJCAI 2005, 364-369.")

# --- Methods implemented in mOWL -------------------------------------------------------

KULMANOV2019 = Reference(
    "kulmanov2019",
    "Kulmanov, M., Liu-Wei, W., Yan, Y., Hoehndorf, R. (2019). EL Embeddings: Geometric "
    "Construction of Models for the Description Logic EL++. In Proceedings of IJCAI 2019, "
    "6103-6109.",
    "https://doi.org/10.24963/ijcai.2019/845")

PENG2020 = Reference(
    "peng2020",
    "Peng, X., Tang, Z., Kulmanov, M., Niu, K., Hoehndorf, R. (2022). Description Logic "
    "EL++ Embeddings with Intersectional Closure. arXiv:2202.14018.",
    "https://arxiv.org/abs/2202.14018")

XIONG2022 = Reference(
    "xiong2022",
    "Xiong, B., Potyka, N., Tran, T.-K., Nayyeri, M., Staab, S. (2022). Faithful Embeddings "
    "for EL++ Knowledge Bases. In The Semantic Web - ISWC 2022, LNCS 13489.",
    "https://doi.org/10.1007/978-3-031-19433-7_2")

JACKERMEIER2024 = Reference(
    "jackermeier2024",
    "Jackermeier, M., Chen, J., Horrocks, I. (2024). Dual Box Embeddings for the "
    "Description Logic EL++. In Proceedings of the ACM Web Conference 2024, 2250-2258.",
    "https://doi.org/10.1145/3589334.3645648")

YANG2025 = Reference(
    "yang2025",
    "Yang, H., Chen, J., Sattler, U. (2025). TransBox: EL++-closed Ontology Embedding. In "
    "Proceedings of the ACM Web Conference 2025.",
    "https://doi.org/10.1145/3696410.3714672")

FALCON2022 = Reference(
    "falcon2022",
    "Tang, Z., Hinnerichs, T., Peng, X., Zhang, X., Hoehndorf, R. (2022). FALCON: Sound and "
    "Complete Neural Semantic Entailment over ALC Ontologies. arXiv:2208.07628.",
    "https://arxiv.org/abs/2208.07628")

CHEN2020 = Reference(
    "chen2020",
    "Chen, J., Althagafi, A., Hoehndorf, R. (2021). Predicting candidate genes from "
    "phenotypes, functions and anatomical site of expression. Bioinformatics 37(6):853-860.",
    "https://doi.org/10.1093/bioinformatics/btaa879")

CHEN2020B = Reference(
    "chen2020b",
    "Chen, J., Hu, P., Jiménez-Ruiz, E., Holter, O. M., Antonyrajah, D., Horrocks, I. "
    "(2021). OWL2Vec*: embedding of OWL ontologies. Machine Learning 110:1813-1845.",
    "https://doi.org/10.1007/s10994-021-05997-6")

SMAILI2018 = Reference(
    "smaili2018",
    "Smaili, F. Z., Gao, X., Hoehndorf, R. (2018). Onto2Vec: joint vector-based "
    "representation of biological entities and their ontology-based annotations. "
    "Bioinformatics 34(13):i52-i60.",
    "https://doi.org/10.1093/bioinformatics/bty259")

SMAILI2019 = Reference(
    "smaili2019",
    "Smaili, F. Z., Gao, X., Hoehndorf, R. (2019). OPA2Vec: combining formal and informal "
    "content of biomedical ontologies to improve similarity-based prediction. "
    "Bioinformatics 35(12):2133-2140.",
    "https://doi.org/10.1093/bioinformatics/bty933")

PEROZZI2014 = Reference(
    "perozzi2014",
    "Perozzi, B., Al-Rfou, R., Skiena, S. (2014). DeepWalk: online learning of social "
    "representations. In Proceedings of KDD 2014.",
    "https://arxiv.org/abs/1403.6652")

GROVER2016 = Reference(
    "grover2016",
    "Grover, A., Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. In "
    "Proceedings of KDD 2016.",
    "https://arxiv.org/abs/1607.00653")

MIKOLOV2013 = Reference(
    "mikolov2013",
    "Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). Efficient Estimation of Word "
    "Representations in Vector Space. arXiv:1301.3781.",
    "https://arxiv.org/abs/1301.3781")

ALI2021 = Reference(
    "ali2021",
    "Ali, M., Berrendorf, M., Hoyt, C. T., Vermue, L., Sharifzadeh, S., Tresp, V., "
    "Lehmann, J. (2021). PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge "
    "Graph Embeddings. Journal of Machine Learning Research 22(82):1-6.",
    "https://jmlr.org/papers/v22/20-825.html")

ZHAPA2026 = Reference(
    "zhapa2026",
    "Zhapa-Camacho, F., Muhammed, N., Hoehndorf, R. Beyond similarity: inductive "
    "gene-disease associations as link prediction. Manuscript in preparation.")
