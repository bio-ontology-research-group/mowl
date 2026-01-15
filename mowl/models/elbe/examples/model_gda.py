from mowl.models import ELBE


class ELBEGDA(ELBE):
    """
    Example of ELBE for gene-disease associations prediction.

    Uses default negative sampling (all classes for gci2) and MSE loss
    inherited from ELBE.
    """

    def get_negative_sampling_config(self):
        """Only do negative sampling for gci2."""
        return {
            "gci2": {"index_pool": "classes", "corrupt_column": 2}
        }

