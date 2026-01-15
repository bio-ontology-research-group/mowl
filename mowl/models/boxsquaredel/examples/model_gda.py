from mowl.models import BoxSquaredEL


class BoxSquaredELGDA(BoxSquaredEL):
    """
    Example of BoxSquaredEL for gene-disease associations prediction.

    Uses default negative sampling (all classes for gci2) and direct loss
    with regularization inherited from BoxSquaredEL.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_gci_name = "gci2"

    def get_negative_sampling_config(self):
        """Only do negative sampling for gci2."""
        return {
            "gci2": {"index_pool": "classes", "corrupt_column": 2}
        }


# Backward compatibility alias
ELBoxGDA = BoxSquaredELGDA

