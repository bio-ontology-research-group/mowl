from mowl.models.boxel.model import BoxEL


class BoxELGDA(BoxEL):
    """
    Example of BoxEL for gene-disease association prediction.

    Uses default negative sampling (all classes for gci2) inherited from BoxEL.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_gci_name = "gci2"

    def get_negative_sampling_config(self):
        return {
            "gci2": {"index_pool": "classes", "corrupt_column": 2}
        }
