from unittest import TestCase
from matplotlib.patches import Circle, Rectangle
from tests.datasetFactory import FamilyDataset
from mowl.models import ELBE, ELEmbeddings, BoxSquaredEL
from mowl.visualization import ELEmVisualizer, ELBEVisualizer, BoxSquaredELVisualizer

_dataset = None


def _get_dataset():
    global _dataset
    if _dataset is None:
        _dataset = FamilyDataset()
    return _dataset


class TestELEmVisualizer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = ELEmbeddings(_get_dataset(), embed_dim=2)
        cls.model.train(epochs=1)
        cls.iri = list(cls.model.class_index_dict.keys())[0]

    def test_raises_on_wrong_embed_dim(self):
        with self.assertRaises(ValueError):
            ELEmVisualizer(ELEmbeddings(_get_dataset(), embed_dim=10))

    def test_patch_is_circle(self):
        self.assertIsInstance(ELEmVisualizer(self.model)._get_patch(self.iri), Circle)

    def test_patch_center_is_floats(self):
        cx, cy = ELEmVisualizer(self.model)._patch_center(self.iri)
        self.assertIsInstance(cx, float)
        self.assertIsInstance(cy, float)

    def test_unknown_entity_raises(self):
        with self.assertRaises(KeyError):
            ELEmVisualizer(self.model)._get_patch("http://does-not-exist/Foo")

    def test_savefig_without_plot_raises(self):
        with self.assertRaises(RuntimeError):
            ELEmVisualizer(self.model).savefig("unused.png")


class TestELBEVisualizer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = ELBE(_get_dataset(), embed_dim=2)
        cls.model.train(epochs=1)
        cls.iri = list(cls.model.class_index_dict.keys())[0]

    def test_raises_on_wrong_embed_dim(self):
        with self.assertRaises(ValueError):
            ELBEVisualizer(ELBE(_get_dataset(), embed_dim=10))

    def test_patch_is_rectangle(self):
        self.assertIsInstance(ELBEVisualizer(self.model)._get_patch(self.iri), Rectangle)

    def test_patch_center_is_floats(self):
        cx, cy = ELBEVisualizer(self.model)._patch_center(self.iri)
        self.assertIsInstance(cx, float)
        self.assertIsInstance(cy, float)


class TestBoxSquaredELVisualizer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = BoxSquaredEL(_get_dataset(), embed_dim=2)
        cls.model.train(epochs=1)
        cls.iri = list(cls.model.class_index_dict.keys())[0]

    def test_raises_on_wrong_embed_dim(self):
        with self.assertRaises(ValueError):
            BoxSquaredELVisualizer(BoxSquaredEL(_get_dataset(), embed_dim=10))

    def test_patch_is_rectangle(self):
        self.assertIsInstance(BoxSquaredELVisualizer(self.model)._get_patch(self.iri), Rectangle)

    def test_patch_center_is_floats(self):
        cx, cy = BoxSquaredELVisualizer(self.model)._patch_center(self.iri)
        self.assertIsInstance(cx, float)
        self.assertIsInstance(cy, float)
