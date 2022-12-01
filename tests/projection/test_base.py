from mowl.projection.base import ProjectionModel
from unittest import TestCase


class TestBase(TestCase):

    def test_raise_not_implemented_error_project_method(self):
        """This should check if project method raises NotImplementedError"""
        projector = ProjectionModel()
        self.assertRaises(NotImplementedError, projector.project, None)
