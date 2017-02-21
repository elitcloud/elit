import elit.structure
import unittest
from elit.test_structure import StructureTest
from elit.test_model import LabelMapTest


suiteModel = unittest.TestLoader().loadTestsFromTestCase(LabelMapTest)
suiteStructure = unittest.TestLoader().loadTestsFromTestCase(StructureTest)
unittest.TextTestRunner().run(suiteStructure)
