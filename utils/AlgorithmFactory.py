import random

from algorithms.IF import IsolationForest
from algorithms.ODIF import DeepIF

class AlgorithmFactory:
    def getAlgorithmFromName(algorithm_name):
        if algorithm_name == "IsolationForest":
            return IsolationForest()
        elif algorithm_name == "OptimizedDeepIF":
            return DeepIF(optimization=True)
        elif algorithm_name == "DeepIF":
            return DeepIF(optimization=False)