import random

from algorithms.ECOD import ECOD
from algorithms.IF import IsolationForest
from algorithms.ODIF import DeepIF
from algorithms.SGAE import SGAE


class AlgorithmFactory:
    def getAlgorithmFromName(algorithm_name):
        if algorithm_name == "IsolationForest":
            return IsolationForest()
        elif algorithm_name == "OptimizedDeepIF":
            return DeepIF(optimization=True)
        elif algorithm_name == "DeepIF":
            return DeepIF(optimization=False)
        elif algorithm_name == "ECOD":
            return ECOD()
        elif algorithm_name == "SGAE":
            return SGAE(seed=random.randint(0, 2 ** 32 - 1))