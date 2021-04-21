import pyAgrum as gum
#from TenGeRine.TensorTrainInference.Tools import tt_product, marginalized, margSumIn, updateTensorTrain
#from TenGeRine.TTBN.TTBN import TTBN

from typing import Dict
from functools import wraps


def check_target(f):
  @wraps(f)
  def hollow_wrapper(self, *args, **kwds):
    if self._target is None:
      raise ValueError("Please add a target first!")
    return f(self, *args, **kwds)

  return hollow_wrapper


class TTForwardInference:
  def __init__(self, bn: gum.BayesNet, precision: float):
    if len(bn.connectedComponents()) > 1:
      raise ValueError("The bayes net should be connected for this inference")

    self._bn = bn

    # todo : could be optimized: many CPTs are repeated
    self._ttbn = TTBN(bn, precision=precision)
    jtg = gum.JunctionTreeGenerator()
    self._jt = jtg.junctionTree(self._ttbn)

    self._target = None
    self._var_to_clique = dict()

    self._obsolete_cliques = None

  def addTarget(self, target: str):
    if self._target is not None:
      raise ValueError(f"A target already exists : {self._target}")

    self._target = target

#     todo : find a clique that contains target
#     todo : populate a map namevar->clique (for evidence)

  @check_target
  def setEvidence(self, dico: Dict[str, int]):
    """

    :param dico:
    :return:
    """
    self._obsolete_cliques = None
    for k, v in dico:
      self._insertEvidence(k, v)

  @check_target
  def chgEvidence(self, k, v):
    """

    :param k:
    :param v:
    :return:
    """
    print(self._bn)
    print(k, v)


if __name__ == "__main__":
  model = gum.fastBN("X0->X1->X2->X3;X0->O0;X1->O1;X2->O2;X3->O3;")

  ie = TTForwardInference(model, 1e-2)
  ie.addTarget("x")
  ie.chgEvidence(1, 2)
