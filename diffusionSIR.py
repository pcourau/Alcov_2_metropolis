import pyAgrum as gum
import pyAgrum.lib.dynamicBN as gdyn
from TTForwardInference import TTForwardInference

import time
import math


class TimeIt:
  """
  A context manager for profiling
  """

  def __init__(self, msg: str):
    """Create the context manager that will end by printing the message and the time.

    Args:
        msg (str): the message to be printed at the end of the context
    """
    self._msg = msg
    self._elapsedTime = -1

  def __enter__(self):
    self.__startTime = time.time()
    return self

  def __exit__(self, *exc):
    self._elapsedTime = time.time() - self.__startTime
    print('[{}] finished in {} ms'.format(
      self._msg, round(self._elapsedTime * 1000, 2)))

  def duration(self) -> float:
    """a way to have access of the time since the start of the context manager

    Returns:
        float: the duration
    """
    return self._elapsedTime


class DiffusionSIR:
  def _getVar(var: str, individu: str, t: str):
    """naming rule for variables in the model (except 'dt')

    Args:
        var (str): s, QS, O (the variable name)
        individu (str): one item of the _people list
        t (str): the number of timeslice (0,...,n or t)

    Returns:
        str: the name
    """
    return f"{var}_{individu}_{t}"

  def _getS(individu: str, t: str):
    """naming rule for variable `State` in the model. 

    State \in {S,I1,...,In,R}

    Args:
        var (str): s, QS, O (the variable name)
        individu (str): one item of the _people list
        t (str): the number of timeslice (0,...,n or t)

    Returns:
        str: the name
    """
    return DiffusionSIR._getVar("S", individu, t)

  def _getQS(individu: str, t: str):
    """naming rule for variable `Qualitative State` in the model

    Qualitative State \in {S,I,R}

    Args:
        var (str): s, QS, O (the variable name)
        individu (str): one item of the _people list
        t (str): the number of timeslice (0,...,n or t)

    Returns:
        str: the name
    """
    return DiffusionSIR._getVar("QS", individu, t)

  def _getO(individu: str, t: str):
    """naming rule for variable `observation` in the model

    observation \in {0,1} 1='first day with symptom'

    Args:
        var (str): s, QS, O (the variable name)
        individu (str): one item of the _people list
        t (str): the number of timeslice (0,...,n or t)

    Returns:
        str: the name
    """
    return DiffusionSIR._getVar("O", individu, t)

  def __init__(self, people="ABC", nbstate=10, T=30):
    """Build a model

    Args:
        people (List[str], optional): List of individus. Defaults to "ABC".
        nbstate (int, optional): Number of days in state I. Defaults to 10.
        T (int, optional): Number of days in the model. Defaults to 30.
    """
    self._people = people
    self._nbstate = nbstate
    self._T = T

    self._motif = gum.BayesNet()

    self._motif.addCOUNT(gum.RangeVariable(
      "dt", "dt", 0, len(self._people)), 1)
    self._motif.addCOUNT(gum.RangeVariable(
      "d0", "d0", 0, len(self._people)), 1)

    for individu in self._people:
      S_0 = DiffusionSIR._getS(individu, "0")  # values S,1...nbstate,R
      S_t = DiffusionSIR._getS(individu, "t")  # values S,1...nbstate,R
      QS_0 = DiffusionSIR._getQS(individu, "0")  # 0=S/1=I/2=R
      QS_t = DiffusionSIR._getQS(individu, "t")  # 0=S/1=I/2=R
      O_t = DiffusionSIR._getO(individu, "t")

      self._motif.add(S_0, 2 + nbstate)
      self._motif.add(S_t, 2 + nbstate)
      self._motif.add(QS_0, 3)
      self._motif.add(QS_t, 3)

      self._motif.addArc(S_0, S_t)

      self._motif.addArc(S_0, QS_0)
      self._motif.addArc(S_t, QS_t)

      self._motif.addArc(QS_0, "d0")
      self._motif.addArc(QS_t, "dt")

      self._motif.addArc("d0", S_t)

      self._motif.cpt(S_0).fillWith([1] + [0] * (1 + nbstate))
      self._motif.cpt(QS_0).fillWith(
        [1, 0, 0] +
        [0, 1, 0] * nbstate +
        [0, 0, 1])
      self._motif.cpt(QS_t).fillWith(
        [1, 0, 0] +
        [0, 1, 0] * nbstate +
        [0, 0, 1])

      self._motif.add(O_t, 2)  # 0: no first symptom / 1:first symptom
      self._motif.addArc(QS_t, O_t)
      self._motif.addArc(QS_0, O_t)

    self._model = gdyn.unroll2TBN(self._motif, T)

  def updateParameters(self, eta0: float, eta: float, sigma: float,
                       dgamma: list = [0.01150603, 0.03916544, 0.07079709, 0.09876255, 0.11953800, 0.13225631,
                                       0.13753429, 0.13667998, 0.13120553, 0.12255478]):
    """ Updates CPTs using the parameters and the model from proposed in Alcov-2 project

    Args:
        eta0 (float): parameter (see Alvoc2)
        eta ([float]): parameter (see Alvoc2)
        sigma (float): parameter (see Alvoc2)
        dgamma (list, optional): parameter (see Alvoc2)
    """
    assert len(
      dgamma) == self._nbstate, f"Not the correct amount of data in dgamma (length {len(dgamma)}!={self._nbstate})"

    # building CPT(St|St-1,dt_1)
    cptS = []
    stillS = (1 - eta0)
    for d in range(len(self._people) + 1):
      cptS += [stillS] + [(1 - stillS) * dg for dg in dgamma] + [0]
      cptS += [0] * (self._nbstate + 1) + [1]
      for s in range(2, self._nbstate + 1):
        cptS += [0] * (s - 1) + [1] + [0] * (self._nbstate + 2 - s)
      cptS += [0] * (self._nbstate + 1) + [1]
      stillS *= (1 - eta)

    for t in range(1, self._T):
      for individu in self._people:
        self.model_cpt("S", individu, t).fillWith(cptS)

    # builiding CPT(Ot|QSt,QSt-1)
    cptO = [1, 0] + [1 - sigma, sigma] + [1, 0] * 7
    for t in range(1, self._T):
      for individu in self._people:
        self.model_cpt("O", individu, t).fillWith(cptO)

  def _createEvidenceFromStory(self, profile: list) -> dict:
    """Create the corresponding evidence dictionnary from a profile

    Args:
        profile (list): the profile as a list of (individu,day_of_symptom)

    Returns:
        dict: the corresponding evidence dictionnary ready for inference
    """
    dico = {}
    for pos in range(len(self._people)):
      for k in range(1, self._T):
        dico[DiffusionSIR._getO(self._people[pos], k)] = 0
    for indiv, day_of_symptom in profile:
      dico[DiffusionSIR._getO(indiv, day_of_symptom - 1)] = 1

    return dico

  def readStories(self, csvfilename: str) -> list:
    """ From a csv, create a list of list of evt

    Args:
        csvfilename (str): the filename to read

    Returns:
        the stories in the form ([evs],weight) sorted in ordre to optimize
        incremental inference
    """
    stories = []
    total_lines = 0
    cpt = dict()  # count the number of times a profile has appeared
    with open(csvfilename, "r") as csvfile:
      for nbr, line in enumerate(csvfile.readlines()):
        total_lines += 1
        if line.strip() == "":
          continue
        t = line.split(",")
        assert len(t) == len(
          self._people), f"Not correct amount of data in \n{csvfilename}:{nbr:4} {line}"
        story = []
        for pos, val in enumerate(t):
          val = val.strip()
          if val != "NA":
            story.append((self._people[pos], val))
        key = ",".join([str(k) for k in story])
        if key in cpt:
          inc = cpt[key][1]
        else:
          inc = 0
        cpt[key] = (story, inc + 1)

    for v in cpt.values():
      stories.append(v)
    stories.sort(key=lambda v: min(int(k[1]) for k in v[0]) if len(v[0]) > 0 else 0)
    stories.reverse()

    return stories

  def LL(self, stories: list, approx: bool, verbose: bool = False) -> float:
    """From a filename of evidence (see profile.csv), compute the log-likelihood.

    Args:
        csvfilename (str): the filename
        approx (bool): do we use TTinference or exact inference
        verbose (bool): with or without print

    Returns:
        float: the log-likelihood of the profile
    """
    LL = 0

    if approx:
      ie = gum.LazyPropagation(self._model)
    else:
      ie = TTForwardInference(self._model)

    ie.addTarget(f"d{self._T - 1}")

    dico = self._createEvidenceFromStory([])
    ie.setEvidence(dico)
    ie.makeInference()
    pevnull = ie.evidenceProbability()
    lpevnull = math.log(pevnull)
    for story, weight in stories:
      if len(story) > 0:
        for (indiv, val) in story:
          nameO = DiffusionSIR._getO(indiv, int(val) - 1)
          dico[nameO] = 1
          ie.chgEvidence(nameO, 1)
        ie.makeInference()
        x = weight * math.log(ie.evidenceProbability())
        if verbose:
          print(f"{nbr} : {weight}x{x}")
        LL += x
        for (indiv, val) in story:
          nameO = DiffusionSIR._getO(indiv, int(val) - 1)
          dico[nameO] = 0
          ie.chgEvidence(nameO, 0)
      else:
        if verbose:
          print(f"{nbr} : {weight}x{lpevnull}")
        LL += weight * lpevnull

    return LL, math.log1p(-pevnull)

  def model(self) -> gum.BayesNet:
    """git access to the underlying BN

    Returns:
        gum.BayesNet : the model as a (dynamic) Bayesian network
    """
    return self._model

  def model_cpt(self, var: str, individu: str, t: str):
    """ give direct access of a CPT using the naming rules.

    Args:
        var (str): S|QS
        individu (str): A,B,C,...
        t (str): number of timeslice (from 0)

    Returns:
        gum.Potential : the CPT of the variable in the underlyng BN
    """
    return self.model().cpt(DiffusionSIR._getVar(var, individu, t))

  def __str__(self):
    return f"""SIR diffusion
  * number of people: {len(self._people)}
  * number of Idays: {self._nbstate}
  * Horizon: {self._T}
  * Model : {self._model}"""

  def showModel(self, size="10"):
    """ use dynamicBN library from pyAgrum to show the model

    Args:
        size (str, optional): size of the figure. Defaults to "10".
    """
    import pyAgrum.lib.dynamicBN as gdyn

    gdyn.showTimeSlices(self._model, size=size)

  def followModel(self, vars, evs, dims=(10, 2)):
    import matplotlib.pyplot as plt
    import pyAgrum.lib.dynamicBN as gdyn

    plt.rcParams['figure.figsize'] = dims
    gdyn.plotFollowUnrolled(vars, self.model(), self._T, evs=evs)

  def followStory(self, story=[], dims=(5, 1)):
    vars = ["d"]
    for indiv in self._people:
      vars.append(f"S_{indiv}_")
    gdyn.plotFollowUnrolled(vars, self.model(), self._T, evs=self._createEvidenceFromStory(story))
