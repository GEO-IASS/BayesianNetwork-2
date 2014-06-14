BayesianNetwork
===============

Construct and calibrate a Bayesian network from survey data. BayesianNetwork contains a set of routines for computing marginal probabilities in a Bayesian network using belief propagation.

## INSTALLATION

1. Unzip the source file and type:  python setup.py install

##### OR

1. Add the package path to python's system directory by:
```python
import sys
sys.path.append(PACKAGE_PATH_NAME)
```
2. Then import the Bayesian network routines with
```python
from BayesianNetwork.BayesNet import *
from BayesianNetwork.utils import *
```
3. Routines should now be in the python environment. See examples for further implementation details/ideas.

##### REQUIREMENTS

Requires numpy, and graphviz for graphical output.
pydot and IPython are optional.

