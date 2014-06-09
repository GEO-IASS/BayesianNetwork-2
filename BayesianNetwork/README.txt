BayesianNetwork 
---------------
BayesianNetwork contains a set of routines for computing marginal probabilities in a Bayesian network using belief propagation.


USING
-----
To create a Bayesian network and compute the marginal probabilites of the nodes in the network:

1) Add the package path to python's system directory by:
	>>> import sys
	>>> sys.path.append(//sys_op/apps/METHODS/Python

2) Then import the Bayesian network routines with
	>>> from BayesianNetwork.BayesNet import *
	>>> from BayesianNetwork.utils import *

3) Routines should now be in the python environment. See examples for further implementation details/ideas.

REQUIREMENTS
------------

Requires numpy, and graphviz for graphical output.
pydot and IPython are optional.

Additional help:
contact bhsu@nationalanalysts.com