#!/usr/bin/python

from BayesianNetwork.BayesianNetwork import *
from BayesianNetwork.utils import *
import pandas as pd


"""
Example loading data in from a CSV file
    
The variables can be multi-level or continuous. The program needs to know what interactions are expected.
These can be listed by the column names. Interactions are specified in a dictionary where the key is
the dependent variable and the value is a list of the independent variables.
    
Finally, to see the impact of additional knowledge, one can add evidence to the network. Each piece of evidence or
extra knowledge is expected to be in the form of a dictionary with the key = observed variable name and value = observed
probability distribution.
    
The output is saved in a *.png file
    
Beliefs are the marginal probabilities for the respective variable.
They are the probability that a variable takes a value given the info
present.    
"""

dat = pd.DataFrame.from_csv('from_CSV_file/example.csv').dropna().astype('int')

### list of interactions bewteen columns of data
### order is Depenedent variable: [list of independent variables]
ints = {'SecondLine_treatment': ['FirstLine_treatment','Response1L','GotMaint_12','Referrer', 'Academic','PatientVol'],
        'ThirdLine_treatment': ['SecondLine_treatment','Response2L','GotMaint_23','Referrer', 'Academic', 'PatientVol'],
        'FourthLine_treatment': ['ThirdLine_treatment', 'Response3L','GotMaint_34','Referrer', 'Academic', 'PatientVol']
        }

### load the graph with the data and the listed interactions.

Graph = BayesNet(data = dat, interactions=ints)
Graph.InitializeGraphMsgs()
Graph.BeliefPropagation()

### add evidence to the network and see how the marginal distribution
### for the variables in the network changes
### Evidence is labelled 'Variable observed': observed probability distribution

Combinations = [ {'FourthLine_treatment': [1,0,0,0]},
                 {'FirstLine_treatment': [1,0,0,0]},
                 {'FirstLine_treatment': [1,0,0,0], 'SecondLine_treatment': [0,1,0,0]}
                ]

file_name = 'from_CSV_file/example_%d.dot'

### loop through each of the cases and save to a *.dot file.
for i,c in enumerate(Combinations):
    print 'Adding evidence: ' +str(c)
    Graph.SetPrior(c)
    Graph.BeliefPropagation()
    ExportAsDOT(Graph, file_name%i, show_interactions=False, style = "filled, bold", penwidth = 1,
                            fillcolor = "white", fontname = "Courier", shape = "Msquare", fontsize=10)
    Graph.RemovePrior(c) ### remove the evidence, otherwise it accumulates in the network


