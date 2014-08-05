from BayesianNetwork.BayesNet import *
from BayesianNetwork.utils import *
from IPython.display import Image
import pydot


def RunManualExample( example ):
    """ 
    Example use of the Bayesian Network library in manual mode
    Graph data is stored in an adjaceny matrix, 'graph.txt' and 
    the CPT's for each node (if it depends on others) are placed in the 
    'interactions/' folder which are labeled {NAME OF NODE}.txt
        
    Expected output from these examples can be found in the examples folder.
    Examples are:
    1) chain - Modelling P(1,2,3) = P( 3 | 2) P(2 | 1) P(1)
    2) sprinkler -  See Wikipedia article on Bayesian networks.
                    '3' = 'Grass Wet'
                    '2' = 'Sprinkler'
                    '1' = 'Rain'
    3) tree - Modelling P(1,2,3,4,5) = P(5|3) P(4|3) P(3|1,2) P(1) P(2)
    4) v - Modelling P(1,2,3) = P(3|1,2) P(1) P(2)
        
        
    Beliefs are the marginal probabilities for the respective variable. 
    They are the probability that a variable takes a value given the info
    present.
    """
    
    G = BayesNet()
    G.GraphFromAdjList( 'manual_specification/%s/graph.txt'%example )
    G.InteractionsFromFolder( 'manual_specification/%s/interactions/'%example )
    G.InitializeGraphMsgs()
    G.BeliefPropagation()
    G.GetBeliefs()

    ### Calibrate the graph without any additional evidence
    ExportAsDOT(G,'hello.dot', style = "filled, bold", penwidth = 1, fillcolor = "white", fontname = "Courier", shape = "Msquare",fontsize=10)

    
    ### Add information to the Bayesian network
    ### make node '3' be in the state 0
    G.SetPrior( {'3': [1.,0.]} )
    G.BeliefPropagation()
    G.GetBeliefs()
    ExportAsDOT(G,'hello.dot', style = "filled, bold", penwidth = 1, fillcolor = "white", fontname = "Courier", shape = "Msquare",fontsize=10)

    ### Display the Bayesian network in the IPython Notebook
    g = pydot.graph_from_dot_data(open('hello.dot','r').read())
    Image(g.create_png())

