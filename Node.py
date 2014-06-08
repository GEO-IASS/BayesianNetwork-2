
from itertools import product
import numpy as np
import glob,re,json

__all__ = ['Node']

class Node():
    """
    Node object of a Bayesian network. A node has as attributes
        1) The number of levels
        2) A CPT or table of interactions describing how its dependent variable interact with it
        3) Beliefs, marginal probabilities for each level
    
    Other attributes it has are:
        1) Order, containing the order in which indices of the CPT are given
        2) parents, the parents of a given node
        3) children, the children of the node
        4) evidence, any observed evidence about the node
    """
    def __init__(self):
        self.levels = 0   ### number of levels/states each node can take
        self.order = []   ### order which indices are listed in the CPT/interaction
        self.cpt = { }    ### interactions/CPT describing how the dependent variable respondes to the independent variables
        self.beliefs = [] ### beliefs/probabilities about being in each state
        self.parents = []
        self.children = []
        self.evidence = []
    
    
    def WhichParent(self,p):
        """
        Finds out which set of indices belong to a parent, p
        """
        for i,k in enumerate(self.parents):
            if k==p: return i
    
    def MergeDown(self, down_msgs, excluding=None):
        """
        Performs a merge of messages coming from the parents of a node.

        INPUT:  down_msgs - set of down messages
                excluding - if messages from any node instance should be ignored
        OUTPUT: merged messages
        """
        return [self.cpt[key]*np.product([down_msgs[(k,self)][key[idx]] for idx,k in enumerate(self.parents) if k != excluding])
                for key in self.cpt.keys()]
    
    def MergeUp(self, up_msgs, excluding=None):
        """
        Performs a merge of messages coming from the children of a node.

        INPUT:  up_msgs - set of up messages
                excluding - if messages from any node instance should be ignored
        OUTPUT: merged messages
        """
        if len(self.evidence)>0: return self.evidence
        return np.product([ up_msgs[(self,k)] for k in self.children if k != excluding ],axis=0)
    
    
    def ComputeBeliefs(self, down_msgs, up_msgs):
        """
        Merge messages coming from parents and from children to form a belief (probability) estimate for each level.
        """
        msgs_up = self.MergeUp(up_msgs)
        
        if len(self.cpt.keys()) >0:
            marginalized_down_msgs = np.sum( self.MergeDown(down_msgs), axis=0)
        else:
            marginalized_down_msgs = [1.]*self.levels
        
        self.beliefs = np.multiply(msgs_up, marginalized_down_msgs)
        self.beliefs /= np.sum(self.beliefs)

