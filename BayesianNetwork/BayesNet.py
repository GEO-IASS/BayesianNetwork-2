from itertools import product
import numpy as np
import glob,re,json

from .Node import *

__all__ = ['BayesNet']

class BayesNet():
    """
    Bayes Net object. A Bayesian network consists of a set of Nodes and a set of directed edges. On each edge is 
    associated an up message (a message going 'upstream' against the direction of the edge) and a down edge (a 
    message going 'downstream' with the direction of the edge). 

    The object has the attributes:
        1) edges
        2) set of downstream messages living on the edges
        3) set of upstream messages living on the edges
        4) set of Nodes 
    
    Messages are updated using belief propagation algorithm.
    """
    def __init__(self, data = None, interactions = None, **attr):
        self.edges = []  ### edges between node instances
        self.m_down = {} ### upstream messages container
        self.m_up = {}   ### downstream messages container
        self.Nodes = {}  ### dictionary containing node names as keys and node instances as values
    
        if data is not None and interactions is not None:
            self.NodesFromCSV(data)
            self.InteractionsFromCSV(interactions,data)
        return
    
    def NodesFromCSV( self, dat):
        """
        Import the nodes from a *.csv file.

        INPUT:  dat - a pandas data frame
        OUTPUT: a Bayes net where the nodes are labelled by the columm names of dat.
        """
        print 'Loading node data from csv file....'
        for label in dat.keys():
            self.Nodes[label] = Node()
            self.Nodes[label].levels = max(dat[label])+1
        return
        
    def InteractionsFromCSV( self, interactions, dat):
        """
        Import the interactions from a *csv file. 

        INPUT:  interactions - a dictionary. key=dependent node, value=list of nodes which the key is dependent on. Names are from the column names of dat
                dat - a panadas data frame of the relevant data
        OUTPUT: a BayesNet with the interactions 
        """
        print 'Loading interaction information....'
        for node_k, nodes in interactions.items():
            self.Nodes[node_k].order = nodes
            self.Nodes[node_k].parents = [self.Nodes[str(k)] for k in self.Nodes[node_k].order]
            
            for n in nodes:
                self.add_edge( self.Nodes[n], self.Nodes[node_k] )
        
            for comb in product(*[range(self.Nodes[v].levels) for v in self.Nodes[node_k].order]):
                condition = ' & '.join(['(dat[\"%s\"]==%d)'%(labs,i)  for i,labs in zip(comb, self.Nodes[node_k].order) ])
                d = dat[eval(condition)]
                if len(d)==0:
                    self.Nodes[node_k].cpt[comb] = np.array([0.]*self.Nodes[node_k].levels)
                else:
                    self.Nodes[node_k].cpt[comb] = np.array([1.*sum(d[node_k]==lvl)/len(d) for lvl in xrange(self.Nodes[node_k].levels) ])

        for k,v in self.Nodes.iteritems():
            self.Nodes[k].children = [ j for i,j in self.edges if i == self.Nodes[k] ]
    

    def GraphFromAdjList(self, adj_list ):
        """
        Import a graph from an adjaceny list.
        """
        for line in open(adj_list,'r').readlines():
            m = re.findall('[0-9]',line)
            if not m: continue
            else:
                for n in m:
                    if n in self.Nodes: continue
                    else: self.Nodes[n] = Node()
                self.add_edge(self.Nodes[m[0]], self.Nodes[m[1]])
        
        for k,v in self.Nodes.iteritems():
            self.Nodes[k].children = [ j for i,j in self.edges if i == self.Nodes[k] ]



    def InteractionsFromFolder( self, folder):
        """
        Loads interactions from a folder. Interactions for a given node are labeled as {Node Name}.txt and the order of the 
        indices describing the interactions are specified by the field ORDER in {Node Name}.txt, e.g. ORDER=1 2 means the indicies listed in the CPT 
        are ordered by (node 1 labels) followed by (node 2 labels). If a node does not have any dependencies, it still needs an interaction file which 
        doesn't contain a CPT 

        INPUT: folder - folder containing the {Node name}.txt interactions 
        OUTPUT: a graph that now has interactions loaded
        """

        for files in glob.glob(folder+'*.txt'):
            node = files.split('/')[-1][0]
            
            for line in open(files,'r').readlines():
                levels = re.findall('LEVELS=[0-9]*',line)
                order  = re.findall('ORDER=[0-9 ]*',line)
                inter  = re.findall('[0-9 ]*: [0-9 .,]*',line)
                
                if levels:
                    self.Nodes[node].levels = int(levels[0].split('=')[1])
                if order:
                    self.Nodes[node].order = np.array(order[0].split('=')[1].split(' ')).astype('int')
                    self.Nodes[node].parents = [self.Nodes[str(k)] for k in self.Nodes[node].order]
                if inter:
                    dat = inter[0].split(':')
                    inputs = np.array(dat[0].split(' ') ).astype('int')
                    interactions = dat[1]
                    self.Nodes[node].cpt[ tuple(inputs) ] = np.array(interactions.split(',')).astype('float')
    
    
    def InitializeGraphMsgs(self):
        """
        Initialize the Bayesian network messages to random values. 
        """

        for u,v in self.edges:
            self.m_down[ (u, v) ] = np.random.rand( u.levels )
            self.m_up[ (u, v) ] = np.random.rand( u.levels )
            
            self.m_down[ (u,v) ] = self.__NormalizeMsg( self.m_down[(u,v)] )
            self.m_up[ (u,v) ] = self.__NormalizeMsg( self.m_up[(u,v)] )
    
    
    def BeliefPropagation( self, MaxIter=1000, tol=1e-4 ):
        """
        Update messages iteratively for MaxIter number of times till the change in messages is less than the tolerance tol. 
        On each iteration, the algorithm goes through the edges of the graph randomly and 1) updates all the downstream messages
        using the messages from the previous iteration 2) updates all the upstream messages using messages from the previous iteration.
        The converged set of messages are used to compute the node beliefs.

        Note: this can be run on loopy graphs as a heuristic.
        """
        ### TODO: parallel implemntation. Take as inputs a set of workers to map to.....
        print 'Running belief propagation algorithm to compute marginal probabilities...'
        for iteration in xrange(MaxIter):
            
            new_up = dict([ ((i,j) , self.__UpdateUpMsgs(i,j)) for i,j in np.random.permutation( self.edges ) ])
            new_down = dict([ ((i,j), self.__UpdateDownMsgs(i,j)) for i,j in np.random.permutation( self.edges ) ])
            
            if self.__MessageDifference(new_down, new_up) < tol:
                self.GetBeliefs()
                print 'Converged in %d steps. Beliefs computed'%iteration
                return
            else:
                self.m_up = new_up.copy()
                self.m_down = new_down.copy()
        print 'Did not converge in %d steps with tolerance %1.6f. Beliefs not computed.'%(iteration,tol)
        return
    

    def GetBeliefs( self ):
        """
        Takes a set of messages on a Bayesian network and gets each node to compute its beliefs.

        OUTPUT: A Bayesian network with updated beliefs for the nodes of the network
        """

        for i,node in self.Nodes.iteritems():
            node.ComputeBeliefs( self.m_down, self.m_up )
    
    
    def SetPrior(self, mode_state_dict):
        """
        Set a prior distribution for a nodes in a Bayesian network

        INPUT: node_state_dict - dictionary of nodes that have evidence. keys=node names, values = probability distributions.
        OUTPUT: A Bayesian network with nodes with a priori distributions set
        """

        for k,v in node_state_dict.items():
            self.Nodes[k].evidence = v/np.sum(v)
    
    def RemovePrior(self, node):
        """
        Removes prior distribution from a node.

        INPUT: node - node name 
        """
        for n in node.keys():
            self.Nodes[n].evidence = []
    
    def add_edge(self, i, j):
        """
        Allows one to add an edge.

        INPUT:  i - Node instance for node i
                j - Node innstance for node j
        OUTPUT: a graph with a directed edge going from node i to node j
        """
        self.edges.append( (i,j) )
    
    
    
    def __UpdateUpMsgs(self, node_i, node_j):
        """
        Update an upstream message. On a direct edge i -> j this is the message j ->i 

        INPUT: node_i,node_j - node instances
        OUTPUT: a normalized new message from j->i
        """
        
        msgs_up = node_j.MergeUp(self.m_up)
        
        marginalized_down_msg = np.sum( np.multiply( msgs_up , node_j.MergeDown(self.m_down, excluding = node_i) ),axis=1 )
        new_msg = [ [elem for key,elem in zip( node_j.cpt.keys(), marginalized_down_msg ) if key[ node_j.WhichParent( node_i )] == ii ]for ii in range( node_i.levels)]
        
        return self.__NormalizeMsg( np.sum(new_msg,axis=1) )
    
    
    def __UpdateDownMsgs(self, node_i, node_j):
        """
        Update a downstream message. On a directed edge i-> j this is the message i->j

        INPUT: node_i,node_j - node instances
        OUTPUT: a normalized new message from i->j
        """
        msgs_up = node_i.MergeUp( self.m_up, excluding = node_j )
        
        if len(node_i.cpt.keys()) == 0:
            marginalized_down_msg = [1.0] * node_i.levels
        else:
            marginalized_down_msg = np.sum( node_i.MergeDown(self.m_down),axis=0)
        
        return self.__NormalizeMsg( np.multiply(msgs_up, marginalized_down_msg) )
    
    
    def __NormalizeMsg(self, msg):
        """
        Normalize the message. Default is for probability vectors to sum to one, but can change if messages are not probabilities, i.e. log probs
        """
        return msg/np.sum(msg)
    
    
    def __MessageDifference(self, down, up ):
        """
        Computing the message differences
        """
        delta = np.sqrt(np.sum([ np.sum((down[(i,j)] - self.m_down[(i,j)])**2) + np.sum((up[(i,j)] - self.m_up[(i,j)])**2) for i,j in self.edges ]))
        return delta


