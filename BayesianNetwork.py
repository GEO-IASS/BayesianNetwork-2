
from itertools import product
import numpy as np
import glob,re,json


class Node():
    def __init__(self):
        self.levels = 0
        self.order = []
        self.cpt = { }
        self.beliefs = []
        self.parents = []
        self.children = []
        self.evidence = []
    
    
    def WhichParent(self,p):
        for i,k in enumerate(self.parents):
            if k==p: return i
    
    def MergeDown(self, down_msgs, excluding=None):
        return [self.cpt[key]*np.product([down_msgs[(k,self)][key[idx]] for idx,k in enumerate(self.parents) if k != excluding])
                for key in self.cpt.keys()]
    
    def MergeUp(self, up_msgs, excluding=None):
        
        if len(self.evidence)>0: return self.evidence
        return np.product([ up_msgs[(self,k)] for k in self.children if k != excluding ],axis=0)
    
    
    def ComputeBeliefs(self, down_msgs, up_msgs):
        msgs_up = self.MergeUp(up_msgs)
        
        if len(self.cpt.keys()) >0:
            marginalized_down_msgs = np.sum( self.MergeDown(down_msgs), axis=0)
        else:
            marginalized_down_msgs = [1.]*self.levels
        
        self.beliefs = np.multiply(msgs_up, marginalized_down_msgs)
        self.beliefs /= np.sum(self.beliefs)



class BayesNet():
    """ Bayes Net, inhereit from the networkx base class DiGraph, directed graphs
        extension of the directed graphs in networkx. Treating a Bayesian network as
        a directed graph, class contains methods to compute maringal probabilities
        using the belief propagation, message passing algorithm
    """
    
    def __init__(self, data = None, interactions = None, **attr):
        self.edges = []
        self.m_down = {} ### upstream messages container
        self.m_up = {} ### downstream messages container
        self.Nodes = {}
    
        if data is not None and interactions is not None:
            self.NodesFromCSV(data)
            self.InteractionsFromCSV(interactions,data)
    
    
    def NodesFromCSV( self, dat):
        print 'Loading node data from csv file....',
        for label in dat.keys():
            self.Nodes[label] = Node()
            self.Nodes[label].levels = max(dat[label])+1
        return 'Done.'
    
    def InteractionsFromCSV( self, interactions, dat):
        print 'Loading interaction information....',
        
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
        return 'Done.'

    def GraphFromAdjList(self, adj_list ):
        
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
        for u,v in self.edges:
            self.m_down[ (u, v) ] = np.random.rand( u.levels )
            self.m_up[ (u, v) ] = np.random.rand( u.levels )
            
            self.m_down[ (u,v) ] = self.__NormalizeMsg( self.m_down[(u,v)] )
            self.m_up[ (u,v) ] = self.__NormalizeMsg( self.m_up[(u,v)] )
    
    
    def BeliefPropagation(self, MaxIter=1000, tol=0.0001 ):
        print 'Running belief propagation algorithm to compute marginal probabilities...'
        for iteration in xrange(MaxIter):
            
            new_up = dict([ ((i,j) , self.__UpdateUpMsgs(i,j)) for i,j in np.random.permutation( self.edges ) ])
            new_down = dict([ ((i,j), self.__UpdateDownMsgs(i,j)) for i,j in np.random.permutation( self.edges ) ])
            
            if self.__MessageDifference(new_down, new_up) < tol:
                print 'Converged in %d steps. Beliefs computed'%iteration
                return
            else:
                self.m_up = new_up.copy()
                self.m_down = new_down.copy()
        print 'Did not converge in %d steps with tolerance %1.6f. Beliefs not computed.'%(iteration,tol)
        return
    
    def GetBeliefs(self):
        for i,node in self.Nodes.iteritems():
            node.ComputeBeliefs( self.m_down, self.m_up )
    
    
    def AddEvidence(self, node_state_dict):
        for k,v in node_state_dict.items():
            self.Nodes[k].evidence = v/np.sum(v)
    
    def RemoveEvidence(self,node):
        self.Nodes[node].evidence = []
    
    def add_edge(self, i, j):
        self.edges.append( (i,j) )
    
    
    
    def __UpdateUpMsgs(self, node_i, node_j):
        
        msgs_up = node_j.MergeUp(self.m_up)
        
        marginalized_down_msg = np.sum( np.multiply( msgs_up , node_j.MergeDown(self.m_down, excluding = node_i) ),axis=1 )
        new_msg = [ [elem for key,elem in zip( node_j.cpt.keys(), marginalized_down_msg ) if key[ node_j.WhichParent( node_i )] == ii ]for ii in range( node_i.levels)]
        
        return self.__NormalizeMsg( np.sum(new_msg,axis=1) )
    
    
    def __UpdateDownMsgs(self, node_i, node_j):
        
        msgs_up = node_i.MergeUp( self.m_up, excluding = node_j )
        
        if len(node_i.cpt.keys()) == 0:
            marginalized_down_msg = [1.0] * node_i.levels
        else:
            marginalized_down_msg = np.sum( node_i.MergeDown(self.m_down),axis=0)
        
        return self.__NormalizeMsg( np.multiply(msgs_up, marginalized_down_msg) )
    
    
    def __NormalizeMsg(self, msg):
        return msg/np.sum(msg)
    
    
    def __MessageDifference(self, down, up ):
        delta = np.sqrt(np.sum([ np.sum((down[(i,j)] - self.m_down[(i,j)])**2) + np.sum((up[(i,j)] - self.m_up[(i,j)])**2) for i,j in self.edges ]))
        return delta





def ExportAsDOT( graph, output_path, show_interactions = True, **kwargs ):
    
    out = open(output_path,"wb")
    out.write('digraph Bayes_Net {\n' )
    
    node_properties = {}
    
    header = "<<table border=\"0\" cellborder=\"0\" cellpadding=\"3\" bgcolor=\"white\">"
    VarName = "<tr><td bgcolor=\"black\" align=\"center\" colspan=\"2\"><font color=\"white\">Var Name {VAR}</font></td></tr>"
    Interaction_header = "<tr><td align=\"left\" colspan=\"2\"><font color=\"black\">Interactions</font></td></tr>"
    Interaction =  "<tr><td align=\"left\" port=\"r0\">{KEY} &#58; {VALUE}</td></tr>"
    Beliefs_header = "<tr><td align=\"left\" colspan=\"2\" bgcolor=\"grey\"><font color=\"black\">Beliefs</font></td></tr>"
    Beliefs =  "<tr><td align=\"left\" port=\"r0\"> {LEVEL} &#58; {VALUE}</td></tr>"
    footer = "</table>>"
    
    for n, Node in graph.Nodes.items():
        
        s = VarName.format(VAR=str(n))
        i  = ''.join([Interaction.format(KEY=k,VALUE=b) for k,b in Node.cpt.iteritems()])
        b  = ''.join([Beliefs.format(LEVEL=k,VALUE=round(v,4)) for k,v in enumerate(Node.beliefs)])
        
        
        if show_interactions:
            node_properties[Node] = {'label':  header + s + Interaction_header + i + Beliefs_header + b + footer}
        else:
            node_properties[Node] = {'label':  header + s + Beliefs_header + b + footer}
        
        for k,v in kwargs.items():
            node_properties[Node][k] = '\"' + str(v) + '\"'
        
        node_attr = ', '.join([str(key)+'=%s' for key in node_properties[Node]] )
        attr = node_attr%tuple(node_properties[Node][key] for key in node_properties[Node] )
        out.write('%s ['%Node+attr+'];\n')
    
    for i,j in graph.edges:
        out.write('%s -> %s;\n'%(i,j) )
    
    
    out.write('}')
    out.close()
    return



def find_index(json_dict, i, inverted_dict):
    for elem in json_dict:
        if elem['name'] == inverted_dict[i]: return json_dict.index(elem)
    return None



def toJSON(graph,outfile):
    inverted_dict = dict( (v,k) for k,v in graph.Nodes.iteritems() )
    
    out_dict = { }
    out_dict['nodes'] = [ {'name': k} for k,v in graph.Nodes.iteritems()]
    for dic in out_dict['nodes']:
        n = dic['name']
        dic['beliefs'] = dict( (lvl, round(graph.Nodes[n].beliefs[lvl],4) ) for lvl in range(graph.Nodes[n].levels))
        dic['cpt'] = dict( (str(k), str(graph.Nodes[n].cpt[k]) ) for k in graph.Nodes[n].cpt)
    
    out_dict['links'] = [ {'source': find_index(out_dict['nodes'],i, inverted_dict), 'target': find_index(out_dict['nodes'], j, inverted_dict )} for i,j in graph.edges ]
    
    return json.dump(out_dict,open(outfile,'w') )



def Summarize( graph ):
    for n, Node in graph.Nodes.iteritems():
        print "Node: "+n
        print "-_"*10
        if len(Node.cpt.keys() )>0:
            print "Node Interactions: "
            print "Order: "+str(Node.order)
            for k,v in Node.cpt.iteritems():
                print str(k)+"\t"+str(v)
            print "-"*20
        print "Node Beliefs: "
        for k,v in enumerate(Node.beliefs):
            print str(k) + ":\t%1.3f"%v
        
        print "_-"*10
        print ""
    return
