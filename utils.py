__all__ = ['ExportAsDOT', 'toJSON', 'Summarize']

"""
Set of utility functions to display or convert Bayesian networks into other formats
"""

def ExportAsDOT( graph, output_path, show_interactions = True, **kwargs ):
    """
    Export the Bayesian Network to a graphviz *.dot format
    INPUT:  graph - the Bayesian Network that has had its beliefs computed, either through running BeliefPropagation or GetBeliefs
            output_path - path to the output *.dot file
            show_interactions - whether or not to print interactions or CPTs of each node
            **kwargs - any other arguments for each node. See dot language documentation for more details
    OUTPUT: a *.dot file that can be converted to a visual graph, e.g. dot -Tpdf foo.dot -o foo.pdf. Requires graphviz to be installed
    """

    out = open(output_path,"w")
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
    """
    Output a Bayesian network into a JSON file.
    INPUT:  graph - A Bayesian network with beliefs computed either through BeliefPropagation or GetBeliefs
            outfile - path to output JSON file
    OUTPUT: JSON file in the outfile
    """

    import json

    inverted_dict = dict( (v,k) for k,v in graph.Nodes.iteritems() )
    
    out_dict = { }
    out_dict['nodes'] = [ {'name': k} for k,v in graph.Nodes.iteritems()]
    for dic in out_dict['nodes']:
        n = dic['name']
        dic['beliefs'] = dict( (lvl, round(graph.Nodes[n].beliefs[lvl],4) ) for lvl in range(graph.Nodes[n].levels))
        dic['cpt'] = dict( (str(k), str(graph.Nodes[n].cpt[k]) ) for k in graph.Nodes[n].cpt)
    
    out_dict['links'] = [ {'source': find_index(out_dict['nodes'],i, inverted_dict), 'target': find_index(out_dict['nodes'], j, inverted_dict )} for i,j in graph.edges ]
    
    return json.dump(out_dict,open(outfile,'w') )



def Summarize( graph,interactions=False ):
    """
    Prints a summary of the graph data. Must have run either BeliefPropagation or GetBeliefs to update node beliefs before hand
    INPUT:  graph - a Bayesian network that has been calibrarted and beliefs computed
            interactions - whether or not to display the node interactions or CPT's. Default is False. 
    OUTPUT: text output summarizing interactions
    """
    for n, Node in graph.Nodes.iteritems():
        print "Node: "+n
        print "-_"*10
        if len(Node.cpt.keys() )>0:
            if interactions:
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
