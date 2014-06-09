from BayesianNetwork.BayesianNetwork import *
from BayesianNetwork.utils import *
import pandas as pd

Data = pd.DataFrame.from_csv('from_CSV_file/BayesNet_datav2.csv')
Interactions = {'FirstLine': ['Academic', 'PatientVol', 'Referrer'],
                'SecondLine': ['FirstLine','ReasonDiscont_1L', 'GotMaint_12','Response1L'],
                'ThirdLine': ['SecondLine', 'ReasonDiscont_2L', 'Response2L','TreatmentBreak_23'],
                'FourthLine': ['ThirdLine', 'ReasonDiscont_3L', 'Response3L', 'TreatmentBreak_34']
                }

Graph = BayesNet(data = Data, interactions = Interactions)
Graph.InitializeGraphMsgs()
Graph.BeliefPropagation()

ExportAsDOT(Graph,'from_CSV_file/hello.dot', show_interactions=False, style = "filled, bold", penwidth = 1, fillcolor = "white", fontname = "Courier", shape = "Msquare",fontsize=10)



from IPython.display import Image
import pydot

g = pydot.graph_from_dot_data(open('from_CSV_file/hello.dot','r').read())
Image(g.create_png())




