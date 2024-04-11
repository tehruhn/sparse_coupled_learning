import numpy as np
from LinearNetwork import LinearNetwork
from LinearNetworkSolver import LinearNetworkSolver

np.random.seed(42)

def SquareGrid(a, b, Periodic=False): # construct a square grid
    NN = a*b
    ys = np.arange(b) - b//2 + 0.5
    xs = np.arange(a) - a//2 + 0.5
    xs = -xs

    EI = []
    EJ = []
    for i in range(b-1):
        for j in range(a-1):
            EI.append(i*a + j)
            EJ.append(i*a + j + 1)
            EI.append(i*a + j)
            EJ.append((i+1)*a + j)
        EI.append(i*a + a-1)
        EJ.append((i+1)*a + a-1)
    for j in range(a-1):
            EI.append((b-1)*a + j)
            EJ.append((b-1)*a + j + 1)
    NE0 = len(EI)

    if Periodic:
        for j in range(a):
            EI.append(j)
            EJ.append((b-1)*a + j)

        for i in range(b):
            EI.append(i*a)
            EJ.append(i*a + a-1)

    EI = np.array(EI)
    EJ = np.array(EJ)
    NE = len(EI)

    print(NN,NE)
    return NN, NE, EI, EJ

def generate_node_and_edge_data(sources, targets, 
                                sourceedges, targetedges):
    np.random.seed(42)
    NodeData = np.random.randn(sources)
    OutNodeData = np.random.randn(targets) * 0.3
    EdgeData = np.random.randn(sourceedges)
    OutEdgeData = np.random.randn(targetedges) * 0.3
    return (NodeData.reshape(1, -1), OutNodeData.reshape(1, -1), 
            EdgeData.reshape(1, -1), OutEdgeData.reshape(1, -1))

if __name__ == "__main__":
    NN, NE, EI, EJ = SquareGrid(3, 3, False)
    graph_dict = {'NN':NN, "NE":NE, "EI":EI, "EJ":EJ}
    print(graph_dict)
    
    sources = 2
    targets = 1
    sourceedges = 0
    targetedges = 0

    trials = 1
    nTaskTypes = 1
    nTasksPerType = 1
    eta = 1.e-3
    lr = 0.05
    Steps = 15000

    tri, trt, tei, tet = generate_node_and_edge_data(sources, targets, 
                                                     sourceedges, targetedges)
    

    GroundNodes = np.array([NN-1])

    NodeList = np.random.choice(range(NN-1), size=sources+targets, replace=False)
    EdgeList = np.random.choice(range(NE), size=sourceedges+targetedges, replace=False)
    SourceNodes = NodeList[:sources]
    TargetNodes = NodeList[sources:]
    SourceEdges = EdgeList[:sourceedges]
    TargetEdges = EdgeList[sourceedges:]

    linNet = LinearNetwork(graph_dict)
    solver = LinearNetworkSolver(linNet)

    print(SourceNodes)
    print(TargetNodes)
    print(GroundNodes)
    print(tri.shape)
    print(trt.shape)

    K, costs = solver.perform_trial(source_nodes=SourceNodes, 
                                    target_nodes=TargetNodes,
                                    ground_nodes=GroundNodes,
                                    in_node=tri,
                                    out_node=trt,
                                    lr=0.05,
                                    steps=15000,
                                    debug=True,
                                    every_nth=100,
                                    init_strategy="ones"
                                    )
