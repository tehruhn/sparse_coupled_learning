import numpy as np
from numpy import *
from numpy.random import *
from scipy.linalg import *

np.random.seed(42)

np.random.seed(42)

def SquareGrid(a, b, Periodic=False): # construct a square grid
    NN = a*b
    ys = arange(b) - b//2 + 0.5
    xs = arange(a) - a//2 + 0.5
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

    EI = array(EI)
    EJ = array(EJ)
    NE = len(EI)
    return NN, NE, EI, EJ

# @numba.jit()
def SparseIncidenceConstraintMatrix(SourceNodes, SourceEdges, TargetNodes, TargetEdges, GroundNodes, NN, EI, EJ):
    NE = len(EI)
    dF = []
    xF = []
    yF = []
    dC = []
    xC = []
    yC = []
    nc = NN
    nc2 = NN
    for i in range(len(GroundNodes)):
        dF.append(1.)
        xF.append(GroundNodes[i])
        yF.append(nc+i)
        dF.append(1.)
        xF.append(nc+i)
        yF.append(GroundNodes[i])

        dC.append(1.)
        xC.append(GroundNodes[i])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(GroundNodes[i])
    nc += len(GroundNodes)
    nc2 += len(GroundNodes)

    for i in range(len(SourceNodes)):
        dF.append(1.)
        xF.append(SourceNodes[i])
        yF.append(nc+i)
        dF.append(1.)
        xF.append(nc+i)
        yF.append(SourceNodes[i])

        dC.append(1.)
        xC.append(SourceNodes[i])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(SourceNodes[i])
    nc += len(SourceNodes)
    nc2 += len(SourceNodes)

    for i in range(len(SourceEdges)):
        dF.append(1.)
        xF.append(EI[SourceEdges[i]])
        yF.append(nc+i)
        dF.append(1.)
        xF.append(nc+i)
        yF.append(EI[SourceEdges[i]])

        dF.append(-1.)
        xF.append(EJ[SourceEdges[i]])
        yF.append(nc+i)
        dF.append(-1.)
        xF.append(nc+i)
        yF.append(EJ[SourceEdges[i]])

        dC.append(1.)
        xC.append(EI[SourceEdges[i]])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(EI[SourceEdges[i]])

        dC.append(-1.)
        xC.append(EJ[SourceEdges[i]])
        yC.append(nc2+i)
        dC.append(-1.)
        xC.append(nc2+i)
        yC.append(EJ[SourceEdges[i]])
    nc += len(SourceEdges)
    nc2 += len(SourceEdges)

    for i in range(len(TargetNodes)):
        dC.append(1.)
        xC.append(TargetNodes[i])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(TargetNodes[i])
    nc2 += len(TargetNodes)

    for i in range(len(TargetEdges)):
        dC.append(1.)
        xC.append(EI[TargetEdges[i]])
        yC.append(nc2+i)
        dC.append(1.)
        xC.append(nc2+i)
        yC.append(EI[TargetEdges[i]])

        dC.append(-1.)
        xC.append(EJ[TargetEdges[i]])
        yC.append(nc2+i)
        dC.append(-1.)
        xC.append(nc2+i)
        yC.append(EJ[TargetEdges[i]])
    nc2 += len(TargetEdges)

    # Incidence matrix templates
    sDMF = csc_matrix((r_[ones(NE),-ones(NE)], (r_[arange(NE),arange(NE)], r_[EI,EJ])), shape=(NE, nc))
    sDMC = csc_matrix((r_[ones(NE),-ones(NE)], (r_[arange(NE),arange(NE)], r_[EI,EJ])), shape=(NE, nc2))

    # Constraint border Laplacian matrices
    sBLF = csc_matrix((dF,                   # data
                      (xF, yF)),             # coordinates
                      shape=(nc, nc))
    sBLC = csc_matrix((dC,                   # data
                      (xC, yC)),             # coordinates
                      shape=(nc2, nc2))

    # Matrix for cost computation
    sDot = sBLC[nc:,:nc]

    return (sDMF, sDMC, sBLF, sBLC, sDot)

np.random.seed(42)

from scipy.sparse import csc_matrix, diags, spdiags
from scipy.sparse.linalg import spsolve, splu, minres
from numpy.random import choice, randn

NN, NE, EI, EJ = SquareGrid(5, 5, True)
print(NN)
print(NE)
print(EI)
print(EJ)

sources = 4
targets = 2
sourceedges = 0
targetedges = 0

trials = 1
nTaskTypes = 1
nTasksPerType = 1
eta = 1.e-3
lr = 0.05
Steps = 15000

GroundNodes = array([NN-1])

num = 430
CompSteps = asarray(unique(around(r_[0, logspace(0, log10(Steps-1), num)])), int)
nComp = len(CompSteps)
ACS = zeros([trials, nComp])
KEs = zeros([trials, NE])

np.random.seed(42)

for t in range(trials):
    # prepare allostery tasks #
    ###########################
    TaskTypes = []
    for tt in range(nTaskTypes):
        # Choose source and target nodes
        np.random.seed(42)

        # print(NN)
        # print(NE)
        # print(EI)
        # print(EJ)

        NodeList = np.random.choice(range(NN-1), size=sources+targets, replace=False)
        EdgeList = np.random.choice(range(NE), size=sourceedges+targetedges, replace=False)
        SourceNodes = NodeList[:sources]       # Source nodes
        TargetNodes = NodeList[sources:]       # Target nodees
        SourceEdges = EdgeList[:sourceedges]   # Source edges
        TargetEdges = EdgeList[sourceedges:]   # Target edges

        # print(SourceNodes)
        # print(TargetNodes)
        # print(GroundNodes)

        print(NodeList)
        print(EdgeList)

        sDMF, sDMC, sBLF, sBLC, sDot = SparseIncidenceConstraintMatrix(SourceNodes, SourceEdges, TargetNodes, TargetEdges, GroundNodes, NN, EI, EJ)
        ff = zeros([nTasksPerType, sDMF.shape[1]])                      # free constraints template vector
        fc = zeros([nTasksPerType, sDMC.shape[1]])                      # clamped constraints template vector
        Desired = zeros([nTasksPerType, sDMC.shape[1]-sDMF.shape[1]])   # Desired outputs
        for tn in range(nTasksPerType):
            np.random.seed(42)
            NodeData = np.random.randn(sources)                    # Input nodes voltages
            OutNodeData = np.random.randn(targets) * 0.3           # Desired output node voltages
            EdgeData = np.random.randn(sourceedges)                # Input edge voltage drops
            OutEdgeData = np.random.randn(targetedges) * 0.3       # Desired output edge voltage drops
            print(NodeData)
            print(OutNodeData)
            ff[tn, NN:] = r_[0., NodeData, EdgeData]                                   # free constraints templeate vector
            fc[tn, NN:] = r_[0., NodeData, EdgeData, OutNodeData, OutEdgeData]         # clamped constraints template vector
            Desired[tn] = r_[OutNodeData, OutEdgeData]                                 # desired node voltages\edge voltage drops

        TaskTypes.append([sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, Desired])
    ###########################
    K = ones(NE)*1
    sK = spdiags(K, 0, NE, NE, format='csc')          # diagonal matrix with the conductance values on the diagonal elements

    # Cost computation and PF, PC mem
    CEq0 = 0.
    PFs = zeros([nTaskTypes, sDMF.shape[1], nTasksPerType])
    # print(PFs.shape)
    PCs = zeros([nTaskTypes, sDMC.shape[1], nTasksPerType])
    for i in range(nTaskTypes):
        # Free state computation
        sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, Desired = TaskTypes[i]
        AFinv = splu(sBLF + sDMF.T*sK*sDMF)           # sparse LU decomposition gives back the inverse matrix
        PF = AFinv.solve(ff.T)
        ans = sum((sDot.dot(PF) - Desired.T)**2)
        CEq0 += ans    # sDot.dot(PF) gives the Free state voltages\voltage drops at the desired sources\edges
        PFs[i] = PF

        ACinv = splu(sBLC + sDMC.T*sK*sDMC)
        PC = ACinv.solve(fc.T)
        PCs[i] = PC
        # print(PF.shape, PC.shape)
        # print(PC)
        # print("---")
    CEq0 /= nTaskTypes*nTasksPerType
    CEq = CEq0

    print(t, 0, CEq0)

    c = 0
    ACS[t,c] = 1.

    #iterate over training steps
    for steps in range(1,Steps+1):
        # choose specific task
        tt = randint(0, nTaskTypes)
        tn = randint(0, nTasksPerType)
        sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, Desired = TaskTypes[tt]

        # Free state computation
        if CEq/CEq0 > 1.e-50:
            # Direct solve
            AFinv = splu(sBLF + sDMF.T*sK*sDMF)
            # print(tn)
            PF = AFinv.solve(ff[tn])
        else:
            # Iterative solve
            PF = minres(sBLF + sDMF.T*sK*sDMF, ff[tn], tol=1.e-10, x0=PFs[tt,:,tn])[0]

        DPF = sDMF * PF
        PPF = DPF**2

        # Clamped state computation
        FST = sDot.dot(PF)
        Nudge = FST + eta * (Desired[tn] - FST)
        fc[tn,sDMF.shape[1]:] = Nudge

        if CEq/CEq0 > 1.e-50:
            # Direct solve
            ACinv = splu(sBLC + sDMC.T*sK*sDMC)
            PC = ACinv.solve(fc[tn])
        else:
            # Iterative solve
            PC = minres(sBLC + sDMC.T*sK*sDMC, fc[tn], tol=1.e-10, x0=PCs[tt,:,tn])[0]

        DPC = sDMC * PC
        PPC = DPC**2

        PFs[tt,:,tn] = PF
        PCs[tt,:,tn] = PC

        DKL = + 0.5 * (PPC - PPF) / eta
        K2 = K - lr * DKL

        K2 = K2.clip(1.e-6,1.e4)

        DK = K2-K
        K = K2
        sK = spdiags(K, 0, NE, NE, format='csc')

        if steps%500==0:
            # Cost computation
            CEq = 0.
            for i in range(nTaskTypes):
                # Free state computation
                sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, Desired = TaskTypes[i]
                AFinv = splu(sBLF + sDMF.T*sK*sDMF)
                PF = AFinv.solve(ff.T)
                CEq += sum((sDot.dot(PF) - Desired.T)**2)
            CEq /= nTaskTypes*nTasksPerType
            print(t, steps, CEq/CEq0, norm(DK))

        if steps in CompSteps:
            c += 1
            # Cost computation
            CEq = 0.
            for i in range(nTaskTypes):
                # Free state computation
                sDMF, sDMC, sBLF, sBLC, sDot, ff, fc, Desired = TaskTypes[i]
                AFinv = splu(sBLF + sDMF.T*sK*sDMF)
                PF = AFinv.solve(ff.T)
                CEq += sum((sDot.dot(PF) - Desired.T)**2)
            CEq /= nTaskTypes*nTasksPerType
            ACS[t,c] = CEq/CEq0


    KEs[t] = K