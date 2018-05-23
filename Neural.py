import netMath
import random
import copy

class Neural(object):
    def __init__(self, numInputs, numOutputs, numLayers, numNodes, outputMap):
        #numLayers is the number of *hidden* layers
        self.__numInputs = numInputs
        self.__numOutputs = numOutputs
        self.__numLayers = numLayers
        self.__numNodes = numNodes
        self.initNodes()
        self.randomiseWeights()
        self.randomiseBiases()
        assert(self.__numInputs == len(outputMap))
        self.__outputMap = outputMap

    def __emptyNodeList(self):
        nlist = [[]]*(self.__numLayers+2)
        for layer in range(self.__numLayers+2):
            numNodes = self.getNumNodesForLayer(layer)
            nlist[layer] = [0]*numNodes
        return nlist

    def __emptyWeightList(self):
        wlist = [[]]*(self.__numLayers+2)
        for layer in range(1,self.__numLayers+2):
            numLeftNodes = self.__numInputs if layer == 1 else self.__numNodes
            numRightNodes = self.__numOutputs if layer == self.__numLayers+1 else self.__numNodes
            wlist[layer] = [[0]*numLeftNodes]*numRightNodes
        return wlist
    
    def initNodes(self):
        self.__nodes = []
        #self.__nodes is structured as [layer][number (from top if visualising)]
        for layer in range(self.__numLayers+2):
            self.__nodes.append([])
            numNodes = None
            if layer == 0:
                numNodes = self.__numInputs
            elif layer == self.__numLayers+1:
                numNodes = self.__numOutputs
            else:
                numNodes = self.__numNodes
            self.__nodes[layer] = [0]*numNodes
                

    def randomiseWeights(self):
        self.__weights = [[]]*(self.__numLayers+2)
        #self.__weights is formatted as [right layer][rightNodeIndex][leftNodeIndex]
        #note that __weights[0] is empty
        for layer in range(1,self.__numLayers+2):
            numLeftNodes = self.__numInputs if layer == 1 else self.__numNodes
            numRightNodes = self.__numOutputs if layer == self.__numLayers+1 else self.__numNodes
            self.__weights[layer] = [[0]*numLeftNodes]*numRightNodes
            for r in range(numRightNodes):
                for l in range(numLeftNodes):
                    self.__weights[layer][r][l] = random.randrange(-5,5)

    def randomiseBiases(self):
        self.__biases = [[]]*(self.__numLayers+2)
        #self.__biases is formatted [layer][nodeIndex]
        for layer in range(self.__numLayers+2):
            numNodes = None
            if layer == 0:
                numNodes = self.__numInputs
            elif layer == self.__numLayers+1:
                numNodes = self.__numOutputs
            else:
                numNodes = self.__numNodes
            self.__biases[layer] = [random.randrange(-5,5) for i in range(numNodes)]

    def getNumNodesForLayer(self, layer):
        assert(layer>=0 and layer<=self.__numLayers+1)
        if layer == 0:
            return self.__numInputs
        elif layer == self.__numLayers+1:
            return self.__numOutputs
        else:
            return self.__numNodes

    #goes through and calculates all the values of the neurons without doing any backprop
    def calc(self,inputMatrix):
        assert(len(inputMatrix)==self.__numInputs)
        #set input nodes
        self.__nodes[0] = inputMatrix

        #calculate activation for each node
        for layer in range(1,self.__numLayers+2):
            numNodes = self.getNumNodesForLayer(layer)
            for node in range(numNodes):
                self.__nodes[layer][node] = netMath.sigmoid(netMath.dot(self.__nodes[layer-1],self.__weights[layer][node])+self.__biases[layer][node])
        #return output matrix
        return self.__nodes[self.__numLayers+1]

    def predict(self,inputMatrix):
        currentMax = -float('inf')
        maxIndex = -1
        output = self.calc(inputMatrix)
        for i in range(self.__numOutputs):
            if output[i] > currentMax:
                maxIndex = i
                currentMax = output[i]
        assert(maxIndex != -1)
        return self.__outputMap[maxIndex]

    #takes in desired output and an input, and returns gradient
    #in form (weightmatrix,biasmatrix) 
    def backprop(self, inputMatrix, desiredOutput):
        assert(len(inputMatrix)==self.__numInputs)
        assert(len(desiredOutput)==self.__numOutputs)
        #set input nodes
        self.__nodes[0] = inputMatrix

        #calculate activation for each node
        for layer in range(1,self.__numLayers+2):
            numNodes = self.getNumNodesForLayer(layer)
            for node in range(numNodes):
                z = netMath.dot(self.__nodes[layer-1],self.__weights[layer][node])+self.__biases[layer][node]
                self.__nodes[layer][node] = netMath.sigmoid(z)
                
        #calculate dC/da for all nodes
        dC = self.__emptyNodeList()
        #calculate dC/da for output nodes
        for node in range(self.__numOutputs):
            outL = self.__numLayers+1
            dC[outL][node] = 2*(self.__nodes[outL][node]-desiredOutput[node])

        #calculate dC/da for all other nodes
        for layer in range(self.__numLayers,0,-1):
            for node in range(self.__numNodes):
                dsum = 0
                for j in range(self.getNumNodesForLayer(layer+1)):
                    dz = self.__weights[layer+1][j][node]
                    dsig = self.__nodes[layer+1][j]*(1-self.__nodes[layer+1][j])
                    dCn = dC[layer+1][j]
                    dsum += dz*dsig*dCn
                dC[layer][node] = dsum

        #calculate dC/dW for all weights
        dW = self.__emptyWeightList()
        for rlayer in range(self.__numLayers+1,0,-1):
            for rnode in range(self.getNumNodesForLayer(rlayer)):
                for lnode in range(self.getNumNodesForLayer(rlayer-1)):
                    rA = self.__nodes[rlayer][rnode]
                    lA = self.__nodes[rlayer-1][lnode]
                    dW[rlayer][rnode][lnode] = lA*rA*(1-rA)*dC[rlayer][rnode]

        #calculate dC/dB for all biases
        dB = self.__emptyNodeList()
        for layer in range(self.__numLayers,-1,-1):
            for node in range(self.getNumNodesForLayer(layer)):
                nodea = self.__nodes[layer][node]
                dB[layer][node] = nodea*(1-nodea)*dC[layer][node]
        
        return (dW,dB)

    def cost(self, inputMatrix, desiredOutput):
        assert(len(inputMatrix) == len(desiredOutput))
        return sum([(inputMatrix[i]-desiredOutput[i])**2 for i in range(len(inputMatrix))])

    def train(self, trainingData, batchSize, batches, learningFactor = 1.0, printProgressInterval = False):
        #trainingData takes the form [(input,desiredOutput),(input,desiredOutput),...]
        tindex = 0
        for batch in range(batches):
            if printProgressInterval:
                if batch%printProgressInterval == 0:
                    print("\nStarting batch "+str(batch)+"\n")
            dW = self.__emptyWeightList()
            dB = self.__emptyNodeList()
            for run in range(batchSize):
                tdata = trainingData[tindex]
                partial = self.backprop(tdata[0],tdata[1])
                dW = netMath.add3D(dW,partial[0])
                dB = netMath.add2D(dB,partial[1])
                tindex = (tindex+1)%len(trainingData)
                if printProgressInterval:
                    if batch%printProgressInterval == 0 and run == 0:
                        print("Current cost:")
                        print(self.cost(tdata[0],self.calc(tdata[0])))
                        
            dW = netMath.multiply3D(dW, learningFactor/float(batchSize))
            dB = netMath.multiply2D(dB, learningFactor/float(batchSize))
            self.__weights = netMath.add3D(self.__weights,dW)
            self.__biases = netMath.add2D(self.__biases,dB)
        if printProgressInterval:
            print("\n\nDone!")

    def getWeights(self):
        return self.__weights

    def getBiases(self):
        return self.__biases
    
if __name__ == "__main__":
    #old tests
    n = Neural(4,4,1,4,[1,2,3,4])
    data = []
    for i in range(500):
        inputMatrix = [random.random(),random.random(),random.random(),random.random()]
        data.append((inputMatrix,inputMatrix))
    print("Data's done")
    n.train(data,len(data),1000)
    print("\n===Test===\n\n")
    print("[1,0,0,0]:",n.calc([1,0,0,0]))
                
