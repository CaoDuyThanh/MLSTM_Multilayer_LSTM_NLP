import pickle
import cPickle
from Layers.MLSTMHiddenLayer import *
from Utils.CostFHelper import *

BETA1 = 0.9
BETA2 = 0.999
DELTA = 0.00000001

class MLSTM:
    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 numLayers,
                 truncate,
                 batchSize,
                 activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.NumLayers = numLayers
        self.Truncate = truncate
        self.BatchSize = batchSize
        self.Activation = activation

        self.createMLSTM()

    def createMLSTM(self):
        # Save shared parameters
        self.Params = None
        self.ParamsLayers = []

        # Create MLSTM model
        self.HiddenLayers = []
        for layerId in range(self.Truncate):
            if layerId == 0:
                hiddenLayer = MLSTMHiddenLayer(
                    rng        = self.Rng,
                    numIn      = self.NumIn,
                    numHidden  = self.NumHidden,
                    numLayers  = self.NumLayers
                )
                self.Params = hiddenLayer.Params
            else:
                hiddenLayer = MLSTMHiddenLayer(
                    rng         = self.Rng,
                    numIn       = self.NumIn,
                    numHidden   = self.NumHidden,
                    numLayers   = self.NumLayers,
                    params      = self.Params
                )
            self.ParamsLayers.append(hiddenLayer.Params)
            self.HiddenLayers.append(hiddenLayer)

        # Create train model
        X = T.ivector('X')
        Y = T.ivector('Y')
        LearningRate = T.fscalar('LearningRate')
        CState = T.matrix('SState', dtype = 'float32')   # Start C state
        SState = T.matrix('SState', dtype = 'float32')   # Start S state

        # Feed-forward
        Yps = []
        C = CState
        S = SState
        finalState = None
        for idx, layer in enumerate(self.HiddenLayers):
            [C, S, Yp] = layer.FeedForward(X[idx], C, S)
            Yps.append(Yp)
            if idx == self.Truncate - 1:
                finalState = C + S

        # Calculate cost | error function
        cost = CrossEntropy(Yps, Y)

        # Get params and calculate gradients - Adam method
        grads = T.grad(cost, self.Params)
        updates = []
        for (param, grad) in zip(self.Params, grads):
            mt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            vt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)

            clipGrad = grad.clip(a_min = -1.0, a_max = 1.0)

            newMt = BETA1 * mt + (1 - BETA1) * clipGrad
            newVt = BETA2 * vt + (1 - BETA2) * T.sqr(clipGrad)

            tempMt = newMt / (1 - BETA1)
            tempVt = newVt / (1 - BETA2)

            step = - LearningRate * tempMt / (T.sqrt(tempVt) + DELTA)
            updates.append((mt, newMt))
            updates.append((vt, newVt))
            updates.append((param, param + step))

        self.TrainFunc = theano.function(
            inputs  = [X, Y, LearningRate, CState, SState],
            outputs = [cost] + finalState,
            updates = updates,
        )

        [C, S, Yp] = self.HiddenLayers[-1].FeedForward(X[0], CState, SState)
        self.PredictFunc = theano.function(
            inputs  = [X, CState, SState],
            outputs = [Yp] + C + S
        )

    def Generate(self, length, x):
        InitState = numpy.zeros(
                    shape=(self.NumLayers, self.NumHidden),
                    dtype=theano.config.floatX
        )

        # Feed-forward
        genStringIdx = [x]
        CState = InitState
        SState = InitState
        for idx in range(length):
            result = self.PredictFunc([x], CState, SState)
            Yp     = result[0]
            CState = numpy.asarray(result[                 1 :     1 + self.NumLayers], dtype='float32')
            SState = numpy.asarray(result[1 + self.NumLayers : 1 + 2 * self.NumLayers], dtype='float32')
            x = numpy.random.choice(range(self.NumIn), p=Yp[0])
            genStringIdx.append(x)
        return genStringIdx

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self. Params]
