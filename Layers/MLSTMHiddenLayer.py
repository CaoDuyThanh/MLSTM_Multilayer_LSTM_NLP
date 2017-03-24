import theano
import theano.tensor as T
import numpy

# Import utils
from UtilLayer import *

class MLSTMHiddenLayer:

    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 numLayers,
                 params = None):
        # Set parameters
        self.Rng       = rng
        self.NumIn     = numIn
        self.NumHidden = numHidden
        self.NumLayers = numLayers
        self.Params    = params

        self.createModel()

    def createModel(self):
        if self.Params is None:
            Wis = []; Wfs = []; Wcs = []; Wos = []  # Weight
            Uis = []; Ufs = []; Ucs = []; Uos = []  # Weight
            bis = []; bfs = []; bcs = []; bos = []  # Bias

            for layerId in range(self.NumLayers):
                if (layerId == 0):
                    numIn = self.NumIn
                else:
                    numIn = self.NumHidden

                # Init Wi | Ui | bi
                Wi = CreateSharedParameter(self.Rng, (self.NumHidden, self.NumHidden), 1, 'Wi_Layer_%d' % (layerId))
                Wis.append(Wi)
                Ui = CreateSharedParameter(self.Rng, (numIn, self.NumHidden), 1, 'Ui_Layer_%d' % (layerId))
                Uis.append(Ui)
                bi = CreateSharedParameter(self.Rng, (self.NumHidden, ), 0, 'bi_Layer_%d' % (layerId))
                bis.append(bi)

                # Init Wf | bf
                Wf = CreateSharedParameter(self.Rng, (self.NumHidden, self.NumHidden), 1, 'Wf_Layer_%d' % (layerId))
                Wfs.append(Wf)
                Uf = CreateSharedParameter(self.Rng, (numIn, self.NumHidden), 1, 'Uf_Layer_%d' % (layerId))
                Ufs.append(Uf)
                bf = CreateSharedParameter(self.Rng, (self.NumHidden, ), 0, 'bf_Layer_%d' % (layerId))
                bfs.append(bf)

                # Init Wc | bc
                Wc = CreateSharedParameter(self.Rng, (self.NumHidden, self.NumHidden), 1, 'Wc_Layer_%d' % (layerId))
                Wcs.append(Wc)
                Uc = CreateSharedParameter(self.Rng, (numIn, self.NumHidden), 1, 'Uc_Layer_%d' % (layerId))
                Ucs.append(Uc)
                bc = CreateSharedParameter(self.Rng, (self.NumHidden, ), 0, 'bc_Layer_%d' % (layerId))
                bcs.append(bc)

                # Init Wo | bo
                Wo = CreateSharedParameter(self.Rng, (self.NumHidden, self.NumHidden), 1, 'Wo_Layer_%d' % (layerId))
                Wos.append(Wo)
                Uo = CreateSharedParameter(self.Rng, (numIn, self.NumHidden), 1, 'Uo_Layer_%d' % (layerId))
                Uos.append(Uo)
                bo = CreateSharedParameter(self.Rng, (self.NumHidden, ), 0, 'bo_Layer_%d' % (layerId))
                bos.append(bo)

            # Init Wy - output
            Wy = [CreateSharedParameter(self.Rng, (self.NumHidden, self.NumIn), 1, 'Wy_Layer')]
            by = [CreateSharedParameter(self.Rng, (self.NumIn, ), 0, 'by_Layer')]

            self.Params = Wis + \
                          Wfs + \
                          Wcs + \
                          Wos + \
                          Uis + \
                          Ufs + \
                          Ucs + \
                          Uos + \
                          bis + \
                          bfs + \
                          bcs + \
                          bos + \
                          Wy + \
                          by



    def FeedForward(self, Xk, Ckm1, Skm1):
        Ss = []
        Cs = []
        for layerId in range(self.NumLayers):
            Wi = self.Params[layerId]
            Wf = self.Params[layerId + self.NumLayers * 1]
            Wc = self.Params[layerId + self.NumLayers * 2]
            Wo = self.Params[layerId + self.NumLayers * 3]
            Ui = self.Params[layerId + self.NumLayers * 4]
            Uf = self.Params[layerId + self.NumLayers * 5]
            Uc = self.Params[layerId + self.NumLayers * 6]
            Uo = self.Params[layerId + self.NumLayers * 7]
            bi = self.Params[layerId + self.NumLayers * 8]
            bf = self.Params[layerId + self.NumLayers * 9]
            bc = self.Params[layerId + self.NumLayers * 10]
            bo = self.Params[layerId + self.NumLayers * 11]

            if layerId == 0:
                i = T.nnet.sigmoid(Ui[Xk] + T.dot(Skm1[layerId], Wi) + bi)
                f = T.nnet.sigmoid(Uf[Xk] + T.dot(Skm1[layerId], Wf) + bf)
                o = T.nnet.sigmoid(Uo[Xk] + T.dot(Skm1[layerId], Wo) + bo)
                g = T.tanh(Uc[Xk] + T.dot(Skm1[layerId], Wc) + bc)
            else:
                i = T.nnet.sigmoid(T.dot(Ui, Ss[-1]) + T.dot(Skm1[layerId], Wi) + bi)
                f = T.nnet.sigmoid(T.dot(Uf, Ss[-1]) + T.dot(Skm1[layerId], Wf) + bf)
                o = T.nnet.sigmoid(T.dot(Uo, Ss[-1]) + T.dot(Skm1[layerId], Wo) + bo)
                g = T.tanh(T.dot(Uc, Ss[-1]) + T.dot(Skm1[layerId], Wc) + bc)

            c = Ckm1[layerId] * f + g * i
            s = T.tanh(c) * o

            Cs.append(c)
            Ss.append(s)

        # Calculate output
        Wy = self.Params[-2]
        by = self.Params[-1]
        out = T.nnet.softmax(T.dot(Ss[-1], Wy) + by)

        return [Cs, Ss, out]