import theano.tensor as T

""" L1 - Regularization """
def L1(W):
    return abs(W).sum()

""" L2 - Regularization """
def L2(W):
    return abs(W ** 2).sum()

""" Cross entropy """
def CrossEntropy(yps, ys):
    cost = []
    for idx in range(yps.__len__()):
        cost.append(T.log(yps[idx][0, ys[idx]]))
    return -T.sum(cost)

""" Category entropy """
def CategoryEntropy(output, y):
    return T.sum(T.nnet.categorical_crossentropy(output, y))

""" Error """
def Error(output, y):
    return T.mean(T.neq(T.argmax(output, 1), y))