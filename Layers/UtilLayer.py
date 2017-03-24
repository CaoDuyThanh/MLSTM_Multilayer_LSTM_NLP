import numpy
import theano

def CreateSharedParameter(rng    = None,
                          shape  = None,
                          factor = 1,
                          nameVar = ''):
    if rng is None:
        rng = numpy.random.RandomState(1993)

    wBound = numpy.sqrt(6.0 / numpy.sum(shape))
    initValue = numpy.asarray(rng.uniform(
                                        low   = -wBound,
                                        high  =  wBound,
                                        size  =  shape
                                    ),
                                    dtype = theano.config.floatX
                            )
    sharedVar = theano.shared(initValue * factor, borrow = True, name = nameVar)
    return sharedVar