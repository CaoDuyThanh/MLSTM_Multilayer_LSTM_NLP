import os.path
from Utils.DataHelper import DataHelper
from MLSTM import *

DATASET_NAME = '../Data/The lord of the rings.txt'
SAVE_PATH = '../Pretrained/model.pkl'

# NETWORK PARAMATERS
NUM_HIDDEN = 500
NUM_LAYERS = 1
TRUNCATE = 25
BATCH_SIZE = 10

# TRAINING PARAMETERS
NUM_EPOCH= 200
LEARNING_RATE = 0.001

VISUALIZE_FREQUENCY = 1000
TEST_FREQUENCY      = 10000

# GLOBAL VARIABLES
Dataset = None

def generateString(lstmModel):
    global Dataset
    print ('Generate a random string...')
    print ('-----------------------------------------------------------------------------')
    start = numpy.random.choice(range(Dataset.NumChars))
    generatedStringIdx = lstmModel.Generate(1000, start)
    generatedString = ''.join([Dataset.IdxToCharacter[charIdx] for charIdx in generatedStringIdx])
    print ('%s' % (generatedString))
    print ('-----------------------------------------------------------------------------')

def loadData():
    global Dataset
    Dataset = DataHelper(DATASET_NAME)

def NLP():
    global Dataset
    #############################
    #        BUILD MODEL        #
    #############################
    rng = numpy.random.RandomState(123)
    lstmModel = MLSTM(
        rng        = rng,
        numIn      = Dataset.NumChars,
        numHidden  = NUM_HIDDEN,
        numLayers  = NUM_LAYERS,
        truncate   = TRUNCATE,
        batchSize=BATCH_SIZE,
        activation = T.tanh
    )

    # Train model - using early stopping
    # Load old model if exist
    if os.path.isfile(SAVE_PATH):
        print ('Load old model and continue the training')
        file = open(SAVE_PATH)
        lstmModel.LoadModel(file)
        file.close()

    # Gradient descent
    # Init start state
    initS = numpy.zeros(
        shape = (NUM_LAYERS, NUM_HIDDEN),
        dtype = theano.config.floatX
    )
    initC = numpy.zeros(
        shape=(NUM_LAYERS, NUM_HIDDEN),
        dtype=theano.config.floatX
    )

    epoch = 0
    iter  = 0
    trainCost = []
    dynamicLearning = LEARNING_RATE
    while (epoch < NUM_EPOCH):
        iter += 1
        if (iter % VISUALIZE_FREQUENCY == 0):
            print ('Epoch = %d, iteration =  %d, cost = %f ' % (epoch, iter, numpy.mean(trainCost)))
            trainCost = []

        # Calculate cost of validation set every VALIDATION_FREQUENCY iter
        if iter % TEST_FREQUENCY == 1:
            generateString(lstmModel)

            file = open(SAVE_PATH, 'wb')
            lstmModel.SaveModel(file)
            file.close()

        # Training state
        [subData, target, currEpoch] = Dataset.NextBatch(TRUNCATE)
        if iter == 1 or currEpoch > epoch:
            SState = initS
            CState = initC
        epoch = currEpoch
        result = lstmModel.TrainFunc(subData, target, dynamicLearning, CState, SState)
        cost  = result[0]
        CState = numpy.asarray(result[             1 : 1 +     NUM_LAYERS], dtype = 'float32')
        SState = numpy.asarray(result[1 + NUM_LAYERS : 1 + 2 * NUM_LAYERS], dtype = 'float32')
        trainCost.append(cost)

    # Load model and test
    if os.path.isfile(SAVE_PATH):
        file = open(SAVE_PATH)
        lstmModel.LoadModel(file)
        file.close()

if __name__ == '__main__':
    loadData()
    NLP()