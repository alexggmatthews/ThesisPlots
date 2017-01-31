from matplotlib import pyplot as plt
from matplotlib2tikz import save as save_tikz
import csv
import numpy as np
from IPython import embed

from ExampleKernels import Brownian
import GPflow

snelson_offset = 3.

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    
    return np.array( dataList )

def getTrainingData():
    X = readCsvFile( 'train_inputs' ) + snelson_offset
    Y = readCsvFile( 'train_outputs' ) 
    trainIndeces = []
    nPoints = X.shape[0]
    for index in range(nPoints):
        if ( (index%2) == 0):
            trainIndeces.append( index )
    return X[trainIndeces,:],Y[trainIndeces,:]

def getTestData():
    xtest = np.linspace( 0., 13., 1000, endpoint=True )
    return np.atleast_2d(xtest).T

def standardPlotLimits(ax):
    ax.set_xlim( [-0.5e-1, 12. ] )
    ax.set_ylim( [-4.0,4.0 ] )
        
def plotPredictions( ax, model, color, label, include_noise=True ):
    X,Y = getTrainingData()
    xtest = getTestData()
    if include_noise:
        predMean, predVar = model.predict_y(xtest)
    else:
        predMean, predVar = model.predict_f(xtest)        
    ax.plot( X, Y, 'ro' )
    ax.plot( xtest, predMean, color, label=label )
    ax.plot( xtest, predMean + 2.*np.sqrt(predVar),color )
    ax.plot( xtest, predMean - 2.*np.sqrt(predVar), color )  
    standardPlotLimits(ax)

def plotPredictiveSamples( ax, model, color, rng ):
    nSamples = 3
    xtest, samples = getPredictiveSamples( model, nSamples, rng )
    X,Y = getTrainingData()
    ax.plot( xtest, samples, color=color )
    ax.plot( X, Y, 'ro' )
    standardPlotLimits(ax)
    
def getRbfKernel():
    kern = GPflow.kernels.RBF(1)
    return kern
     
def getBrownianKernel():
    kern = Brownian()
    return kern     

def runModel(kernel):
    X,Y = getTrainingData()
    model = GPflow.gpr.GPR( X=X, Y=Y, kern=kernel )
    model.optimize()
    return model

def drawNormalSamples( mean, cov, nSamples, rng ):
    whitening = 1e-6
    chol = np.linalg.cholesky( cov+np.eye( cov.shape[0] )*whitening)
    samples = np.dot( chol, rng.randn( chol.shape[0], nSamples ) ) + mean[:,None]
    return samples

def getPredictiveSamples( model, nSamples, rng ):
    xtest = getTestData()
    mean, cov = model.predict_f_full_cov(xtest) 
    samples = drawNormalSamples( mean.flatten(), cov[:,:,0], nSamples, rng )
    return xtest, samples

def output_current_plot( fileName ):
    save_tikz(fileName,figurewidth='\\figurewidth', figureheight = '\\figureheight')       

def posteriorDemo():
    rng = np.random.RandomState(4)
    rbfModel = runModel( kernel = getRbfKernel() )
    print "rbfModel \n"
    print rbfModel
    brownianModel = runModel( kernel = getBrownianKernel() )
    print "brownianModel \n"
    print brownianModel
    fig, axes = plt.subplots(3,2)
    plotPredictiveSamples( axes[0,0], rbfModel, 'c', rng )
    plotPredictiveSamples( axes[0,1], brownianModel, 'c', rng )
    plotPredictions( axes[1,0], rbfModel, 'g', 'RBF kernel', include_noise=False )
    plotPredictions( axes[1,1], brownianModel, 'g', 'Wiener kernel', include_noise=False )
    plotPredictions( axes[2,0], rbfModel, 'b', 'RBF kernel', include_noise=True )
    plotPredictions( axes[2,1], brownianModel, 'b', 'Wiener kernel', include_noise=True )
    output_current_plot( 'posteriors.tikz' )
    embed()

if __name__ == '__main__':
    posteriorDemo()  
