import GPflow
from GPflow.tf_hacks import eye
from GPflow.param import Param, Parameterized, AutoFlow
import GPflow.transforms as transforms
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cProfile
from scipy.cluster import vq
import csv

#import conditionals

from IPython import embed
from matplotlib2tikz import save as save_tikz

colors = ['b','r','g','c']
x_range =  [0.,10. ] 

class Brownian(GPflow.kernels.Kern):
    def __init__(self, scale=1.0, lengthscale=1.0, active_dims=None ):
        GPflow.kernels.Kern.__init__(self, input_dim=1, active_dims=active_dims)
        self.scale = Param(scale, transforms.positive)
        self.lengthscale = Param(lengthscale, transforms.positive)
        
    def K(self, X, X2=None):
        if X2 is None:
            return self.scale * tf.minimum(X / self.lengthscale, tf.transpose(X) / self.lengthscale)
        else:
            return self.scale * tf.minimum(X / self.lengthscale, tf.transpose(X2) / self.lengthscale )

    def Kdiag(self, X):
        return tf.squeeze(X / self.lengthscale ) * self.scale    

#def referenceBrownianKernel( X, lengthScale, scale ):

def testBrownian():
   lengthScale = 2.
   scale = 2.
   kernel = Brownian()
   kernel.lengthscale = lengthScale
   kernel.scale = scale
   rng = np.random.RandomState(1)
   
   x_free = tf.placeholder('float64')
   kernel.make_tf_array(x_free)
   X = tf.placeholder('float64')
   X_data = rng.rand(3, 1)
   #reference_gram_matrix = referenceRbfKernel(X_data, lengthScale, scale)
   with kernel.tf_mode():
      gram_matrix = tf.Session().run(kernel.K(X), feed_dict={x_free: kernel.get_free_state(), X: X_data})
   print "X_data ", X_data, "\n"
   print "gram_matrix ", gram_matrix, "\n" 
    
def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    
    return np.array( dataList )
    
def getTrainingData():
    overallX = readCsvFile( 'train_inputs' )
    overallY = readCsvFile( 'train_outputs' )
    
    #take every fourth point
    step = 4
    overallX = overallX[0::step,:]
    overallY = overallY[0::step,:]
    
    return overallX, overallY
    
def getPredictPoints():
    predpoints = readCsvFile( 'test_inputs' )
    return predpoints
    
def getPredictPoints():
    predpoints = readCsvFile( 'test_inputs' )
    return predpoints

def getRbfKernel():
    kern = GPflow.kernels.RBF(1)
    kern.variance=1.0
    kern.lengthscales._array[0] = 1.0
    return kern
     
def getBrownianKernel():
    kern = Brownian()
    kern.scale=0.3**2.
    kern.lengthscale = 1.
    return kern     
     
def getRegressionModel(X,Y,kernel):
    m = GPflow.gpr.GPR(X, Y, kern=kernel )
    return m
    
def plotInducingPoints( model ):
    plt.plot( model.Z._array, np.zeros( model.Z._array.shape ), 'ko', markersize=5  )

def drawNormalSamples( mean, cov, nSamples, rng ):
    chol = np.linalg.cholesky( cov )
    samples = np.dot( chol, rng.randn( chol.shape[0], nSamples ) ) + mean[:,None]
    return samples

def drawPriorSamples( trained_model, X_vals, nSamples, rng ):
    
    x_free = tf.placeholder('float64')
    trained_model.kern.make_tf_array(x_free)
    X = tf.placeholder('float64')
    
    with trained_model.kern.tf_mode():
        gram_matrix = tf.Session().run( trained_model.kern.K(X) , feed_dict={x_free:trained_model.kern.get_free_state(), X:X_vals})
    
    samples = drawNormalSamples( np.zeros( gram_matrix.shape[0] ),  gram_matrix + np.eye( gram_matrix.shape[0] )*1e-5 , nSamples, rng )
    return samples 
    
def standardPlotLimits(ax):
    ax.set_ylim( [-3.,3. ] )
    ax.set_xlim( x_range )
    
def singleFunctionPlot( x, y, markersize=10 ):
    plt.plot( x, y, 'ko', markersize=markersize )
    standardPlotLimits()    

def samplesPlot( x_new, f_new_samples, x_old=None, f_old=None, cross_size=20, circle_size=10 ):
    for sampleIndex in range(f_new_samples.shape[1] ):
        plt.plot( x_new, f_new_samples[:,sampleIndex], color=colors[sampleIndex], marker='x', markersize=cross_size )
    if x_old!=None:
        assert(f_old!=None)
        singleFunctionPlot( x_old, f_old, circle_size ) 
    
    standardPlotLimits()

def output_current_plot( fileName ):
    save_tikz(fileName,figurewidth='\\figurewidth', figureheight = '\\figureheight')    

def plotKernelSamples( ax, kernel, rng, nTest ):
    xtrain,ytrain = getTrainingData()
    xtest = np.atleast_2d( np.linspace( 0.,10., nTest , endpoint=True )).T
    nSamplesPerPlot = 5
    
    model = getRegressionModel(xtrain,ytrain,kernel=kernel )
   
    #Use these hyperparameters and draw data from the prior.
    priorSamples = drawPriorSamples( model , xtest , nSamplesPerPlot, rng )
    
    ax.plot( xtest, priorSamples )
    ax.set_xlabel('Input $x$')
    ax.set_ylabel('Output $f(x)$')        
    standardPlotLimits(ax)

def kernelDemo():
    gs = gridspec.GridSpec(2, 1)
    gs.update( hspace = 0.3 )
    
    plotKernelSamples( plt.subplot(gs[0]), getRbfKernel(), np.random.RandomState(1), 100 )
    plotKernelSamples( plt.subplot(gs[1]), getBrownianKernel(), np.random.RandomState(3), 500 )    
    
    output_current_plot( 'kernel_examples.tikz' )
    embed()
   
if __name__ == '__main__':
    kernelDemo()    
