from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib2tikz import save as save_tikz
import csv
import numpy as np
from sklearn.cluster import KMeans as kmeans
from scipy import linalg as slinalg
from IPython import embed

import GPflow


from ExamplePosteriors import getRbfKernel, output_current_plot

snelson_offset = 3.

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    
    return np.array( dataList )
    
def runModel(kernel):
    X,Y = getTrainingHoldOutData()
    model = GPflow.gpr.GPR( X=X, Y=Y, kern=kernel )
    #model.optimize()
    return model

def getModelParameters(model):
    hyper_dict = {}
    hyper_dict['lengthscale'] = model.kern.lengthscales.value
    hyper_dict['signal_variance'] = model.kern.variance.value
    hyper_dict['noise_variance'] = model.likelihood.variance.value
    return hyper_dict

def setModelParameters(model,hyper_dict):
    model.kern.lengthscales = hyper_dict['lengthscale']
    model.kern.variance = hyper_dict['signal_variance']
    model.likelihood.variance = hyper_dict['noise_variance']

def getTrainingHoldOutData():
	rng = np.random.RandomState(1)
	nPoints = 3
	X = np.atleast_2d(rng.rand( nPoints ) * 5. ).T
	Y = np.atleast_2d(rng.randn( nPoints ) ).T
	return X,Y

def getSparseModel(num_inducing):
    random_state = np.random.RandomState(1)
    X,Y = getTrainingHoldOutData()
    km = kmeans(n_clusters = num_inducing, random_state = random_state ).fit(X)
    Z = km.cluster_centers_
    model = GPflow.sgpr.SGPR(X, Y, kern=getRbfKernel(),  Z=Z)
    return model 

def computeKLdivergence(exactModel,sparseModel):
    # KL[Q||\hat{P}] = -sparseModel.logLikelihood + exactModel.logLikelihood
    return -sparseModel.compute_log_likelihood() + exactModel.compute_log_likelihood()

def finiteGaussianKL( meanA, covA, meanB, covB ):
    # numpy implementation of KL divergence between two multivariate Gaussians.
    #Computes KL[ Normal( meanA, covA) || Normal(meanB,covB) ]
    D = covA.shape[0] #dimensionality of space
    #meanA and meanB are shape (D,) 
    #covA and covB are shape (D,D)
    whitening = 1e-5
    #whitening = 0#1e-6
    covBwhite = covB + np.eye(D)*whitening
    covAwhite = covA + np.eye(D)*whitening
    delta = meanB - meanA
    mahalanobisTerm = np.dot( delta, np.linalg.solve( covBwhite, delta) )
    logDeterminantTerm = np.linalg.slogdet( covBwhite )[1] - np.linalg.slogdet( covAwhite )[1]
    traceTerm = np.trace( slinalg.solve( covBwhite, covAwhite) )
    return 0.5 * ( traceTerm + mahalanobisTerm - D + logDeterminantTerm )

def testFiniteGaussianKL():
	#These two things should give the same result since the Gaussians have diagonal covariance.
	nDimensions = 4
	rng = np.random.RandomState(1)
	meanA = rng.randn(nDimensions)
	meanB = rng.randn(nDimensions)
	diagCovA = rng.rand(nDimensions)
	diagCovB = rng.rand(nDimensions)
	covA = np.diag(diagCovA)
	covB = np.diag(diagCovB)
	
	accumulatorA = 0.
	for index in range(nDimensions):
		currentMeanA = np.atleast_1d(meanA[index])
		currentMeanB = np.atleast_1d(meanB[index])
		currentCovA = np.atleast_2d(diagCovA[index]) 
		currentCovB = np.atleast_2d(diagCovB[index])
		accumulatorA += finiteGaussianKL( currentMeanA, currentCovA, currentMeanB, currentCovB)
		
	print "Method A for finite Gaussian KL gives ", accumulatorA
	print "Method B for finite Gaussian KL gives ", finiteGaussianKL( meanA, covA, meanB, covB )

def output_current_plot( fileName ):
    save_tikz(fileName,figurewidth='\\figurewidth', figureheight = '\\figureheight') 
	
def regressionExample():
    testFiniteGaussianKL()
    X,Y = getTrainingHoldOutData()
    fig, axes = plt.subplots(2,1) 
    limits = [-1.,6.]  
    axes[0].plot(X,Y,'ro')
    axes[0].set_xlim(limits )
    axes[0].set_ylim([-2., 2.])
    axes[0].grid(True)
    axes[0].axhline(y=0, color='k')
    axes[0].set_xlabel('Input position')
    axes[0].set_ylabel('Observed output')
    kernel = getRbfKernel()
        
    exactModel = runModel( getRbfKernel() )
    exactModel.kern.fixed=True
    exactModel.likelihood.fixed=True

    posteriorMean, posteriorCov = exactModel.predict_f_full_cov(X)
		
    sparseModel = getSparseModel( num_inducing = 1)
    sparseModel.kern.fixed=True
    sparseModel.likelihood.fixed=True
    

    KLbetweenProcesses = []
    finiteDimensionalKL = []
    augmentedFiniteKL = []    
    	
    Z_range = np.linspace( limits[0], limits[1], 100 )
    for Z in Z_range:
		sparseModel.Z = Z
		sparseMean, sparseCov = sparseModel.predict_f_full_cov( X )
		XcupZ = np.hstack([X.T,np.atleast_2d(Z).T]).T
		augmentedSparseMean, augmentedSparseCov = sparseModel.predict_f_full_cov( XcupZ )
		augmentedPosteriorMean, augmentedPosteriorCov = exactModel.predict_f_full_cov( XcupZ )
		KLbetweenProcesses.append( computeKLdivergence( exactModel, sparseModel )[0] )
		finiteDimensionalKL.append( finiteGaussianKL( sparseMean.flatten(), sparseCov[:,:,0], posteriorMean.flatten(), posteriorCov[:,:,0]  ) )
		augmentedFiniteKL.append( finiteGaussianKL( augmentedSparseMean.flatten(), augmentedSparseCov[:,:,0], augmentedPosteriorMean.flatten(), augmentedPosteriorCov[:,:,0]  ) ) 
    
    KLbetweenProcesses = np.array( KLbetweenProcesses )
    finiteDimensionalKL = np.array( finiteDimensionalKL )
    augmentedFiniteKL = np.array( augmentedFiniteKL )
    
    print "Minimum KLbetween processes ", Z_range[np.argmin( KLbetweenProcesses )]
    print "Minimum finiteDimensionalKL ",  Z_range[np.argmin( finiteDimensionalKL )]
    print "Minimum augmented KL ", Z_range[np.argmin( augmentedFiniteKL )]
    
    #lineA, = axes[1].plot(Z_range, KLbetweenProcesses,'g', label='KL processes')
    lineB, = axes[1].plot(Z_range, finiteDimensionalKL,'r', label='Unaugmented KL')
    lineC, = axes[1].plot(Z_range, augmentedFiniteKL,'b', label='Augmented KL')
    axes[1].set_xlabel('Inducing input position')
    axes[1].set_ylabel('Divergence in Nats')
    plt.legend( handles=[lineB, lineC], loc=2 )
    
    output_current_plot('KL_comparison.tikz')
    embed()
    
if __name__ == '__main__':
    regressionExample()
