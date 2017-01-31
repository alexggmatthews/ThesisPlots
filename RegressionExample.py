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

def getTrainingHoldOutData(isTrain=True):
    X = readCsvFile( 'train_inputs' ) + snelson_offset
    Y = readCsvFile( 'train_outputs' ) 
    trainIndeces = []
    testIndeces = []
    nPoints = X.shape[0]
    skip = 2
    for index in range(nPoints):
        if ( (index%skip) == 0):
            trainIndeces.append( index )
        else:
            testIndeces.append( index )
    
    if isTrain:    
        return X[trainIndeces,:],Y[trainIndeces,:]
    else:
        return X[testIndeces,:],Y[testIndeces,:]        

def runModel(kernel):
    X,Y = getTrainingHoldOutData()
    model = GPflow.gpr.GPR( X=X, Y=Y, kern=kernel )
    model.optimize()
    return model

def getTestData():
    xtest = np.linspace( -3. + snelson_offset, 10.+snelson_offset, 1000, endpoint=True )
    return np.atleast_2d(xtest).T

def standardPlotLimits(ax):
    ax.set_xlim( [-3+snelson_offset, 9.+snelson_offset ] )
    ax.set_ylim( [-4.0,4.0 ] )

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
    D = covA.shape[0] #dimensionality of space
    #meanA and meanB are shape (D,) 
    #covA and covB are shape (D,D)
    whitening = 1e-5
    covBwhite = covB + np.eye(D)*whitening
    covAwhite = covA + np.eye(D)*whitening
    delta = meanB - meanA
    mahalanobisTerm = np.dot( delta, np.linalg.solve( covBwhite, delta) )
    logDeterminantTerm = np.linalg.slogdet( covBwhite )[1] - np.linalg.slogdet( covAwhite )[1]
    traceTerm = np.trace( slinalg.solve( covBwhite, covAwhite) )
    return 0.5 * ( traceTerm + mahalanobisTerm - D + logDeterminantTerm )

def getReverseKL( exactModel, sparseModel, reverse=True ):
    ZstackX = np.vstack( [ sparseModel.Z.value, exactModel.X.value ] )
    posteriorMean , posteriorCovariance = exactModel.predict_f_full_cov( ZstackX )
    approximatingMean, approximatingCovariance = sparseModel.predict_f_full_cov( ZstackX )
    if reverse:
        KL = finiteGaussianKL( posteriorMean.flatten(), posteriorCovariance[:,:,0], approximatingMean.flatten(), approximatingCovariance[:,:,0] )
    else:
        KL = finiteGaussianKL( approximatingMean.flatten(), approximatingCovariance[:,:,0] , posteriorMean.flatten(), posteriorCovariance[:,:,0] )
    return KL

def plotPredictions( ax, model, color, label, include_noise=True, Z=None ):
    X,Y = getTrainingHoldOutData()
    xtest = getTestData()
    if include_noise:
        predMean, predVar = model.predict_y(xtest)
    else:
        predMean, predVar = model.predict_f(xtest)        
    ax.plot( X, Y, 'ro' )
    ax.plot( xtest, predMean, color, label=label )
    ax.plot( xtest, predMean + 2.*np.sqrt(predVar),color )
    ax.plot( xtest, predMean - 2.*np.sqrt(predVar), color )  
    ax.text( 6., 2.5, label )
    if Z!=None:
        ax.plot( Z, -3.*np.ones( Z.shape ), 'ko' )
    standardPlotLimits(ax)

def getHoldOutMetrics(model):
    XholdOut,YholdOut = getTrainingHoldOutData(False)
    mean_log_density = model.predict_density( XholdOut, YholdOut ).mean()
    predictive_means = model.predict_y( XholdOut )
    rmsError = np.sqrt( np.mean( (mean_log_density - predictive_means)**2 ) )
    return mean_log_density, rmsError 
    
def regressionExample():
    exactModel = runModel( getRbfKernel() )
    
    fig, axes = plt.subplots(5,1)

    plotInducingPointNumbers = [2,4,8,16]
    plotPredictions( axes[4], exactModel, 'g', 'Exact', include_noise=True )
    
    maximal_ind_number = 16
    step = 2
    num_inducing_points = range(2,maximal_ind_number+1,step)
    forwardKLs = []
    forwardKLSAlternate = []
    reverseKLs = []
    for index in range(len(num_inducing_points)):
        num_inducing = num_inducing_points[index]
        sparseModel = getSparseModel( num_inducing = num_inducing)
        sparseModel.kern.fixed = True
        sparseModel.likelihood.fixed = True
        setModelParameters( sparseModel, getModelParameters( exactModel ) )
        sparseModel.optimize()
        forwardKLs.append( computeKLdivergence(exactModel,sparseModel) )
        forwardKLSAlternate.append( getReverseKL(exactModel,sparseModel, reverse=False ) )
        reverseKLs.append( getReverseKL( exactModel, sparseModel ) )
        if num_inducing in plotInducingPointNumbers:
            plotPredictions( axes[ plotInducingPointNumbers.index(num_inducing) ], sparseModel, 'g', 'M='+str(num_inducing), include_noise=True, Z=sparseModel.Z.value )
        #standardPlotLimits( axes[index+1,0] )

    output_current_plot( 'regression_example_predictions.tikz' )

    #fig, axes = plt.subplots(3,1)
    plt.figure()
    gs1 = gridspec.GridSpec(3,1)
    gs1.update(wspace=0.1, hspace=0.05)
    axes = [ plt.subplot( elem ) for elem in gs1 ]
    
    axes[0].plot( num_inducing_points, forwardKLs,'g', label='KL[Q||phat]' )
    axes[0].set_ylabel('KL[Q||phat] / nats')
    plt.setp(axes[0].get_xticklabels(), visible=False)
    
    axes[1].plot( num_inducing_points, reverseKLs,'b', label='KL[Phat||Q]' )
    axes[1].set_ylabel('KL[Phat||Q] / nats')
    plt.setp(axes[1].get_xticklabels(), visible=False)
    
    #Start by finding optimal hyperparameters.

    
    mean_log_exact, rms_exact = getHoldOutMetrics( exactModel )
    
    mean_logs = []
    rms = []
    
    for index in range(len(num_inducing_points)):
        num_inducing = num_inducing_points[index]
        sparseModel = getSparseModel( num_inducing = num_inducing)
        #setModelParameters( sparseModel, getModelParameters( exactModel ) )
        sparseModel.optimize()
        current_mean_log, current_rms = getHoldOutMetrics( sparseModel )
        mean_logs.append( current_mean_log )
        rms.append( current_rms )
    
    axes[2].plot( num_inducing_points, -1.*np.array(mean_logs)  )
    axes[2].plot( np.array(num_inducing_points)[[0,-1]], -1.*np.array([mean_log_exact,mean_log_exact]) )
    axes[2].set_xlabel('Number of inducing points')
    axes[2].set_ylabel('Hold out negative log probability')

    #axes.plot( num_inducing_points, rms )

    output_current_plot( 'regression_example.tikz' )
    
    embed()
    
if __name__ == '__main__':
    regressionExample()
