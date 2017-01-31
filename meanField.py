from matplotlib import pylab as plt
import numpy as np
from scipy import linalg as sp_linalg
from scipy.stats import norm
from scipy.stats import multivariate_normal
import GPflow
import cProfile
from matplotlib2tikz import save as save_tikz
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import embed

plot_min = -2.
plot_max = 4.

nGaussHermite = 100
hermgauss = np.polynomial.hermite.hermgauss
gh_x, gh_w = hermgauss(nGaussHermite)
gh_w /= np.sqrt(np.pi)

class SigmoidLikelihood(object):
    def __init__(self, gamma, y ):
        #gamma \in (0,\infty) is a scalar
        #y is of shape (N,) and each entry is in set {-1,1}
        self.gamma = gamma
        self.y = y
        self.N = len(y)
        #conveniently all we have to do is multiply these two together.        
        self.alphas = self.gamma * self.y 
        
    def likelihoodQuadrature( self, means, variances ):
        #evaluates <p(y|f)> and <p(y|f) f> under normal distributions.
        #means is of shape (N,)
        #variances is of shape (N,) and each entry is in set (0,\infty)
        
        #check shapes
        assert( len(means) == self.N )
        assert( len(variances) == self.N )

        intermediates = 1. + variances * self.alphas**2
        denominators = np.sqrt( intermediates  )
        numerators = self.alphas * means
        first_exp = norm.cdf( numerators / denominators )
        second_exp = means * first_exp + np.exp( - 0.5* (means * self.alphas)**2 / intermediates ) * variances * self.alphas / denominators / np.sqrt( 2. * np.pi )
        return first_exp, second_exp        

    def evaluateLikelihoods( self, fs ):
        #evaluate p(y|f)
        #fs is of shape (N,K) where K represents different function evaluations.
        #returned shape is also (N,K)
        epsilons = self.alphas[:,None] * fs
        return norm.cdf( epsilons )

    def evaluateLikelihoodIndividual( self, index, fs ):
        #evaluate p(y|f) for element given by index.
        epsilons = self.alphas[index] * fs 
        return norm.cdf( epsilons )

class GaussianLikelihood(object):
    def __init__(self, epsilon, y ):
        #epsilon \in (0,\infty) is a scalar which is the noise variance.
        #y is of shape (N,) and each entry is a real number.
        self.epsilon = epsilon
        self.y = y
        self.N = len(y)
        
    def likelihoodQuadrature( self, means, variances ):
        #evaluates <p(y|f)> and <p(y|f) f> under normal distributions.
        #means is of shape (N,)
        #variances is of shape (N,) and each entry is in set (0,\infty)
        
        #check shapes
        assert( len(means) == self.N )
        assert( len(variances) == self.N )

        # <p(y|f)> derivation.
        #  N(y | f, \epsilon ) N( f | mean, variance )
        # =N(f | y, \epsilon ) N( f | mean, variance ) apply Gaussian product formula to obtain.
        # =\beta N( f | a, B ) where \beta is not a function of f.
        # hence \int \beta N( f | a, B) \dee f = \beta
        # further \beta = \normal( y | mean, \epsilon + \variance )
        
        betas = np.zeros( self.N )
        for index in range(self.N):
            betas[index] = norm.pdf( self.y[index], loc = means[index], scale = np.sqrt((self.epsilon + variances[index] )) )
                
        # <p(y|f) f > derivation.
        # as above but now we have
        # \int \beta N( f | a, B) f \dee f 
        # = \beta a
        # where a = ( y * variance + mean * \epsilon  ) / ( variance + \epsilon )
        a = ( self.y * variances + means * self.epsilon ) / ( variances + self.epsilon )
        
        return betas, a * betas

    def evaluateLikelihoods( self, fs ):
        #evaluate p(y|f)
        #fs is of shape (N,K) where K represents different function evaluations.
        # and first point represents data point index.
        (N,K) = fs.shape
        assert( self.N == N)
        #y is of shape (N,)
        deltas = self.y[:,None] - fs
        mahalanobisTerms = -0.5*deltas**2 / self.epsilon
        normalizingTerms = -0.5*np.log( self.epsilon * 2. * np.pi )
            
        #returned shape is also (N,K)
        return np.exp( mahalanobisTerms + normalizingTerms )

    def evaluateLikelihoodIndividual( self, index, fs ):
        #evaluate p(y|f) for element given by index.        
        K = len( fs )
        likelihoods = np.zeros( K )
        for k in range(K):
            likelihoods[k] = norm.pdf( self.y[index] , loc = fs[k] , scale = np.sqrt(self.epsilon) )
        return likelihoods

def runMeanField( likelihood, K, n_iters = 20 ):
    #implementation of mean field for a Gaussian prior with a given likelihood.
    #Based on implementation by Nikisch and Rasmussen 2008
    
    #y is data
    
    #likelihood is a function that takes (means, variances) which
    #are vectors of the same length and returns < p(y|f) > and < p(y|f)f > 
    #for the normal distributions implied by the means and variances.
    
    #K is the covariance matrix of the Gaussian prior.
    
    #We assume q factorizes which is the standard mean field assumption.
    #Then q_i(f_i) = Z_i^{-1} p(y_i | f_i ) \exp( -(f_i - a_i)^2 / 2\sigma_i^2 )
    # Z_i normalizes the distribution
    # and \sigma_i^2 is the conditional variance of the i th component conditioned on everything else
    
    #an unpublished result due to Matthews, Hensman and Ghahramani says ...
    #a unit natural gradient step is implemented by the following update.     
    
    #\mathbf{a} <- ( \mathbf{I} - \diag(\sigma^2) \mathbf{K}^{-1} ) <f>_{q}
    
    #<f>_q can be recoverred using the ratio of the first to second moments of the likelihood.
    
    whitening_constant = 1e-5
    whitened_K = K + np.eye(K.shape[0])*whitening_constant
    
    cholK = np.linalg.cholesky(whitened_K)
    
    invCholK = np.linalg.inv( cholK )
    invK = np.dot( invCholK.T, invCholK )
    
    a = np.zeros(K.shape[0]) #intialize a
    sigma_squared = np.diag( invK ) ** (-1) 
    
    for index in range(n_iters):
        first_exp, second_exp = likelihood( a, sigma_squared )
        exp_f = second_exp / first_exp #expectation of marginals of f under q.
        a = exp_f - sigma_squared * sp_linalg.cho_solve( (cholK,True), exp_f )
    return a, sigma_squared, first_exp

class MeanFieldDistribution(object):
    def __init__(self, likelihood, K ):
        #likelihood is an object.
        self.likelihood = likelihood
        self.K = K
        self.D = K.shape[0]
        likelihoodIntegrator = lambda u,v : self.likelihood.likelihoodQuadrature( u, v )
        self.optimal_a, self.sigma_squared, self.Zs = runMeanField( likelihoodIntegrator, self.K  )
    
    def marginalDensity( self, index , f ):
        #index is a dimension in range [0,...,D-1]
        #f is of shape (J,)
        normalDensities = normalDensity( np.atleast_1d(self.optimal_a[index]), np.atleast_2d(self.sigma_squared[index]), np.atleast_2d( f ) )
        likelihoods = self.likelihood.evaluateLikelihoodIndividual( index, f )
        #returns shape (J,)
        return normalDensities * likelihoods / self.Zs[index]
        
    def densities( self, f ):
        #f is an array of shape (D,J)
        marginal_densities = np.zeros( f.shape )
        for index in range(self.D):
            marginal_densities[index,:] = self.marginalDensity( index, f[index,:] )
        #returns shape (J,)
        return marginal_densities.prod(axis=0)

def bruteForceNormalizer( likelihood, K):
    #likelihood is a function that returns the individual likelihood terms.
    #K is the Gaussian covariance function.
    nSamples = 10**6
    cK = np.linalg.cholesky(K)
    rng = np.random.RandomState(1)
    samples = np.dot(cK, rng.randn( K.shape[0],nSamples ) )
    likelihoods = likelihood( samples ).prod( axis=0 )
    marginalLikelihood = likelihoods.mean()
    stdLikelihood = likelihoods.std() / np.sqrt( nSamples )
    return marginalLikelihood, stdLikelihood

def bruteForceMarginalizer( marginalIndex, marginalValue, likelihood, K, conditional, marginalLikelihood ):
    #marginalIndex is the index of the requested marginal.
    #marginalValue is the value of the variable in the corresponding marginal.
    #likelihood evaluates the likelihood componentwise.
    #K is the covariance matrix.
    #conditional evaluates the conditional p(f_index | f_{not index} ). 
    #its arguments are f and index.
    nSamples = nGaussHermite
    otherMarginalIndex = 1-marginalIndex
    samples = np.zeros( (K.shape[0], nSamples ) )
    samples[otherMarginalIndex,:] =  gh_x * np.sqrt(2.0 * K[otherMarginalIndex,otherMarginalIndex])
    samples[marginalIndex,:] = np.ones( nSamples ) * marginalValue
    likelihoods = likelihood( samples ).prod( axis=0 ) * conditional(samples, marginalIndex) / marginalLikelihood * gh_w
    mean = likelihoods.sum()
    std = likelihoods.std() / np.sqrt( nSamples )
    return mean, std

def gaussianConditional( mean, K, fs, index  ):
    # K is covariance matrix of shape (D,D)
    # fs is set of function values of shape (D,J)
    # index is the index of the corresponding marginal.
    jointDensities = normalDensity( mean, K, fs )
    otherIndeces = [ elem for elem in range(K.shape[0]) if elem!=index ]
    subMean = mean[otherIndeces]
    subK = K[otherIndeces,:][:,otherIndeces]
    subFs = fs[otherIndeces,:]
    denominatorDensities = normalDensity( subMean, subK, subFs )
    return jointDensities / denominatorDensities

def plotDensity( fig, ax, density, levels, include_colorbar=False ):
    #density is a function that takes a array of shape (D,J)
    #where D is the dimensionality of the space and J is the number of density points evaluated.
    grid = []
    num = 100
    grid_points = np.linspace( plot_min, plot_max, num = num, endpoint=True ) 
    grid_spacing = np.diff(grid_points)[0]
    grid_area = grid_spacing**2
    for elem_a in grid_points:
        for elem_b in grid_points:
            grid.append( [elem_a, elem_b] )
    array_grid = np.array( grid ).T
    densities = density( array_grid )
    integral_check = np.sum(densities) * grid_area
    print "Joint check integral: ", integral_check
    maximal_density = np.max(densities)
    print "Maximal density: ", maximal_density
    assert( maximal_density < np.max(levels) )
    assert( np.abs( integral_check - 1. ) < 1e-3 )
    X,Y = np.meshgrid( grid_points, grid_points )
    co = ax.contour( X,Y, densities.reshape(num,num), levels = levels )
    ax.grid(True)     
    if include_colorbar:
        axins = inset_axes(ax,width="5%",height="100%", loc=6, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        fig.colorbar( co, cax=axins )   

def plotMarginalDensity( ax, density ):
    num = 100
    grid_points = np.linspace( plot_min, plot_max, num = num, endpoint=True ) 
    grid_spacing = np.diff( grid_points )[0]
    densities = density( grid_points )
    marginal_integral = np.sum(densities)*grid_spacing
    print "Marginal check integral: ",marginal_integral
    assert( abs( marginal_integral - 1. ) < 1e-2 )
    ax.plot( grid_points, densities )    
    ax.set_ylim( [0. , 0.9 ] )
    ax.grid(True)

def normalDensity( mean, K, f ):
    #mean is of shape (D,)
    #cov is of shape (D,D) 
    #f is an array of shape (D,J)
    #where D is the dimensionality of the space and J is the number of density points evaluated.    
    D = K.shape[0]
    deltas = mean[:,None] - f # shape (D,J)
    whitening_constant = 1e-5
    whitened_K = K + np.eye(K.shape[0])*whitening_constant
    alphas = np.linalg.solve( whitened_K, deltas ) #shape (D,J)
    mahalanobisTerms = -0.5 * (deltas * alphas).sum(axis=0) #shape (J,)
    logDet = -0.5*np.linalg.slogdet( whitened_K )[1] #a scalar
    constantTerms = -0.5*D*np.log( 2. * np.pi ) #another scalar
    logDensities = constantTerms + logDet + mahalanobisTerms #shape (J,)
    return np.exp(logDensities) #shape (J,)

def getRbfKernel(variance,lengthscale):
    kern = GPflow.kernels.RBF(1)
    kern.variance=variance
    kern.lengthscales._array[0] = lengthscale
    return kern

def plotFlowModel( fig, fullAx, marginalAx, model, signalVariance, X, levels, diag, include_colorbar=False ):
    
    if diag:
        full_mean = model.q_mu.value
        full_variance = np.diag( model.q_sqrt.value.flatten()**2 )
    else:
        full_mean, full_variance = model.predict_f_full_cov( X )
        full_variance = full_variance[:,:,0]
    joint_density = lambda x : normalDensity( full_mean.flatten()/np.sqrt(signalVariance), full_variance/signalVariance, x )
    plotDensity( fig, fullAx, joint_density, levels, include_colorbar )
    
    if diag:
        mean = full_mean[0,:]
        variance = full_variance[0,0]
    else:
        mean, variance = model.predict_f( np.atleast_2d( X[0,:] ) )
    marginal_density = lambda x: normalDensity( mean.flatten()/np.sqrt(signalVariance), np.atleast_2d(variance)/signalVariance, x) 
    plotMarginalDensity( marginalAx , marginal_density )    

def demoLikelihood(likelihoodObject, varianceRescale, flowLikelihood ):
    mean = np.zeros( 2 )
    beta = 0.85
    K = np.array( [[1. , beta], [beta, 1.] ] )

    #get brute force estimate of normalizing constant.
    likelihoodEvaluator = lambda x : likelihoodObject.evaluateLikelihoods(x)
    marginalLikelihood, stdLikelihood = bruteForceNormalizer( likelihoodEvaluator, K )
    
    #plot the exact posterior density.
    #plot the exact posterior density.
    posteriorDensity = lambda x : normalDensity( mean, K, x ) * likelihoodObject.evaluateLikelihoods(x).prod(axis=0) / marginalLikelihood
    fig, axes = plt.subplots(2,4)
    levels = np.linspace( 0., 0.8, 11 )
    plotDensity( fig, axes[0,0], posteriorDensity, levels )
    
    conditional = lambda x, y : gaussianConditional( mean, K, x, y )
    posteriorMarginalDensity = lambda x: np.array( [bruteForceMarginalizer( 0, elem, likelihoodEvaluator, K, conditional, marginalLikelihood )[0] for elem in x] )
    plotMarginalDensity( axes[1,0] , posteriorMarginalDensity )

    #setup and run mean field.
    meanFieldDistribution = MeanFieldDistribution( likelihoodObject, K )
    plotDensity( fig, axes[0,1] , meanFieldDistribution.densities, levels )
    meanFieldMarginalDensity = lambda x: meanFieldDistribution.marginalDensity( 0, x )
    plotMarginalDensity( axes[1,1] , meanFieldMarginalDensity )
    
    #setup and run correlated Gaussian in GPflow.
    #need to shoe horn this into existing GPflow functionality. 

    #scaling likelihood by gamma is the same as scaling f by gamma.
    #thus the variance is scaled by sqrt(gamma)
    #we will need to account for this afterwards.
    X = np.atleast_2d( np.array( [0., 1.] ) ).T
    Y = np.atleast_2d( np.array( [1., 1.] ) ).T
    #need to relate beta to lengthscale
    lengthscale = np.sqrt( -0.5 / np.log( beta ) )
    correlated_gauss_model = GPflow.vgp.VGP(X, Y, kern=getRbfKernel(varianceRescale,lengthscale), likelihood=flowLikelihood )
    correlated_gauss_model.kern.fixed=True
    correlated_gauss_model.optimize()

    plotFlowModel( fig, axes[0,3], axes[1,3], correlated_gauss_model, varianceRescale, X, levels, diag=False, include_colorbar=True  )
    
    #setup and run uncorrelated Gaussian in GPflow.
    uncorrelated_gauss_model = GPflow.svgp.SVGP(X,Y, Z=X, kern=getRbfKernel(varianceRescale,lengthscale), likelihood=flowLikelihood, q_diag=True, whiten=False)
    uncorrelated_gauss_model.Z.fixed = True
    uncorrelated_gauss_model.kern.fixed= True
    uncorrelated_gauss_model.optimize()
    
    plotFlowModel( fig, axes[0,2], axes[1,2], uncorrelated_gauss_model, varianceRescale, X, levels, diag=True)
    #save_tikz('two_dimensions.tikz', figurewidth='\\figurewidth', figureheight = '\\figureheight')
    
def getFlowGaussianLikelihood(epsilon):
    likelihood = GPflow.likelihoods.Gaussian() 
    likelihood.variance = epsilon
    likelihood.fixed=True
    return likelihood

def testSigmoidLikelihood():
    gamma = 4.
    nTest = 100
    y = np.ones( nTest )
    likelihood = SigmoidLikelihood( gamma, y )
    nSamples = 100000
    rng = np.random.RandomState(1)
    means = rng.randn( nTest )
    variances = rng.randn( nTest ) ** 2

    #setup crude reference implementation using sampling.
    validationLikelihood = lambda x : norm.cdf( gamma * x )
    higherMoment = lambda x : norm.cdf( gamma * x ) * x
    
    normal_samples = means[:,None] + np.sqrt(variances[:,None]) * rng.randn( nTest,nSamples )
    likelihood_samples = validationLikelihood( normal_samples.T ).T
    likelihood_means = likelihood_samples.mean( axis = 1 )
    likelihood_stds = likelihood_samples.std( axis = 1 ) / np.sqrt( nSamples )
    
    moment_samples = higherMoment(normal_samples.T).T
    moment_means = moment_samples.mean( axis = 1 )
    moment_stds = moment_samples.std( axis = 1 ) / np.sqrt( nSamples )
    
    #setup test using exact integrals
    test_first_moments, test_second_moments = likelihood.likelihoodQuadrature( means, variances )
    
    assert( (np.abs( moment_means - test_second_moments) < 1e-2).all() )
    assert( (np.abs( likelihood_means - test_first_moments ) < 1e-2).all() )

def demoMeanField():
    testSigmoidLikelihood()
    gamma = 3.0
    #gamma = 1.0
    epsilon = 1.
    y = np.array( [1., 1.] )
    demoLikelihood( SigmoidLikelihood( gamma, y ), np.sqrt(gamma), GPflow.likelihoods.Bernoulli() )
    #demoLikelihood( GaussianLikelihood( epsilon, y ) , 1., getFlowGaussianLikelihood(epsilon)  )
    embed()
    
if __name__ == "__main__":
    demoMeanField()
