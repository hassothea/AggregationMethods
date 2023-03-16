
# Import all the libraries 
# ========================
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV, BayesianRidge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from random import sample
from scipy import spatial, optimize
from sklearn import metrics
import warnings
import math
from sklearn import preprocessing
# Plotting figures
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
# Table type
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class KernelCAMRegressors(BaseEstimator):
    def __init__(self, 
                random_state = None, 
                kernel = 'radial', 
                estimator_list = None,
                show_progress = True, 
                estimator_params = None, 
                optimize_method = "grid", 
                optimize_params = {'bandwidth_list' : np.linspace(0.00001, 100, 300),
                                   'epsilon' : 10 ** (-10),
                                   'learning_rate' : 0.01,
                                   'speed' : 'constant',
                                   'n_tries' : int(5),
                                   'start' : None,
                                   'max_iter' : 100,
                                   'n_cv' : int(5)}):
        """
        This is a class of the implementation of the Kernel-based consensual aggregation method for regression by Has (2023).


        * Parameters:
        ------------
        - random_state: (default is `None`) set the random state of the random generators in the class.
        kernel: (default is 'radial') the name of kernel function to be used for the aggregation. 
            It should be an element of the list ['exponential', 'gaussian', 'radial', 'epanechnikov', 'biweight', 'triweight', 'triangular', 'cobra', 'naive'].
            Some options such as 'gaussian' and 'radial' lead to the same gaussian kernel function. Same for 'cobra' and 'naive', corresponds to Biau et al. (2016).
        
        - estimator_list: (default is None) the list of intial estimators (machines). If None, intial estimators 'knn', 'ridge', 'lasso', 'tree', 'random_forest' and 'svm' are used.
            It should be a sublist of the following list: ['knn', 'ridge', 'lasso', 'tree', 'random_forest' and 'svm', 'sgd', 'bayesian_ridge', 'adaboost', 'gradient_boost'].
        
        - show_progress: (default is `True`) boolean defining whether or not to show the progress of the optimization algorithm.
        
        - estimator_params: (default is `None`) a dictionary of parameters of the basic estimators given in the `estimator_list` argument. It should be a dictionary containing the names of the basic estimators, 
            which is also a dictionary of the names of its paremters.
        
        - optimize_method: (default is "grid") optimization algorithm for learning the bandwidth parameter. It should be either "grid" (grid search) or "grad" (gradient descent for non-compactly supported kernel such as radial kernel).
        
        - optimize_params: a dictionary of parameters of the optimization algorithm (both grid and grad). Its should be dictionary of the following elements:
            - 'bandwidth_list' : bandwidth grid for grid search algorithm
            - 'epsilon' : threshold to stop the gradient descent algorithm
            - 'learning_rate' : learning rate for grad algorithm
            - 'speed' : speed of the learning rate,
            - 'n_tries' : the number of tries for the initial step of the grad algorithm,
            - 'start' : the initial value of the bandwidth parameter
            - 'max_iter' : maximum iteration of gradient descent algorithm
            - 'n_cv' : number of cross-validation folds.

        * Returns:
        ---------
        self : returns an instance of self. 

        * Methods: 
        ---------
        - fit : fitting the aggregation method on the design features (original data or predicted features).
        - split_data : split the data into D_k = {(X_k,y_k)} and D_l = {(X_l,y_l)} to construct the estimators and perform aggregation respectively.
        - build_basic_estimators : build basic estimators for the aggregation. It is also possible to set the values of (hyper) parameters for each estimators.
        - load_predictions : to make predictions using constructed basic estimators.
        - distances : construct distance matrix according to the kernel function used in the aggregation.
        - kappa_cross_validation_error : the objective function to be minimized.
        - optimize_bandwidth : the optimization method to estimate the optimal bendwidth parameter.
        - predict : for building prediction on the new observations using any given bendwidth or the estimated one.
        - plot_learning_curve : for plotting the graphic of learning algorithm (error vs parameter).

        """
        opt_param = {'bandwidth_list' : np.linspace(0.00001, 100, 300),
                     'epsilon' : 10 ** (-10),
                     'learning_rate' : 0.001,
                     'speed' : 'constant',
                     'n_tries' : 5,
                     'start' : None,
                     'max_iter' : 100,
                    'n_cv' : int(5)
                    }
        for obj in optimize_params:
            opt_param[obj] = optimize_params[obj]
        if kernel not in ['radial', 'gaussian', 'exponential']:
            optimize_method = 'grid'
        self.random_state = random_state
        self.kernel = kernel
        self.estimator_list = estimator_list
        self.show_progress = show_progress
        self.estimator_params = estimator_params
        self.optimize_method = optimize_method
        self.optimize_params = opt_param

    def fit(self, X, y, split = .5, overlap = 0, X_k = None, y_k = None, X_l = None, y_l = None, as_predictions = False):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.X_k_ = X_k
        self.X_l_ = X_l
        self.y_k_ = y_k
        self.y_l_ = y_l
        self.as_predictions = as_predictions
        self.basic_estimtors = {}
        if not as_predictions:
            self.split_data(split = split, overlap=overlap)
            self.build_baisc_estimators()
            self.load_predictions()
            self.optimize_bandwidth(method = self.optimize_method, params = self.optimize_params)
        else:
            self.pred_X_l = pd.DataFrame(X)
            self.estimator_names = X.dtype.names
            self.number_estimators = X.shape[1]
            self.X_l_ = pd.DataFrame(X)
            self.y_l_ = y
            self.optimize_bandwidth(method = self.optimize_method, params = self.optimize_params)
        return self
    
    def split_data(self, split, overlap, k = None, shuffle_data = True):
        if shuffle_data:
            self.X_, self.y_ = shuffle(self.X_, self.y_, random_state = self.random_state)
        if k is None:
            k1 = int(len(self.y_) * split)
            k2 = int(len(self.y_) * (split+overlap))
        self.X_k_ = self.X_[:k2,:]
        self.X_l_ = self.X_[k1:,:]
        self.y_k_ = self.y_[:k2]
        self.y_l_ = self.y_[k1:]
        return self

    def build_baisc_estimators(self):
        all_estimators = {
            'extra_tree' : ExtraTreesRegressor(random_state=self.random_state),
            'knn' : KNeighborsRegressor(),
            'lasso' : LassoCV(),
            'ridge' : RidgeCV(),
            'tree' : DecisionTreeRegressor(random_state=self.random_state),
            'random_forest' : RandomForestRegressor(random_state=self.random_state),
            'svm' : SVR(),
            'bayesian_ridge' : BayesianRidge(),
            'sgd' : SGDRegressor(random_state=self.random_state),
            'adaboost' : AdaBoostRegressor(random_state=self.random_state),
            'gradient_boost' : GradientBoostingRegressor(random_state=self.random_state)
        }
        estimator_dict = {}
        if self.estimator_list == "all":
            estimator_dict = all_estimators
        elif self.estimator_list is None:
            estimator_dict = {'knn' : KNeighborsRegressor(),
                              'lasso' : LassoCV(),
                              'ridge' : RidgeCV(),
                              'tree' : DecisionTreeRegressor(random_state=self.random_state),
                              'random_forest' : RandomForestRegressor(random_state=self.random_state),
                              'extra_tree' : ExtraTreesRegressor(random_state=self.random_state),
                              'svm' : SVR()}
        else:
            for name in self.estimator_list:
                estimator_dict[name] = all_estimators[name]
        self.estimator_names = list(estimator_dict.keys())
        param_dict = {
            'knn' : None,
            'lasso' : None,
            'ridge' : None,
            'tree' : None,
            'random_forest' : None,
            'svm' : None,
            'bayesian_ridge' : None,
            'sgd' : None,
            'adaboost' : None,
            'gradient_boost' : None,
            'extra_tree' : None
        }
        self.basic_estimators = {}
        if self.estimator_params is not None:
            for name in list(self.estimator_params):
                param_dict[name] = self.estimator_params[name]
        for machine in self.estimator_names:
            try:
                mod = estimator_dict[machine]
                if param_dict[machine] is not None:
                    if machine == 'adaboost':
                        mod.estimator = DecisionTreeRegressor()
                        param_ = {}
                        for p_ in mod.estimator.get_params():
                            if p_ in list(param_dict[machine].keys()):
                                param_[p_] = param_dict[machine][p_]
                                param_dict[machine].pop(p_)
                        mod.estimator.set_params(**param_)
                        mod.set_params(**param_dict[machine])
                    else:
                        mod.set_params(**param_dict[machine])
                self.basic_estimators[machine] = mod.fit(self.X_k_, self.y_k_)
            except ValueError:
                continue
        return self

    def load_predictions(self):
        self.pred_features = {}
        for machine in self.estimator_names:
            self.pred_features[machine] = self.basic_estimators[machine].predict(self.X_l_)
        self.pred_X_l = pd.DataFrame(self.pred_features)
        self.number_estimators = len(self.estimator_names)
        return self

    def distances(self, pred_k = None, pred_l = None, distance = None, fold = None):
        lk = pred_k.shape
        ll = pred_l.shape
        D = np.full(shape = (lk[0], ll[0]), fill_value = np.float32)
        if distance in [None, "euclid", "euclidean", "l2", "L2"]:
            for i in range(lk[0]):
                D[i,:] = (np.subtract(np.array(pred_k.iloc[i,:]), pred_l) ** 2).sum(axis = 1)
        if distance in ["l1", "L1" ,"abs", "absolute"]:
            for i in range(lk[0]):
                D[i,:] = np.abs(np.subtract(np.array(pred_k.iloc[i,:]), pred_l)).sum(axis = 1)
        if distance in ["hamming", "binary", "naive"]:
            for i in range(lk[0]):
                D[i,:] = [spatial.distance.hamming(np.array(pred_k.iloc[i,:]), pred_l.iloc[j,:]) for j in range(ll[0])]
        if fold is None:
            self.distance_matrix[distance] = D
        else:
            self.distance_matrix[distance+str(fold)] = D
        return self

    def kappa_cross_validation_error(self, bandwidth = 1):
        list_kernels = {
            'exponential' : lambda x: np.exp(-x*bandwidth),
            'gaussian' : lambda x: np.exp(-x*bandwidth),
            'radial' : lambda x: np.exp(-x*bandwidth),
            'epanechnikov' : lambda x: (1 - x*bandwidth) * (x*bandwidth < 1),
            'biweight' : lambda x: (1-x*bandwidth) ** 2 * (x*bandwidth < 1),
            'triweight' : lambda x: (1-x*bandwidth) ** 3 * (x*bandwidth < 1),
            'triangular' : lambda x: (1-np.abs(x*bandwidth)) * (x*bandwidth < 1),
            'cobra' : lambda x: np.array(x*bandwidth),
            'naive' : lambda x: np.array(x*bandwidth)
        }
        if self.kernel in ['cobra', 'naive']:
            cost = np.full((self.optimize_params['n_cv'], self.number_estimators+1), fill_value = np.float32)
            for m in range(self.number_estimators+1):
                for i in range(self.optimize_params['n_cv']):
                    D_k = (list_kernels[self.kernel](np.float32(self.distance_matrix[self.distance+str(i)])) <= m/self.number_estimators)
                    D_k_ = np.sum(D_k, axis=0, dtype=np.float32)
                    D_k_[D_k_ == 0] = np.Inf
                    res = np.matmul(self.y_l_[self.index_each_fold[self.distance+str(i)]], D_k)/D_k_
                    cost[i,self.number_estimators-m] = ((res - self.y_l_[~self.index_each_fold[self.distance+str(i)]]) ** 2).mean()
            cost_ = cost.mean(axis=0)
        else:
            cost = np.full(self.optimize_params['n_cv'], fill_value = np.float32)
            for i in range(self.optimize_params['n_cv']):
                D_k = list_kernels[self.kernel](np.float32(self.distance_matrix[self.distance+str(i)]))
                D_k_ = np.sum(D_k, axis=0, dtype=np.float32)
                D_k_[D_k_ == 0] = np.Inf
                res = np.matmul(self.y_l_[self.index_each_fold[self.distance+str(i)]], D_k)/D_k_
                cost[i] = ((res - self.y_l_[~self.index_each_fold[self.distance+str(i)]]) ** 2).mean()
            cost_ = cost.mean()
        return cost_
        
    def optimize_bandwidth(self, method, params):
        def select_best_index(arr):
            l, c = arr.shape
            if l > 1:
                return arr[l//2,:]
            else:
                return arr
        kernel_to_dist = {'naive' : 'naive',
                          'cobra' : 'naive',
                          '0-1' : 'naive',
                          'uniform' : 'naive',
                          'exponential' : 'l2',
                          'gaussian' : 'l2',
                          'normal' : 'l2',
                          'radial' : 'l2',
                          'epanechnikov' : 'l2',
                          'biweight' : 'l2',
                          'triweight' : 'l2',
                          'triangular' : 'l1',
                          'triang' : 'l1'}
        self.distance_matrix = {}
        self.index_each_fold = {}
        self.distance = kernel_to_dist[self.kernel]
        st = len(self.y_l_) // params['n_cv']
        for i in range(params['n_cv']):
            idx = np.full(len(self.y_l_), True)
            if i < params['n_cv'] - 1:
                idx[int(i*st):int((i+1)*st)] = False
                x_k = self.pred_X_l.iloc[idx,:]
                x_l = self.pred_X_l.iloc[~idx,:]
            else:
                idx[int((params['n_cv']-1)*st):] = False
                x_k = self.pred_X_l.iloc[idx,:]
                x_l = self.pred_X_l.iloc[~idx,:]
            self.distances(pred_k=x_k, pred_l=x_l, distance=self.distance, fold=i)
            self.index_each_fold[self.distance+str(i)] = idx
        if self.optimize_method in ['grid', 'grid search']:
            n_iter = len(params['bandwidth_list'])
            if self.kernel in ['cobra', 'naive']:
                errors = np.full((n_iter, self.number_estimators+1), np.float32)
                if self.show_progress:
                    print('\n\t-> Grid search algorithm with '+ str(self.kernel) + ' kernel is in progress...')
                    print('\t\t~ Full process|--------------------------------------------------|100%')
                    print('\t\t~   Processing|', end ='')
                    m = 1
                    count = 0
                    if n_iter <= 50:
                        n_ = n_iter
                    else:
                        n_ = 50
                    cut = n_iter // n_
                    for iter in range(n_iter):
                        if iter == m * cut:
                            print("=", end = '')
                            if m == 49:
                                print("=|100%")
                            m += 1
                        count += 1
                        errors[iter, :] = self.kappa_cross_validation_error(bandwidth=params['bandwidth_list'][iter])
                else:
                    for iter in range(n_iter):
                        errors[iter, :] = self.kappa_cross_validation_error(bandwidth=params['bandwidth_list'][iter])
                opt_risk = np.min(np.min(errors))
                opt_id = select_best_index(np.array(np.where(errors == opt_risk)))
                self.optimize_outputs = {
                    'number_retained_estimators' : self.number_estimators-opt_id[1],
                    'opt_method' : 'grid',
                    'opt_bandwidth' : self.optimize_params['bandwidth_list'][opt_id[0]],
                    'opt_index': opt_id[0],
                    'kappa_cv_errors': errors
                }
            else:
                errors = np.full(n_iter, float)
                if self.show_progress:
                    print('\n\t-> Grid search algorithm with '+ str(self.kernel) + ' kernel is in progress...')
                    print('\t\t~ Full process|--------------------------------------------------|100%')
                    print('\t\t~   Processing|', end ='')
                    m = 1
                    count = 0
                    if n_iter <= 50:
                        n_ = n_iter
                    else:
                        n_ = 50
                    cut = n_iter // n_
                    for iter in range(n_iter):
                        if iter == m * cut:
                            print("=", end = '')
                            if m == 49:
                                print("=|100%")
                            m += 1
                        count += 1
                        errors[iter] = self.kappa_cross_validation_error(bandwidth=params['bandwidth_list'][iter])
                else:
                    for iter in range(n_iter):
                        errors[iter] = self.kappa_cross_validation_error(bandwidth=params['bandwidth_list'][iter])
                opt_risk = np.min(np.min(errors))
                opt_id = select_best_index(np.array(np.where(errors == opt_risk)))
                self.optimize_outputs = {
                    'opt_method' : 'grid',
                    'opt_bandwidth' : self.optimize_params['bandwidth_list'][opt_id[0]][0],
                    'opt_index': opt_id[0],
                    'kappa_cv_errors': errors
                }
        if self.optimize_method in ['grad', 'gradient descent', 'gd', 'GD']:
            n_iter = len(params['bandwidth_list'])
            errors = np.full(n_iter, float)
            collect_bw = []
            gradients = []
            initial_tries = []
            speed_list = {
                'constant' : lambda x, y: y,
                'log' : lambda x, y: np.log(1+x) * y,
                'root' : lambda x, y: np.sqrt(1+x) * y,
                'sqrt' : lambda x, y: (1+x ** 2) * y,
                'exp' : lambda x, y: np.exp(x) * y
            }
            if self.optimize_params['start'] is None:
                bws = np.linspace(0.0001, 1/np.var(self.y_l_), num = self.optimize_params['n_tries'])
            for i in range(self.optimize_params['n_tries']):
                initial_tries.append(self.kappa_cross_validation_error(bandwidth=bws[i]))
                bw0 = bws[np.argmin(initial_tries)]
                grad = optimize.approx_fprime(bw0, self.kappa_cross_validation_error, 10 ** (-100))
            if self.show_progress:
                print('\n\t* Gradient descent with '+ str(self.kernel) + ' kernel is implemented...')
                print('\t\t~ Initial t = 0:    \t~ bandwidth: %.5f \t~ gradient: %.5f \t~ threshold: ' %(bw0, grad[0]), end = '')
                print(str(self.optimize_params['epsilon']))
                r0 = self.optimize_params['learning_rate'] / abs(grad)        # make the first step exactly equal to `learning-rate`.
                rate = speed_list[self.optimize_params['speed']]              # the learning rate can be varied, and speed defines this change in learning rate.
                count = 0
                grad0 = grad
                while count < self.optimize_params['max_iter']:
                    bw = bw0 - rate(count, r0) * grad
                    if bw < 0 or bw is None:
                        bw = bw / 1.25
                    if count > 3:
                        if np.sign(grad)*np.sign(grad0) < 0:
                            r0 = r0 / 1.25
                        if test_threshold > self.optimize_params['epsilon']:
                            bw0 = bw
                            grad0 = grad
                        else:
                            break
                    relative = abs((bw - bw0) / bw0)
                    test_threshold = np.mean([relative, abs(grad)])
                    grad = optimize.approx_fprime(bw0, self.kappa_cross_validation_error, 10 ** (-100))
                    count += 1
                    print('\t\t~     Iteration: %d \t~ epsilon: %.5f \t~ gradient: %.5f \t~ max(change,grad): %.5f' % (count, bw[0], grad[0], test_threshold), end="\r")
                    collect_bw.append(bw[0])
                    gradients.append(grad[0])
                if self.show_progress:
                    print("                                                                                                                                                                            ", end = '\r')
                    print('\t\t~    Stopped at: %d \t~ epsilon: %.5f \t~ gradient: %.5f \t~ max(change,grad): %.5f' % (count, bw[0], grad[0], test_threshold))
            else:
                r0 = self.optimize_params['learning_rate'] / abs(grad)
                rate = speed_list[self.optimize_params['speed']]
                count = 0
                grad0 = grad
                while count < self.optimize_params['max_iter']:
                    bw = bw0 - rate(count, r0) * grad
                    if bw < 0 or bw is None:
                        bw = bw / 1.25
                    if count > 3:
                        if np.sign(grad)*np.sign(grad0) < 0:
                            r0 = r0 / 1.25
                        if test_threshold > self.optimize_params['epsilon']:
                            bw0 = bw
                            grad0 = grad
                        else:
                            break
                    relative = abs((bw - bw0) / bw0)
                    test_threshold = np.max([relative, abs(grad)])
                    grad = optimize.approx_fprime(bw0, self.kappa_cross_validation_error, 10 ** (-100))
                    count += 1
                    collect_bw.append(bw[0])
                    gradients.append(grad[0])
            opt_bw = bw[0]
            opt_risk = self.kappa_cross_validation_error(opt_bw)
            self.optimize_outputs = {
                'opt_method' : 'grad',
                'opt_bandwidth' : opt_bw,
                'bandwidth_collection' : collect_bw,
                'gradients': gradients,
                'kappa_cv_errors': np.array([self.kappa_cross_validation_error(b) for b in collect_bw])
            }
        return self

    def predict(self, X, bandwidth = None):
        if bandwidth is None:
            bandwidth = self.optimize_outputs['opt_bandwidth']
        if self.as_predictions:
            self.pred_features_test = X
        else: 
            self.pred_features_test = {}
            for machine in self.estimator_names:
                self.pred_features_test[machine] = self.basic_estimators[machine].predict(X)
        self.pred_features_x_test = pd.DataFrame(self.pred_features_test)
        list_kernels = {
            'gaussian' : lambda x: np.exp(-x*bandwidth),
            'radial' : lambda x: np.exp(-x*bandwidth),
            'epanechnikov' : lambda x: (1 - x*bandwidth) * (x*bandwidth < 1),
            'biweight' : lambda x: (1-x*bandwidth) ** 2 * (x*bandwidth < 1),
            'triweight' : lambda x: (1-x*bandwidth) ** 3 * (x*bandwidth < 1),
            'triangular' : lambda x: (1-np.abs(x*bandwidth)) * (x*bandwidth < 1),
            'cobra' : lambda x: np.array(x*bandwidth),
            'naive' : lambda x: np.array(x*bandwidth)
        }
        self.distances(pred_k=self.pred_X_l, pred_l=self.pred_features_x_test, distance=self.distance, fold=None)
        if self.kernel in ['cobra', 'naive']:
            D_k = (list_kernels[self.kernel](np.float32(self.distance_matrix[self.distance])) <= (self.optimize_outputs['number_retained_estimators'])/self.number_estimators)
            
        else:
            D_k = list_kernels[self.kernel](np.array(self.distance_matrix[self.distance], dtype=np.float32))
        D_k_ = np.sum(D_k, axis=0, dtype=np.float32)
        D_k_[D_k_ == 0] = np.Inf
        res = np.matmul(self.y_l_, D_k)/D_k_
        res[res == 0] = res[res != 0].mean()
        self.test_prediction = res
        return res
    def plot_learning_curve(self, y_test = None,  fig_type = 'qq'):
        if (y_test is not None) and (fig_type in ['qq', 'qq-plot', 'qqplot', 'QQ-plot', 'QQplot']):
            plt.figure(figsize=(7, 3))
            plt.plot(y_test, y_test, 'r')
            plt.scatter(y_test, self.test_prediction)
            plt.xlabel('y_test')
            plt.ylabel('prediction')
            plt.title('QQ-plot: actual Vs prediction')
            plt.legend()
            plt.show()
        else:
            if self.optimize_outputs['opt_method'] == 'grid':
                if self.kernel in ['naive', 'cobra']:
                    num_estimators, bandwidths = np.meshgrid(list(range(0,self.number_estimators+1,1)), self.optimize_params['bandwidth_list'])
                    err = self.optimize_outputs['kappa_cv_errors']
                    num_opt = self.optimize_outputs['number_retained_estimators']
                    band_opt = self.optimize_outputs['opt_bandwidth']
                    fig = plt.figure(figsize=(10,6))
                    axs = fig.add_subplot(projection='3d')
                    surf = axs.plot_surface(bandwidths, num_estimators, err, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                    axs.plot(band_opt, num_opt, self.optimize_outputs['kappa_cv_errors'][self.optimize_outputs['opt_index'], num_opt], 'o')
                    axs.set_title("Errors Vs bandwidth and number of estimators ("+ str(self.kernel)+ " kernel)")
                    axs.set_xlabel("bandwidth")
                    axs.set_ylabel("number of estimators")
                    axs.set_zlabel("Kappa cross-validation error")
                    axs.view_init(30, 60)
                    plt.show()
                else:
                    plt.figure(figsize=(7, 3))
                    plt.plot(self.optimize_params['bandwidth_list'], self.optimize_outputs['kappa_cv_errors'])
                    plt.title('Kappa cross-validation error Vs bandwidth paramter (grid search)')
                    plt.xlabel('bandwidth')
                    plt.ylabel('error')
                    plt.scatter(self.optimize_outputs['opt_bandwidth'], self.optimize_outputs['kappa_cv_errors'][self.optimize_outputs['opt_index']], c = 'r')
                    plt.vlines(x=self.optimize_outputs['opt_bandwidth'], ymin=self.optimize_outputs['kappa_cv_errors'][self.optimize_outputs['opt_index']]/5, ymax=self.optimize_outputs['kappa_cv_errors'][self.optimize_outputs['opt_index']], colors='r', linestyles='--')
                    plt.hlines(y=self.optimize_outputs['kappa_cv_errors'][self.optimize_outputs['opt_index']], xmin=0, xmax=self.optimize_outputs['opt_bandwidth'], colors='r', linestyles='--')
                    plt.show()
            else:
                plt.figure(figsize=(7, 3))
                plt.plot(range(len(self.optimize_outputs['bandwidth_collection'])), self.optimize_outputs['bandwidth_collection'])
                plt.title('Kappa cross-validation error Vs bandwidth parameter (gradient descent)')
                plt.xlabel('iteration')
                plt.ylabel('bandwidth')
                plt.hlines(y=self.optimize_outputs['bandwidth_collection'][-1], xmin=0, xmax=self.optimize_params['max_iter'], colors='r', linestyles='--')
                plt.show()