import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
import random
import sys
from multiprocessing import Pool
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
class NullDevice():
    def write(self, s):
        pass
class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.a), n_samples)

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)
        self.b = -self.b #done so that we can use -b in project
        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) - self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict - self.b

    def predict(self, X):
        return np.sign(self.project(X))

class SMO(SVM):

    def __init__(self, kernel=linear_kernel, C=1, tol = 0.001):
        SVM.__init__(self, kernel, C)
        self.tol = tol
    def __take_step(self, i1, i2):
        if i1 == i2:
            return False
        tol = self.tol
        alpha1, alpha2 = self.alpha[i1], self.alpha[i2]
        p1, p2 = self.X[i1], self.X[i2]
        y1, y2 = self.y[i1], self.y[i2]
        E1, E2 = self.Ecache[i1], self.Ecache[i2]
        s = int(y1) * int(y2)
        if s < 0:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        if L == H:
            return False

        kernel = self.kernel
        k11 = kernel(p1, p1)
        k12 = kernel(p1, p2)
        k22 = kernel(p2, p2)

        eta = k11 + k22 - (2 * k12)
        if eta > 0:
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            alpha2_new = max(alpha2_new, L)
            alpha2_new = min(alpha2_new, H)
        else:
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - alpha2 * k22 - s * alpha1 * k12
            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)
            L_obj = sum([
                    L1*f1, L*f2, 0.5*L1*L1*k11, 0.5*L*L*k22, s*L*L1*k12
                    ])
            H_obj = sum([
                    H1*f1, H*f2, 0.5*H1*H1*k11, 0.5*H*H*k22, s*H*H1*k12
                    ])
            if (H_obj - L_obj) > tol:
                alpha2_new = L
            elif (L_obj - H_obj) > tol:
                alpha2_new = H
            else:
                alpha2_new = alpha2
        alpha2_new = max(0, alpha2_new)
        alpha2_new = min(self.C, alpha2_new)

        if abs(alpha2_new - alpha2) < (tol * (alpha2_new + alpha2 + tol)):
            return False
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)
        alpha1_new = max(0, alpha1_new)
        alpha1_new = min(self.C, alpha1_new)

        # Did the new alphas changed the boundedness?
        if alpha1_new in [0, self.C]:
            if not self.alpha_bound[i1]:
                self.alpha_bound[i1] = True
                self.alpha_non_bound -= 1
        else:
            if self.alpha_bound[i1]:
                self.alpha_bound[i1] = False
                self.alpha_non_bound += 1
        if alpha2_new in [0, self.C]:
            if not self.alpha_bound[i2]:
                self.alpha_bound[i2] = True
                self.alpha_non_bound -= 1
        else:
            if self.alpha_bound[i2]:
                self.alpha_bound[i2] = False
                self.alpha_non_bound += 1

        old_b = self.b
        b1 = old_b+E1+y1*k11*(alpha1_new-alpha1)+y2*k12*(alpha2_new-alpha2)
        b2 = old_b+E2+y1*k12*(alpha1_new-alpha1)+y2*k22*(alpha2_new-alpha2)


        if not self.alpha_bound[i1]:
            self.b = b1
        elif not self.alpha_bound[i2]:
            self.b = b2
        else:
            self.b = 0.5 * (b1 +b2)

        for i in range(self.n_samples):
            k1i = kernel(self.X[i], p1)
            k2i = kernel(self.X[i], p2)
            self.Ecache[i] += y1*k1i*(alpha1_new-alpha1)+y2*k2i*(alpha2_new-alpha2) + old_b - self.b

        self.alpha[i1] = alpha1_new
        self.alpha[i2] = alpha2_new
        return True
        
    def __heuristic(self, E2):
        candidates = [
            (abs(E2 - self.Ecache[i]), i) 
            for i in range(self.n_samples)
            if not self.alpha_bound[i]
            ]
        if not candidates:
            return -1
        return min(candidates, key = lambda x: x[0])[1]
    def __examine_example(self, i2):
        p2 = self.X[i2]
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        E2 = self.Ecache[i2]
        r2 = E2 * y2
        tol = self.tol
        if (r2 < -tol and  alpha2 < self.C) or (r2 > tol and alpha2 > 0):
            if self.alpha_non_bound > 1:
                i1 = self.__heuristic(E2)
                if i1 >= 0: # only if all alpha's are bounded
                    if self.__take_step(i1, i2):
                        return True
            # if none of the take_step succeeded, we choose a random
            # i1 from those points whose alpha are bounded and keep trying
            all_points = range(self.n_samples)
            random.shuffle(all_points)
            bounded_points = []

            for i1 in all_points:
                if not self.alpha_bound[i1]:
                    if self.__take_step(i1, i2):
                        return True
                else:
                    bounded_points.append(i1)
            # if still take_step did not succeed we try rest of points
            for i1 in bounded_points:
                if self.__take_step(i1, i2):
                    return True

        return False
            
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.alpha_bound = [True] * n_samples
        self.alpha_non_bound = 0
        self.Ecache = self.y * -1
        self.b = 0

        examine_all = True
        any_changed = False
        loop_counter = 0
        while (any_changed) or examine_all:
            loop_counter += 1
            any_changed = False
            if examine_all:
                for i in range(n_samples):
                    any_changed |= self.__examine_example(i)
            else:
                for i in range(n_samples):
                    if not self.alpha_bound[i]:
                        any_changed |= self.__examine_example(i)
            if examine_all:
                examine_all = False
            elif not any_changed:
                examine_all = True
            if (loop_counter-1)%100 == 0:
                print loop_counter, self.b, self.alpha_non_bound
        # calculation of weight vector 
        sv =  self.alpha > 0
        self.a = self.alpha[sv]
        self.sv = self.X[sv]
        self.sv_y = self.y[sv]
        print "%d support vectors out of %d points" % (len(self.a), n_samples)
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for i in range(len(self.alpha)):
                self.w += self.alpha[i] * self.y[i] * self.X[i]
        else:
            self.w = None

        print "w", self.w
        
    

if __name__ == "__main__" or True:
    import pylab as p
    def get_dataset_data():
        ds_file = open('heart_scale') #open(sys.argv[1])
        neglected_index = [11]
        ds_raw = ds_file.readlines()
        data_set = []
        data_set_pos = []
        data_set_neg = []
        missing = []
        for k,x in enumerate(ds_raw):
            feat_vect = dict([(int(y.split(':')[0]),float(y.split(':')[1],)) 
                              for y in x.split(' ')[1:-1]])
            svm_class = int(x.split(' ')[0])
            n_feat = max(feat_vect.keys())
            feat_row = [0]*n_feat
            for i in sorted(feat_vect.keys()):
                if i not in neglected_index:
                    feat_row[i-1] = feat_vect[i]
            miss = set(range(1, n_feat)) - (set(feat_vect.keys()).union(set(neglected_index)))
            if (miss) :
                missing.append(k)
                continue
            data_set.append((svm_class,feat_row))
            if svm_class == 1 :
                data_set_pos.append(feat_row)
            else:
                data_set_neg.append(feat_row)
        pos_matrix = np.delete(np.array(data_set_pos), [10], axis = 1)
        neg_matrix = np.delete(np.array(data_set_neg), [10], axis = 1)
        return pos_matrix, np.ones(len(pos_matrix)), neg_matrix, np.ones(len(neg_matrix)) * -1

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        #print X1, y1, X2, y2
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test
    def get_train(X1, y1, X2, y2):
        X1_train = X1[:]
        y1_train = y1[:]
        X2_train = X2[:]
        y2_train = y2[:]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        data = zip(X_train, y_train)
        random.shuffle(data)
        X_train, y_train = zip(*data)
        return np.array(X_train), np.array(y_train)
    def get_test(X1, y1, X2, y2):
        X1_test = X1[:]
        y1_test = y1[:]
        X2_test = X2[:]
        y2_test = y2[:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_linear():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM()
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)
    def worker(work):
        print "WORKER"
        X = [x[0] for x in work]
        y = [x[1] for x in work]

        X = np.array(X)
        y = np.array(y)
        clf = SVM(C=1.0)
        clf.fit(X, y)
        return clf
    def split_seq(seq, size):
        newseq = []
        splitsize = 1.0/size*len(seq)
        for i in range(size):
                newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
        return newseq

    def get_train_dataset_data():
        X1, y1, X2, y2 = get_dataset_data()  
        return get_train(X1, y1, X2, y2)


    def test_dataset_parallel(n_process = 1, X_train = None, y_train = None):
        if X_train == None and  y_train == None:
            X_train, y_train = get_train_dataset_data()
        else:
            print X_train.shape
        if not len(X_train):
            return 0
        work_split = split_seq(zip(X_train, y_train), n_process)
        pool = Pool(processes=n_process)
        stdout = sys.stdout
        sys.stdout = NullDevice()
        t1 = time()
        result = pool.map(worker, work_split)
        total_time = time() - t1
        sys.stdout = stdout
        print sum([x.b for x in result])
        print sum([x.w for x in result])
        X_new =  np.concatenate([x.sv for x in result])
        y_new =  np.concatenate([x.sv_y for x in result])
        print X_new.shape, y_new.shape
        if len(result) != 1 :
            total_time += test_dataset_parallel(n_process/2, X_new, y_new)
        else:
            clf = result[0]
            orig_X, orig_y = get_train_dataset_data()
            y_predict = result[0].predict(orig_X)
            correct = np.sum(y_predict == orig_y)
            print "%d out of %d predictions correct" % (correct, len(y_predict))
            
        return total_time
           

    def test_dataset():
        X1, y1, X2, y2 = get_dataset_data()
        X_train, y_train = get_train(X1, y1, X2, y2)
        X_test, y_test = get_train(X1, y1, X2, y2)

        clf = SVM(C=1.0)

        clf.fit(X_train, y_train)
        print "B=", clf.b
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))
        print clf.w
#        plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)



    def test_non_linear():
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(gaussian_kernel)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    def test_soft():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(C=0.1)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print "%d out of %d predictions correct" % (correct, len(y_predict))

        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)

    from time import time
    test_dataset()
    t1 =test_dataset_parallel()
    t2 = test_dataset_parallel(4)
    print "Non-parallel time:", t1
    print "Paralel time:", t2
    print "Improvement:", t1/t2, "X faster"
