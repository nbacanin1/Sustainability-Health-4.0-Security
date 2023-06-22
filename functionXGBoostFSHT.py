
#XGBOOST funkcija za feature selection i tuning


import math
import numpy as np
from utilities.load_dataset import load_dataset
from ml_models.ELM import  ELM
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
#ovo je za cohen kappa score
from sklearn.metrics import cohen_kappa_score

class XGBoostFunction:
    def __init__(self, D, no_classes,nfeatures, fitness_type,alpha,x_train,x_test,y_train,y_test,intParams,bounds,features_list=None):
        #D je broj parametara resenja (broj hiperparametara koji se optimizuju)
        #bounds je dictionary sa lower and upper bounds, ovaj dictionary se ubacuje iz spoljnog koda
        #parametri za tuning
        # x[0] - learning rate (lr), eta parametar float
        # x[1] - min_child_weight, float
        #x[2] - subsample, float
        #x[3] - collsample_bytree
        #x[4] - max_depth
        #x[5] - gamma

        #postavljamo datasetove
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.y_train1  = np.zeros(shape=len(y_train))
        self.y_test1  = np.zeros(shape=len(y_test))

        self.nfeatures = nfeatures

        self.intParams = intParams

        # koji objective koristimo, ovo mora biti prvi parametar koji se vraca
        # moze kombinacija broja features i error rate-a, posto se radi o minimizaciji
        # fitness type - 0 je samo error, a 1 je composite error i number of features
        self.fitness_type = fitness_type
        # ovo su weighted coefficient za error i broj selektovanih features
        self.alpha = alpha
        self.beta = 1 - alpha  # alpha je obicnao za error, a beta je obicno za broj feature-a

        # ako se radi bez feature selection, onda se prosledjuje features list
        # dakle, ako je features_list=None (po default), onda se radi fs i optimizacija, a ako ima necega, onda se radi samo optimizacija
        # za svaki slucaj konvertujemo sve elemente u int
        self.features_list = [int(x) for x in features_list]
        print(self.features_list)


        #converting y_test and y_train to single label classification

        #fajl za load datasetova kodira target varijablu, pa pravi kolone u zavisnosti broja klasa, npr. 0 0 1, 0 1 0, itd.
        #medjutim, XGBoost kao i DMatrix ne koriste one hot encoding, vec se koriste u target koloni klase, 0,1,2,3, itd.
        #zbog toga pravimo pomocne setove y_test1 i y_train1 gde se koriste 0,1,2,3,4 za target, umesto one hot encoding

        for i in range(len(self.y_test)):
            self.y_test1[i] = np.argmax(self.y_test[i])


        for i in range(len(self.y_train1)):
            self.y_train1[i] = np.argmax(self.y_train[i])

        #print(f'y_test shape: {self.y_test1.shape}' )
        #print(f'y_train shape: {self.y_train1.shape}')

        #ovo je za DMatrix

        #koristimo bez encodinga za target
        #pravimo Dmatrix jer XGBoost radi bolje
        #self.d_train = xgb.DMatrix(self.x_train, self.y_train1)
        #self.d_test = xgb.DMatrix(self.x_test, self.y_test1)


        self.no_classes = no_classes
        self.D = D

        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = "XGBoost Function"

        # pomocna promenljiva koja pokazuje da li radimo feature selection
        # postavljamo ovu promenjivu samo na osnovu neatures
        if self.nfeatures > 0:
            self.fs = True
        else:
            self.fs = False


        if self.fs:
            for i in range(self.nfeatures):
                self.lb[i] = 0
                self.ub[i] = 1

            self.lb[self.nfeatures] = bounds['lb_lr']  # lower bound of learning rate
            self.ub[self.nfeatures] = bounds['ub_lr']  # upper bound of learning rate
            self.lb[self.nfeatures + 1] = bounds['lb_mcw']  # lower bound of min_child_weight
            self.ub[self.nfeatures + 1] = bounds['ub_mcw']  # upper bound of min_child_weight
            self.lb[self.nfeatures + 2] = bounds['lb_ss']  # lower bound of sub-sample
            self.ub[self.nfeatures + 2] = bounds['ub_ss']  # upper bound of sub-sample
            self.lb[self.nfeatures + 3] = bounds['lb_cst']  # lower bound of coll sample by tree
            self.ub[self.nfeatures + 3] = bounds['ub_cst']  # upper bound of coll sample by tree
            self.lb[self.nfeatures + 4] = bounds['lb_md']  # lower bound of maximum depth of tree, ovo je int
            self.ub[self.nfeatures + 4] = bounds['ub_md']  # upper bound of maximum depth of tree, ovo je int
            self.lb[self.nfeatures + 5] = bounds['lb_g']  # lower bound of maximum depth of tree
            self.ub[self.nfeatures + 5] = bounds['ub_g']  # upper bound of maximum depth of tree

            # SADA DEFINISEMO PARAMETRE AKO SE NE RADI FEATURE SELECTION
        else:

            self.lb[self.nfeatures] = bounds['lb_lr']  # lower bound of learning rate
            self.ub[self.nfeatures] = bounds['ub_lr']  # upper bound of learning rate
            self.lb[self.nfeatures + 1] = bounds['lb_mcw']  # lower bound of min_child_weight
            self.ub[self.nfeatures + 1] = bounds['ub_mcw']  # upper bound of min_child_weight
            self.lb[self.nfeatures + 2] = bounds['lb_ss']  # lower bound of sub-sample
            self.ub[self.nfeatures + 2] = bounds['ub_ss']  # upper bound of sub-sample
            self.lb[self.nfeatures + 3] = bounds['lb_cst']  # lower bound of coll sample by tree
            self.ub[self.nfeatures + 3] = bounds['ub_cst']  # upper bound of coll sample by tree
            self.lb[self.nfeatures + 4] = bounds['lb_md']  # lower bound of maximum depth of tree, ovo je int
            self.ub[self.nfeatures + 4] = bounds['ub_md']  # upper bound of maximum depth of tree, ovo je int
            self.lb[self.nfeatures + 5] = bounds['lb_g']  # lower bound of maximum depth of tree
            self.ub[self.nfeatures + 5] = bounds['ub_g']  # upper bound of maximum depth of tree

        #podesavanje granica parametara
        #typical values: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
        #za razliku od implementacije bez feature selection, ovde se sve postavlja u odnosu na broj features,
        #npr. learning rate je index nfeatures, pa sledeci je nfeatures+1, itd.





        #self.x_train, self.y_train, self.x_test, self.y_test = load_dataset(path_train, path_test, self.no_classes, normalize, test_size)
        '''
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test = y_test
        '''
        # coverting train and test datasets to DMatrix da bi radilo brze

        #self.d_train = xgb.DMatrix(self.x_train, self.y_train)
        #self.d_test = xgb.DMatrix(self.x_test, self.y_test)


        #ovo nam koristi za solution
        self.y_test_length = len(y_test)

        '''
        self.D = D
        #D is weighs*input features length + 1
        #in second experiment we use NN as well as the first argument of solution
        self.lb_w = lb_w
        self.ub_w = ub_w
        self.lb_nn= lb_nn
        self.ub_nn = ub_nn
         #ub[0],lb[0] is for hidden neurons size

        self.lb[0] = lb_nn
        self.ub[0] = ub_nn


        for i in range(1, self.D):
            self.lb[i] = self.lb_w
            self.ub[i] = self.ub_w
        #self.lb[0] = self.lb_nn
        #self.ub[0] = self.ub_nn
        '''

    def function(self, x):
        x = np.array(x)

        # parametri za XGBoost
        params = {
            'booster': 'gbtree',
            'max_depth': int(round(x[self.nfeatures+4],0)),
            'learning_rate': x[self.nfeatures+0],
            'sample_type': 'uniform',
            'normalize_type': 'tree',
            #'objective': 'binary:hinge',
            #'objective': 'multi:softprob', #logistic function da bismo dobili y_proba
            #'objective':'binary:logistic',
            'objective':'multi:softprob',
            'rate_drop': 0.1,
            'n_estimators': 100,
            'min_child_weight': x[self.nfeatures+1],
            'subsample': x[self.nfeatures+2],
            'colsample_bytree': x[self.nfeatures+3],
            'random_state': 23,
            'seed': 23,
            'silent': 1, #logging mode za silent
            'num_class':self.no_classes,
            'gamma':x[self.nfeatures+5],
            'verbosity':0
            # 'num_boost_round':10
        }
        #params = {'max_depth':10,'num_class':self.no_classes, 'rate_drop': 0.1,
            #'#n_estimators':100}

        #sada se prvo kao u fazi inicijalizacije igramo sa selektovanim features i dmatrix
        #sada gledamo prvo da li koristimo feature selection
        if self.fs:

            x_train_fs = self.x_train[:, x[0:self.nfeatures] == 1]
            x_test_fs = self.x_test[:, x[0:self.nfeatures] == 1]

            # broj odabranih featrues
            feature_size = np.sum(x[0:self.nfeatures])

            if(feature_size==0):
                return 1,1,0,0,0

            #sada pravimo novi dmatrix na osnovu odabranih features

            self.d_train = xgb.DMatrix(x_train_fs, self.y_train1)
            self.d_test = xgb.DMatrix(x_test_fs, self.y_test1)
        # sada gledamo ako ne radimo feature selection i onda gledamo unapred prosledjenu listu features, koja predstavlja koji se features uzimaju u obzir
        else:

            # print("OVO RADI")
            x_train_fs = self.x_train[:, np.array(self.features_list) == 1]
            x_test_fs = self.x_test[:, np.array(self.features_list) == 1]
            # broj odabranih featrues
            feature_size = np.sum(self.features_list)
            # print(x_train_fs.shape[1])
            # print(x_train_fs)
            # ovo za svaki slucaj ako neko prosledi sve nule
            if feature_size == 0:
                return 1, 1, 0, 0, 0
            self.d_train = xgb.DMatrix(x_train_fs, self.y_train1)
            self.d_test = xgb.DMatrix(x_test_fs, self.y_test1)


        '''
        xgb_clf = xgb.train(params, self.d_train)

        y_proba = xgb_clf.predict(self.d_test)

        y = np.zeros((len(y_proba), y_proba.shape[1]))
        for i in range(len(y_proba)):
            y[i][np.argmax(y_proba[i])] = 1

        acc = np.round(accuracy_score(self.y_test, y_proba) * 100, 2)

        error = 1-acc

        return (error,y_proba,y)
        '''
        correct = 0
        xgb_clf = xgb.train(params, self.d_train)
        y_proba = xgb_clf.predict(self.d_test)
        total = y_proba.shape[0]
        #print(y_proba)
        #print(y_proba.shape)
        for i in range(total):
            predicted = np.argmax(y_proba[i])
            # test = np.argmax(self.y_test[i])
            test = self.y_test1[i]
            correct = correct + (1 if predicted == test else 0)
        #sada pravimo i predkcije po klasama za kohen kappa
        y_cappa = np.zeros(len(y_proba))
        for i in range(total):
            y_cappa[i] = np.argmax(y_proba[i])


        y = np.zeros((len(y_proba), y_proba.shape[1]))
        for i in range(len(y_proba)):
            y[i][np.argmax(y_proba[i])] = 1
        # classification error
        error = round(1 - (correct / total),30)
        #print(f'error: {error}')

        #print("OVO JE Y1: ", self.y_test1)
        #print("OVO JE Y: ",y)

        #racunamo cohen kappa za statistiku, ovo je indikator
        cohen_kappa = cohen_kappa_score(y_cappa,self.y_test1)



        if self.fitness_type == 0:
            return (error,cohen_kappa,y_proba,y,feature_size,xgb_clf)
        else:
            objective = self.alpha*error + self.beta*(feature_size/self.nfeatures) #ovo je za minimization
            return (objective,error,y_proba,y,feature_size,xgb_clf)



