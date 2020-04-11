from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from MLP import Fct_acti, Dense, Fct_cout, MultiPerceptron
from tools_test import interval, stats, plot_stats, plot_val_cost
import pickle

"""Script pour faire les graph automatiquement pour mesurer les influences de :

    - learning_rate
    - profondeur
    - largeur
    - optimizer
    - initialisation
    - nombre de données

"""


(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

x_train_mnist = x_train_mnist.reshape(60000, 784) / 255.
x_test_mnist = x_test_mnist.reshape(10000, 784) / 255.
y_train_mnist = np_utils.to_categorical(y_train_mnist)
y_test_mnist = np_utils.to_categorical(y_test_mnist)

x_train, y_train = x_train_mnist, y_train_mnist
x_test, y_test = x_test_mnist, y_test_mnist


#AJOUTER UN TIMER

#test de l'effet de la learning rate
def test_lr ():

    L_samples = []
    L_val_cost_k_i = []
    L_meta_data = []

    N = 50 #nombre d'essais pour moyenner

    L_lr = [5, 1, 0.1, 0.01, 0.001]

    for i in range (len(L_lr)):

        lr = L_lr[i]

        L_val_cost_k = []

        for k in range(N):

            print("START ETAPE "+str(i+1)+"/"+str(len(L_lr))+" - essai : "+str(k+1)+"/"+str(N)+" - lr : "+str(lr))

            input = Dense(28*28, "sigmoid", _reg = 0.0001)
            couche1 = Dense(200, "sigmoid", _reg = 0.0001)
            Lcouches = [input, couche1]
            loss1 = Fct_cout("mse")
            MP = MultiPerceptron(output_size = 10, _layers = Lcouches, loss = loss1)
            L_val_cost = MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 1000, lr_dict = {0 : lr}, momentum_dict = {0 : 0.0001}, graph=0)
            acc = np.round(np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2)

            L_samples.append(acc)
            L_meta_data.append({'depth' : 3, 'nb_neurons' : 200, 'acti' : "sigmoid", 'cost' : 'mse', 'lr' : lr, 'regu' : 0.0001, 'epochs': 200, 'batch_size' : 32, 'momentum' : 0.0001})
            L_val_cost_k.append(L_val_cost)

            print("FIN ETAPE "+str(i+1)+"/"+str(len(L_lr))+"essai "+str(k+1)+"/"+str(N)+" - précision : "+str(acc))
        
        L_val_cost_k_i.append([L_val_cost_k, "learning rate : "+str(lr)])

    tested_param = 'lr'
    log = True

    return L_samples, L_meta_data, tested_param, L_val_cost_k_i, log


#test de l'effet de la profondeur
def test_depth ():

    L_samples = []
    L_meta_data = []
    L_val_cost_k_i = []

    N = 30 #nombre d'essais pour moyenner
    nb_couches = 4

    for i in range (1, nb_couches+1):

        L_val_cost_k = []

        for k in range(N):

            print("START ETAPE "+str(i)+"/"+str(nb_couches+1)+" - essai : "+str(k+1)+"/"+str(N)+" - nb_layers : "+str(i))

            input = Dense(28*28, "sigmoid", _reg = 0.0001)
            Lcouches = [input]
            for j in range (1, i):
                Lcouches.append(Dense(60, "sigmoid", _reg = 0.0001))
            loss1 = Fct_cout("mse")
            MP = MultiPerceptron(output_size = 10, _layers = Lcouches, loss = loss1)
            L_val_cost = MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 1000, lr_dict = {0 : 1}, momentum_dict = {0 : 0.0001}, optimizer = 'sgd', graph=0)
            acc = np.round(np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2)

            L_samples.append(acc)
            L_meta_data.append({'depth' : i, 'nb_neurons' : 60, 'acti' : "sigmoid", 'cost' : 'mse', 'lr' : 1, 'regu' : 0.0001, 'epochs': 1000, 'batch_size' : 32, 'momentum' : 0.0001})
            L_val_cost_k.append(L_val_cost)

            print("FIN ETAPE "+str(i)+"/"+str(nb_couches+1)+" - essai : "+str(k+1)+"/"+str(N)+" - précision : "+str(acc))

        L_val_cost_k_i.append([L_val_cost_k, "depth : "+str(i)])
        #print(L_val_cost_k_i)

    tested_param = 'depth'
    log = False

    return L_samples, L_meta_data, tested_param, L_val_cost_k_i, log


#test de l'effet de la largeur
def test_width ():

    L_samples = []
    L_val_cost_k_i = []
    L_meta_data = []

    N = 30 #nombre d'essais pour moyenner

    L_nb_neurons = [1, 10, 100, 1000]

    for i in range (len(L_nb_neurons)):

        nb_neurons = L_nb_neurons[i]

        L_val_cost_k = []

        for k in range(N):

            print("START ETAPE "+str(i+1)+"/"+str(len(L_nb_neurons))+" - essai : "+str(k+1)+"/"+str(N)+" - largeur_couche : "+str(nb_neurons))

            input = Dense(28*28, "sigmoid", _reg = 0.0001)
            couche1 = Dense(nb_neurons, "sigmoid", _reg = 0.0001)
            Lcouches = [input, couche1]
            loss1 = Fct_cout("mse")
            MP = MultiPerceptron(output_size = 10, _layers = Lcouches, loss = loss1)
            L_val_cost = MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 1000, lr_dict = {0 : 1}, momentum_dict = {0 : 0.0001}, optimizer = 'sgd', graph=0)
            acc = np.round(np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2)

            L_samples.append(acc)
            L_meta_data.append({'depth' : 2, 'nb_neurons' : nb_neurons, 'acti' : "sigmoid", 'cost' : 'mse', 'lr' : 1, 'regu' : 0.0001, 'epochs': 1000, 'batch_size' : 32, 'momentum' : 0.0001})
            L_val_cost_k.append(L_val_cost)

            #file = open("FIN ETAPE "+str(i+1)+"/"+str(len(L_nb_neurons))+"essai "+str(k+1)+"/"+str(N), 'wb')
            #pickle.dump([L_val_cost_k, L_meta_data], file)

            print("FIN ETAPE "+str(i+1)+"/"+str(len(L_nb_neurons))+"essai "+str(k+1)+"/"+str(N)+" - précision : "+str(acc))
        
        L_val_cost_k_i.append([L_val_cost_k, "largeur_couche : "+str(nb_neurons)])
        

    tested_param = 'nb_neurons'
    log = True

    return L_samples, L_meta_data, tested_param, L_val_cost_k_i, log


#test de l'effet de l'optimizer
def test_opti ():

    L_samples = []
    L_val_cost_k_i = []
    L_meta_data = []

    N = 10 #nombre d'essais pour moyenner

    L_optimizer = ["sgd", "rmsprop", "adam"]
    L_alpha = [1, 0.01, 0.005]

    for i in range (len(L_optimizer)):

        opti = L_optimizer[i]
        alpha = L_alpha[i]

        L_val_cost_k = []

        for k in range(N):

            print("START ETAPE "+str(i+1)+"/"+str(len(L_optimizer))+" - essai : "+str(k+1)+"/"+str(N)+" - opti : "+str(opti))

            input = Dense(28*28, "sigmoid", _reg = 0.0001)
            couche1 = Dense(200, "sigmoid", _reg = 0.0001)
            couche2 = Dense(100, "sigmoid", _reg = 0.0001)
            Lcouches = [input, couche1, couche2]
            loss1 = Fct_cout("mse")
            MP = MultiPerceptron(output_size = 10, _layers = Lcouches, loss = loss1)
            L_val_cost = MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 2000, lr_dict = {0 : alpha}, momentum_dict = {0 : 0.0001}, optimizer = opti, graph=0)
            acc = np.round(np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2)

            L_samples.append(acc)
            L_meta_data.append({'depth' : 3, 'nb_neurons' : 200, 'acti' : "sigmoid", 'cost' : 'mse', 'lr' : alpha, 'regu' : 0.0001, 'epochs': 2000, 'batch_size' : 32, 'momentum' : 0.0001, 'optimizer' : opti})
            L_val_cost_k.append(L_val_cost)

            #file = open("FIN ETAPE "+str(i+1)+"/"+str(len(L_nb_neurons))+"essai "+str(k+1)+"/"+str(N), 'wb')
            #pickle.dump([L_val_cost_k, L_meta_data], file)

            print("FIN ETAPE "+str(i+1)+"/"+str(len(L_optimizer))+"essai "+str(k+1)+"/"+str(N)+" - précision : "+str(acc))
        
        L_val_cost_k_i.append([L_val_cost_k, "opti : "+str(opti)])
        

    tested_param = 'optimizer'
    log = False

    return L_samples, L_meta_data, tested_param, L_val_cost_k_i, log

#test de l'effet de l'initialisation
def test_init ():
    
    #tester avec différent optimizers
    #(normal(0, 1)) n'est pas bien, normal(0, 0,1) mache mieux

    L_samples = []
    L_val_cost_k_i = []
    L_meta_data = []

    N = 40 #nombre d'essais pour moyenner

    L_init_tuning_unif = [['uniform', -0.01, 0.01], ['uniform', -0.1, 0.1], ['uniform', -1, 1], ['uniform', -10, 10]]
    L_init_tuning_normal = [['normal', 0, 0.01], ['normal', 0, 0.1], ['normal', 0, 1], ['normal', 0, 10]]
    L_init_extrem = [['normal', 0, 0.1], ['normal', 0, 0], ['uniform', -1, 1], ['uniform', 0, 1], ['uniform', -1, 0], ['normal', 1, 0.1], ['uniform', -1, 10]]
    L_init = L_init_tuning_normal

    for i in range (len(L_init)):

        init = L_init[i]

        L_val_cost_k = []

        for k in range(N):

            print("START ETAPE "+str(i+1)+"/"+str(len(L_init))+" - essai : "+str(k+1)+"/"+str(N)+" - init : "+str(init))

            input = Dense(28*28, "sigmoid", _reg = 0.0001)
            couche1 = Dense(200, "sigmoid", _reg = 0.0001)
            couche2 = Dense(100, "sigmoid", _reg = 0.0001)
            Lcouches = [input, couche1, couche2]
            loss1 = Fct_cout("mse")
            MP = MultiPerceptron(output_size = 10, _layers = Lcouches, loss = loss1, init =  init)
            L_val_cost = MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 500, lr_dict = {0 : 1}, momentum_dict = {0 : 0.0001}, optimizer = 'sgd', graph=0)
            acc = np.round(np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2)

            L_samples.append(acc)
            L_meta_data.append({'depth' : 3, 'nb_neurons' : 200, 'acti' : "sigmoid", 'cost' : 'mse', 'lr' : 1, 'regu' : 0.0001, 'epochs': 500, 'batch_size' : 32, 'momentum' : 0.0001, 'optimizer' : 'sgd', 'init' : str(init) })
            L_val_cost_k.append(L_val_cost)

            #file = open("FIN ETAPE "+str(i+1)+"/"+str(len(L_nb_neurons))+"essai "+str(k+1)+"/"+str(N), 'wb')
            #pickle.dump([L_val_cost_k, L_meta_data], file)

            print("FIN ETAPE "+str(i+1)+"/"+str(len(L_init))+"essai "+str(k+1)+"/"+str(N)+" - précision : "+str(acc))
        
        L_val_cost_k_i.append([L_val_cost_k, "initialisation : "+str(init)])
        

    tested_param = 'init'
    log = False

    return L_samples, L_meta_data, tested_param, L_val_cost_k_i, log

#test de l'effet du nombre de données
def test_nb_datas ():
    
    #tester avec différent optimizers
    #(normal(0, 1)) n'est pas bien, normal(0, 0,1) mache mieux

    L_samples = []
    L_val_cost_k_i = []
    L_meta_data = []

    N = 40 #nombre d'essais pour moyenner

    L_nb_datas = [60000, 10000, 1000, 100, 10]

    for i in range (len(L_nb_datas)):


        x_train = x_train_mnist[:L_nb_datas[i]]
        y_train = y_train_mnist[:L_nb_datas[i]]

        L_val_cost_k = []

        for k in range(N):

            print("START ETAPE "+str(i+1)+"/"+str(len(L_nb_datas))+" - essai : "+str(k+1)+"/"+str(N)+" - nb_datas : "+str(len(x_train)))

            input = Dense(28*28, "sigmoid", _reg = 0.0001)
            couche1 = Dense(200, "sigmoid", _reg = 0.0001)
            couche2 = Dense(100, "sigmoid", _reg = 0.0001)
            Lcouches = [input, couche1, couche2]
            loss1 = Fct_cout("mse")
            MP = MultiPerceptron(output_size = 10, _layers = Lcouches, loss = loss1, init =  ['uniform', -1, 1])
            L_val_cost = MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 1000, lr_dict = {0 : 1}, momentum_dict = {0 : 0.0001}, optimizer = 'sgd', graph=0)
            acc = np.round(np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2)

            L_samples.append(acc)
            L_meta_data.append({'depth' : 3, 'nb_neurons' : 200, 'acti' : "sigmoid", 'cost' : 'mse', 'lr' : 1, 'regu' : 0.0001, 'epochs': 1000, 'batch_size' : 32, 'momentum' : 0.0001, 'optimizer' : 'sgd', 'init' : "uniform", "nb_datas" : str(len(x_train))})
            L_val_cost_k.append(L_val_cost)

            #file = open("FIN ETAPE "+str(i+1)+"/"+str(len(L_nb_neurons))+"essai "+str(k+1)+"/"+str(N), 'wb')
            #pickle.dump([L_val_cost_k, L_meta_data], file)

            print("FIN ETAPE "+str(i+1)+"/"+str(len(L_nb_datas))+"essai "+str(k+1)+"/"+str(N)+" - precision : "+str(acc))
        
        L_val_cost_k_i.append([L_val_cost_k, "échantillons d'apprentissage : "+str(len(x_train))])
        

    tested_param = 'nb_datas'
    log = False

    return L_samples, L_meta_data, tested_param, L_val_cost_k_i, log


L_samples, L_meta_data, tested_param, L_L_val_cost, log = test_depth ()
D_group_stat, D_groups_metadata = stats (L_samples, L_meta_data, tested_param, 95)
plot_stats (D_group_stat, D_groups_metadata, xlabel = tested_param, log = False, label = "accuracy")
plot_val_cost (L_L_val_cost, D_groups_metadata, "epoch")