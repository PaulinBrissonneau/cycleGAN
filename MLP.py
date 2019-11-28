import numpy as np
from keras.datasets import mnist #juste pour la dataset
from keras.utils import np_utils #juste pour ne pas recoder le "to_categorical"
import numpy as np
import matplotlib.pyplot as plt
import time
import tools_test as tt

#####################################"
"""
Il donne 96.58% avec les conditions décrites à la fin.

Problème : mes relus ne sont pas stables
"""


(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

x_train_mnist = x_train_mnist.reshape(60000, 784) / 255.
x_test_mnist = x_test_mnist.reshape(10000, 784) / 255.
y_train_mnist = np_utils.to_categorical(y_train_mnist)
y_test_mnist = np_utils.to_categorical(y_test_mnist)

#les fonctions de cout : mse, log
class Fct_cout () :

    def __init__ (self, _loss = "mse") :
        self.loss = _loss

    def cost(self, H, Y):

        assert (len(H) == len(Y))

        if self.loss == "mse" :
            return np.mean(np.sum(((H-Y)**2/2), axis = 1), axis = 0)

        if self.loss == "log": #categorical log, pas binary log
            #print(-np.mean(np.sum(Y*np.log(H), axis = 1), axis = 0))
            return -np.mean(np.sum(Y*np.log(H), axis = 1), axis = 0)

        else :
            raise ValueError('cost pas définie (cost())')

    def cost_derivative(self, H, Y):

        assert (len(H) == len(Y))

        if self.loss == "mse" :
            return H-Y

        if self.loss == "log": #problème en H = 0 ? calculs pas surs 
            
            return H-Y

        else :
            raise ValueError('cost pas définie (cost_deriv())')


    def softmax (self, X) :
        epsi = 10e-6
        batch_soft = []

        for batch in X :
            exps= np.exp(batch)
            sum_exp = np.sum(exps)
            batch_soft.append(exps/(sum_exp+epsi))

        return np.array(batch_soft).reshape(X.shape)

#les fonctions d'activation : sigmoid, tanh, relu, softmax, Mish à voir ?
class Fct_acti ():

    def __init__(self, _acti = "sigmoid"):
        self.ac = _acti
        self.coeff_relu = 0.001

    def acti(self, x):

        if self.ac == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))

        if self.ac == "tanh":
            return np.tanh(x)

        if self.ac == "relu":
            return np.where(x>0, x, self.coeff_relu*x)

        if self.ac == "softmax":
            epsi = 10e-6
            batch_soft = []

            for batch in x :
                exps= np.exp(batch)
                sum_exp = np.sum(exps)
                batch_soft.append(exps/(sum_exp+epsi))

            return np.array(batch_soft).reshape(x.shape)

        else :
            raise ValueError('acti pas définie (acti())')

    def acti_derivative(self, x):

        if self.ac == "sigmoid":
            sig = self.acti(x)
            return sig * (1 - sig)

        if self.ac == "tanh":
            return 1/(x*x +1)

        if self.ac == "relu":
            out = np.where(x>0, 1, self.coeff_relu)
            return out

        if self.ac == "softmax":
            sm = self.acti(x)
            return sm*(1. - sm)

        else :
            raise ValueError('acti pas définie (acti_deriv())')

class Dense () :

    def __init__ (self, _taille, _acti, _reg):
        self.taille = _taille
        self.acti_fct = Fct_acti(_acti)
        self.reg = _reg #scalaire
        self.weights = None
        self.biais = None

class Dropout () : # à gérer dans la classe MultiPerceptron

    def __init__ (self, _taux):
        self.taux = _taux

class MultiPerceptron():

    def __init__(self, output_size, _layers, loss, init = ['uniform', -1, 1]):

        self.cost_fct = loss #objet Fct_cout
        self.layers = _layers

        def init_distrib (init, a, b, taille_i, taille_i1): #choix de la distribution initiale des poids
            if init == 'normal':
                #print(np.random.normal(a, b, (taille_i, taille_i1)))
                return np.random.normal(a, b, (taille_i, taille_i1))

            if init == 'uniform':
                #print(np.random.uniform(a, b, (taille_i, taille_i1)))
                return np.random.uniform(a, b, (taille_i, taille_i1))

        #initialisation de tous les poids
        for i in range(len(self.layers) -1) :
            self.layers[i].weights = init_distrib(init[0], init[1], init[2], self.layers[i].taille, self.layers[i+1].taille)
            self.layers[i].biais = np.zeros(self.layers[i+1].taille)
        self.layers[len(self.layers)-1].weights = init_distrib(init[0], init[1], init[2], self.layers[len(self.layers)-1].taille, output_size)
        self.layers[len(self.layers)-1].biais = np.zeros(output_size)

        self.bestModel = [np.inf, self.layers]

        #checkpoint :
        #print("lenLayers : ", len(self.layers))
        #print([self.layers[i].weights.shape for i in range (len(self.layers))])

    def prediction(self, X, best = False):#propagation avant

        #on récupère le meilleur modèle de l'entrainement
        if best :
            L = len(self.layers)
            A = X
            for i in range(L): #-1
                best_biais = np.array(self.bestModel[1][i].biais)
                best_weights = np.array(self.bestModel[1][i].weights)
                A = self.bestModel[1][i].acti_fct.acti(best_biais + np.dot(A, best_weights))
            return A

        #propagation avant sur le modèle actuel
        else :
            L = len(self.layers)
            A = X
            for i in range(L): #-1
                layer = self.layers[i]
                biais = layer.biais
                weights = layer.weights
                A = layer.acti_fct.acti(biais + np.dot(A, weights))
            return A

    def backpropagation(self, X, Y): #propagation arrière (après une propragation avant)

        L = len(self.layers)

        pre_actis, actis = [], []

        # Propagation avant avec mémoire des pré-activations et des activations
        A = X
        actis.append(A) #erreur (ok)
        for i in range(L):

            layer = self.layers[i]
            biais = layer.biais
            weights = layer.weights

            Z = biais + np.dot(A, weights)
            
            pre_actis.append(Z)
            A = layer.acti_fct.acti(Z)

            actis.append(A)

            """Debugage :
            if i == L-1 :
                print("Z : ", Z)
                print("A :", A)
                print(Z.shape)
                print(A.shape)
                print(np.sum(A, axis = 1))"""

            #print("avant : ", layer.acti_fct.ac)


        # Propagation arrière et stockage des erreurs
        delta_W, delta_B = [], []

        layer_back1 = self.layers[-1]

        delta = self.cost_fct.cost_derivative(actis[-1], Y)*layer_back1.acti_fct.acti_derivative(pre_actis[-1])

        dW = np.einsum('bi, bj->bij', actis[-2] , delta) #notation d'Einstein (chgt à faire sur les poids entre l'avant-dernière activation, et la dernière pré-ativation)
        dB = delta #chgt à faire  sur les biais de la dernière pré-activation

        delta_W.append(dW)
        delta_B.append(dB)

        #print("arriere_end : ", layer_back1.acti_fct.ac)


        for j in range(L-1): #[::-1]: # à l'envers

            i = L-2-j


            layer_i = self.layers[i]
            layer_i1 = self.layers[i+1]

            #calcul des errreurs à chaque couche

            delta = np.dot(delta, np.transpose(layer_i1.weights))*layer_i.acti_fct.acti_derivative(pre_actis[i])
            #delta = np.dot(delta, np.transpose(self.weights[i+1]))*sigmoid_derivative(pre_actis[i])

            dW = np.einsum('bi, bj->bij', actis[i], delta)
            dB = delta
            delta_W.append(dW)
            delta_B.append(dB)

        return delta_W[::-1], delta_B[::-1] #erreur sur les poids et sur les biais

    #créer des batchs (shuffle et bonne taille)
    def make_batch(self, X, Y, batch_index, batch_size):

        def shuffle_data(X, Y):
            n = X.shape[1]
            data = np.concatenate([X, Y], axis=1)
            np.random.shuffle(data)
            return np.split(data, [n], axis=1)

        if batch_size >= X.shape[0]:
            return X, Y, 0

        elif batch_index + batch_size >= X.shape[0]:
            X, Y = shuffle_data(X, Y)
            return X[0:batch_size], Y[0:batch_size], 0

        else:
            return X[batch_index:batch_index + batch_size], Y[batch_index:batch_index + batch_size], batch_index + batch_size

    def train(self, X_train, Y_train, batch_size, val_batch_size, epochs, lr_dict, momentum_dict, optimizer = 'sgd', cross_validation=0.8, verbosity=1, graph = 1):

        #params pour Adam (valeurs conseillées dans le papier)
        beta1 = 0.9
        beta2 = 0.999
        epsi0 = 10e-8
        epsi_w = [np.full_like(self.layers[i].weights, epsi0) for i in range (len(self.layers))]
        epsi_b = [np.full_like(self.layers[i].biais, epsi0) for i in range (len(self.layers))]

        #on vérifie qu'on a bien des valeurs pour commencer l'entrainement
        assert(0 in lr_dict)
        assert(0 in momentum_dict)

        L_val_cost = []

        n, p = X_train.shape[1], Y_train.shape[1]
        L = len(self.layers)

        #séparation entre train et validation

        data = np.concatenate([X_train, Y_train], axis=1)
        np.random.shuffle(data)
        data, val_data = np.split(data, [int(data.shape[0] * cross_validation)])

        x_train, y_train = np.split(data, [n], axis=1)
        x_val, y_val = np.split(val_data, [n], axis=1)

        batch_index, val_batch_index = 0, 0

        #initialisation de toutes les matrices dont on a besoin (erreurs et matrices pour Adam)
        last_delta_W = [np.zeros_like(self.layers[i].weights) for i in range (len(self.layers))]
        Mt_w = [np.zeros_like(self.layers[i].weights) for i in range (len(self.layers))]
        Vt_w = [np.zeros_like(self.layers[i].weights) for i in range (len(self.layers))]
        Mt_b = [np.zeros_like(self.layers[i].biais) for i in range (len(self.layers))]
        Vt_b = [np.zeros_like(self.layers[i].biais) for i in range (len(self.layers))]

        #affichage
        if graph == 1 :
            L_cost = []
            L_val_cost = []
            L_epoch = []
            plt.ion()
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            plt.xlim(0, epochs)
            line1, =ax.plot(L_epoch, L_cost)
            line2, = ax.plot(L_epoch, L_val_cost)
            plt.show()

        t = 0
        for epoch in range(epochs):
            t+=1

            if epoch in lr_dict :
                alpha = lr_dict[epoch]

            if epoch in momentum_dict :
                momentum = momentum_dict[epoch]

            # Preparation des batchs de manière efficace
            x_train_batch, y_train_batch, batch_index = self.make_batch(x_train, y_train, batch_index, batch_size)
            x_val_batch, y_val_batch, val_batch_index = self.make_batch(x_val, y_val, val_batch_index, val_batch_size)

            #récupération des erreurs
            delta_W, delta_B = self.backpropagation(x_train_batch, y_train_batch)

            for i in range (L) : #L-1

                layer = self.layers[i]

                #SGD

                if optimizer == 'sgd' :

                    DW = alpha*np.mean(delta_W[i], axis = 0)
                    layer.weights -= DW  + momentum*last_delta_W[i]
                    last_delta_W[i] = DW
                    layer.biais -= alpha*np.mean(delta_B[i], axis = 0)


                #RMSprop
                
                elif optimizer == 'rmsprop' :
                    DW = np.mean(delta_W[i], axis = 0)
                    DB = np.mean(delta_B[i], axis = 0)
                    Vt_w[i] = beta1*Vt_w[i] + (1-beta1)*DW*DW
                    Vt_b[i] = beta1*Vt_b[i] + (1-beta1)*DB*DB
                    layer.weights -= alpha*(1/(np.sqrt(Vt_w[i])+epsi_w[i]))*DW
                    layer.biais -= alpha*(1/(np.sqrt(Vt_b[i])+epsi_b[i]))*DB

                #Adam
                
                elif optimizer == 'adam' :
                    DW = np.mean(delta_W[i], axis = 0)
                    DB = np.mean(delta_B[i], axis = 0)
                    Mt_w[i] = beta1*Mt_w[i] + (1-beta1)*DW
                    Mt_b[i] = beta1*Mt_b[i] + (1-beta1)*DB
                    Vt_w[i] = beta2*Vt_w[i] + (1-beta2)*DW*DW
                    Vt_b[i] = beta2*Vt_b[i] + (1-beta2)*DB*DB

                    
                    Mt_w_corr_i = Mt_w[i]/(1-beta1**t)
                    Mt_b_corr_i = Mt_b[i]/(1-beta1**t)
                    Vt_w_corr_i = Vt_w[i]/(1-beta2**t)
                    Vt_b_corr_i = Vt_b[i]/(1-beta2**t)

                    """
                    Pour comparer sans les prise en compte du biais
                    Mt_w_corr_i = Mt_w[i]
                    Mt_b_corr_i = Mt_b[i]
                    Vt_w_corr_i = Vt_w[i]
                    Vt_b_corr_i = Vt_b[i]"""

                    layer.weights -= alpha*(Mt_w_corr_i/(np.sqrt(Vt_w_corr_i)+epsi_w[i]))
                    layer.biais -= alpha*(Mt_b_corr_i/(np.sqrt(Vt_b_corr_i)+epsi_b[i]))

                else :
                    raise NameError("L'optimizer n'existe pas")

            #tracer le schéma en temps réel (ne fonctionne pas pour l'instant)
            if graph == 1 and epoch % 20 == 0:
                L_cost.append(self.cost_fct.cost(self.prediction(x_train_batch), y_train_batch))
                L_val_cost.append(self.cost_fct.cost(self.prediction(x_val_batch), y_val_batch))
                L_epoch.append(epoch)

                line1.set_xdata(L_epoch)
                line2.set_xdata(L_epoch)
                line1.set_ydata(L_cost)
                line2.set_ydata(L_val_cost)
                fig.canvas.draw()
                fig.canvas.flush_events()

            #affichage l'avancée en console
            if verbosity == 1 and epoch % 50 == 0:

                cost = self.cost_fct.cost(self.prediction(x_train_batch), y_train_batch)
                validation_cost = self.cost_fct.cost(self.prediction(x_val_batch), y_val_batch)
                L_val_cost.append(self.cost_fct.cost(self.prediction(x_val_batch), y_val_batch))

                print('------------')
                print("Epochs {} - cost : {} - valid_cost : {} - lr : {}".format(epoch, cost, validation_cost, alpha))
                if validation_cost < self.bestModel[0] :
                    self.bestModel = [validation_cost, self.layers]

                pass

        return L_val_cost


x_train, y_train = x_train_mnist, y_train_mnist
x_test, y_test = x_test_mnist, y_test_mnist


#exemple d'appel :
"""
input = Dense(28*28, "sigmoid", _reg = 0.0001)
couche1 = Dense(200, "sigmoid", _reg = 0.0001)
couche2 = Dense(100, "sigmoid", _reg = 0.0001)
loss1 = Fct_cout("mse")
MP = MultiPerceptron(output_size = 10, _layers = [input, couche1, couche2], loss = loss1)
MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 2000, lr_dict = {0 : 0.005}, momentum_dict = {0 : 0.0001}, optimizer = 'adam', graph=0)
#Pour configurer lr et l'inertie : les mettre dans un dictionnaire où la clé est l'epoch où l'on applique le changement, et la valeur associée est la valeur à appliquer à cette epoch

print("Précision sur les datas test :" + str(np.round(100 * np.mean(np.equal(np.argmax(MP.prediction(x_test), axis=1), np.argmax(y_test, axis=1))), 2))+"%")
print("Meilleur modèle loss : {} - précision : {}%".format(MP.bestModel[0], str(np.round(100 * np.mean(np.equal(np.argmax(MP.prediction(x_test, best=True), axis=1), np.argmax(y_test, axis=1))), 2))))
"""



#Quelques résultats :
"""
96.58% :
    input = Couche(28*28, "sigmoid", _reg = 0.0001)
    couche1 = Couche(200, "sigmoid", _reg = 0.0001)
    couche2 = Couche(100, "sigmoid", _reg = 0.0001)
    loss1 = Fct_cout("mse")
    MP = MultiPerceptron(output_size = 10, _layers = [input, couche1, couche2], loss = loss1, output_acti = "sigmoid")
    MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 20000, lr_dict = {0 : 2.5, 1000 : 2, 10000 : 1.5, 15000 : 1}, momentum_dict = {0 : 0.0001}, graph=0)
    
96,42%
    input = Couche(28*28, "sigmoid", _reg = 0.0001)
    couche1 = Couche(400, "sigmoid", _reg = 0.0001)
    couche2 = Couche(160, "sigmoid", _reg = 0.0001)
    couche3 = Couche(80, "sigmoid", _reg = 0.0001)
    loss1 = Fct_cout("mse")
    MP = MultiPerceptron(output_size = 10, _layers = [input, couche1, couche2, couche3], loss = loss1, output_acti = "sigmoid")
    MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 20000, lr_dict = {0 : 2.5, 1000 : 2, 10000 : 1.5, 15000 : 1}, momentum_dict = {0 : 0.0001}, graph=0)

92.79%

    input = Dense(28*28, "sigmoid", _reg = 0.0001)
    couche1 = Dense(200, "sigmoid", _reg = 0.0001)
    couche1 = Dense(200, "sigmoid", _reg = 0.0001)
    couche2 = Dense(100, "softmax", _reg = 0.0001)
    loss1 = Fct_cout("log")
    MP = MultiPerceptron(output_size = 10, _layers = [input, couche1, couche2], loss = loss1)
    MP.train (x_train, y_train, batch_size = 32, val_batch_size = 32, epochs = 2000, lr_dict = {0 : 2.5, 100 : 2, 1000 : 1.5, 1500 : 1}, momentum_dict = {0 : 0.0001}, graph=0)


"""
