import math as m
import numpy as np
import pylab
from pylab import arange,pi,sin,cos,sqrt
import matplotlib.pyplot as plt

"""Quelques outils pour accélérer les tests"""

#calcul des intervales de confiance
def interval (N, esti_mean, esti_deviation, confidence):
    """N : nb échantillons
        esti_mean : estimation de la moyenne
        esti_deviation : estimation écart-type
        confidence : niveau de confiance (%)"""

    t_alpha_dict = {90 : 1.645, 95 : 1.960, 99 : 2.576} #lien entre confiance et t_aplha
    
    assert confidence in t_alpha_dict, "L'intervalle de confiance n'est pas dans la table"
    
    t_alpha = t_alpha_dict[confidence]

    err_inf = t_alpha*(esti_deviation/m.sqrt(N)) #esti_mean-t_alpha*(esti_deviation/m.sqrt(N))
    err_sup = t_alpha*(esti_deviation/m.sqrt(N))#esti_mean+t_alpha*(esti_deviation/m.sqrt(N))

    return (err_inf, err_sup)


#séparation des groupes, et calcul des résultats de stats
def stats (L_samples, L_meta_data, tested_param, confidence = 95):
    """L_samples : résultat des écantillons
        L_meta_data : dictionnaire des méta_données pré-défini, contient 'nom_group', 'learning_rate', etc
        tested_param : paramètre sur lequel séparer les données (str)
        """

    #tri de données selon les groupes d'échantillons
    #D_groups est un dico de clé : le nom du groupe | de valeur : la liste des données
    group = tested_param
    print("START stats, separation : ", group)

    assert len(L_samples) == len(L_meta_data), "Les méta-data doivent correspondre aux datas"

    D_groups_samples = {}
    D_groups_metadata = {}
    for i in range (len(L_samples)):
        #print(L_meta_data[i][group])
        #print(D_groups_samples)
        if L_meta_data[i][group] in D_groups_samples :
            D_groups_samples[L_meta_data[i][group]].append(L_samples[i]) #on ajoute les samples au groupe déjà existant
            #print(L_meta_data[i])
            #print(D_groups_metadata)
            assert D_groups_metadata[L_meta_data[i][group]] == L_meta_data[i] #on s'assure que les metadata concernant le groupe correspondent bien
        else : #on créer un nuveau groupe et on rentre les samples et les metadonnées
            D_groups_samples[L_meta_data[i][group]] = [L_samples[i]] 
            D_groups_metadata[L_meta_data[i][group]] = L_meta_data[i]
    
    #statitiques basiques sur les groupes
    D_group_stat = {} #dico : clé-groupe, valeur-dico des metadonnées
    for group_name, samples in D_groups_samples.items():
        mean = np.mean(samples)
        deviation = np.std(samples)
        #assert len(samples) > 50, "Ca fait pas au moins 50 points par groupes" àà remettre !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        (err_inf, err_sup) = interval (len(samples), mean, deviation, confidence)
        D_group_stat[group_name] = [mean, deviation, err_inf, err_sup]

    print("END stats, separation : ", group)

    return D_group_stat, D_groups_metadata


#enregsitrement du graph
def plot_stats (D_group_stat, D_groups_metadata, xlabel, log=False, label = "precision") : #AFFICHER L'EVOLUTION DE ACC EN FONCTION DES EPOCHS !!!! (à faire)
    
    print("START PLOT")

    Lx = []
    Lmean = []
    L_err_inf = []
    L_err_sup = []
    LmetaData = []


    for group_name in D_group_stat.keys():
        Lx.append(group_name) #paramètre d'interet en abscisse
        Lmean.append(D_group_stat[group_name][0]) #c'est le mean
        L_err_inf.append(D_group_stat[group_name][2]) #c'est le inf
        L_err_sup.append(D_group_stat[group_name][3]) #c'est le sup
        LmetaData.append(D_groups_metadata[group_name]) #les metadata (liste de dicos)
        #les metadatas peuvent etre intéressante à afficher mais pour l'instant chaque groupe contient des metadata et c'est trop, il faut les regrouper,mais enlever celle qui ne sont pas constantes

    yerr=np.array(L_err_inf+L_err_sup).reshape(2, -1)
    #print(yerr)

    plt.figure(1)
    plt.clf()

    plt.errorbar(Lx, Lmean,
       yerr=yerr,
       marker='o',
       color='b',
       ecolor='k',
       markerfacecolor='b',
       label=label,
       capsize=5,
       linestyle='None')
       

    #plt.axes([0.125,0.2,0.95-0.125,0.95-0.2])
    plt.title("Final accuracy", loc="center")
    plt.xlabel(xlabel)
    plt.legend()
    if log == True :
        plt.xscale('log')

    plt.savefig('end_acc_'+str(list(D_groups_metadata.values())[0])+'.eps')

    print("END PLOT")


#affichage du cout à chaque epoch
def plot_val_cost (L_val_cost_k_i, D_groups_metadata, xlabel, label = "cost"):

    plt.figure(2)
    plt.clf()


    X = [i*50 for i in range (len(L_val_cost_k_i[0][0][0]))]

    for i in range (len(L_val_cost_k_i)):

        color=np.random.rand(3,)
        
        L_val_cost_k = L_val_cost_k_i[i][0]
        label2 = L_val_cost_k_i[i][1]

        means = np.mean(L_val_cost_k, axis=0)
        deriv = np.std(L_val_cost_k, axis=0)

        N = len(L_val_cost_k)
        confidence = 95

        #calcul de l'intervalle de confiance
        Lerr_inf = [interval(N, means[i], deriv[i], confidence)[0] for i in range (len(means))]
        Lerr_sup = [interval(N, means[i], deriv[i], confidence)[1] for i in range (len(means))]
        yerr=np.array(Lerr_inf+Lerr_sup).reshape(2, -1)

        plt.plot(X, means, label = label2, color = color)

        plt.errorbar(X, means,
       yerr=yerr,
       marker='o',
       color=color,
       markerfacecolor='black',
       markersize='1',
       markerevrey = None,
       capsize=1,
       linestyle='None',
       linewidth = '1',
       elinewidth = '1')

    plt.title("Training Comparison", loc="center")
    plt.xlabel(xlabel)
    plt.ylabel(label)
    plt.legend()
    plt.savefig('cost_'+str(list(D_groups_metadata.values())[0])+'.eps')


def generation_test_datas (): # juste pour vérifier si l'affichage des intervalles se fait correctement

    """Doit générer :

        L_samples, L_meta_data, tested_param pour une fonction sinus (meta : periode, amplitude, offset, avec des erreurs ! (tested params : erreurs))

    """

    X = pylab.arange(-2*pi,2*pi,0.5)

    #params
    ampl = 1
    offest = 0
    Y1 =  np.random.normal(loc=offest, scale=ampl, size=60)
    ampl = 2
    offest = 0
    Y2 = np.random.normal(loc=offest, scale=ampl, size=60)#+np.random.normal(loc= 1, scale = 0.4) #essayer avec des size différents
    ampl = 3
    offest = 0
    Y3 = np.random.normal(loc=offest, scale=ampl, size=60)#+np.random.normal(loc= 1, scale = 0.4) #essayer avec des size différents

    L_samples = []
    L_meta_data = []
    for i in range (len(Y1)):
        L_samples.append(Y1[i])
        L_meta_data.append({'ampl' : 1, 'offset' : 0, 'autre' : None})
    for i in range (len(Y2)):
        L_samples.append(Y2[i])
        L_meta_data.append({'ampl' : 2, 'offset' : 0, 'autre' : None})
    for i in range (len(Y3)):
        L_samples.append(Y3[i])
        L_meta_data.append({'ampl' : 3, 'offset' : 0, 'autre' : None})

    tested_param = 'ampl'

    return L_samples, L_meta_data, tested_param


