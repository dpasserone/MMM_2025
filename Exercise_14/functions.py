import pandas as pd
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import math
import itertools
import lightgbm as lgb
import pickle

# takes as input strings with list of elements and relative concentrations. An example would be:
#X_new = generate_features(m_names='Ti,Cr,V,Nb,Mo',a_names='Al,Ga',x_names='C,N',c_m='0.2,0.2,0.2,0.2,0.2',c_a='0.7,0.3',c_x='0.95,0.05')
# generates as output the corresponding 117-dimensional vector

def generate_features(m_names, a_names, x_names, c_m, c_a, c_x):


    atom_data_df = pd.read_csv('atoms_data.csv')
    atom_data = np.asarray(atom_data_df)

    atom_data_name = atom_data[:,0]

    m_names_list = m_names.split(",")
    m_names_list = [j.strip() for j in m_names_list]

    a_names_list = a_names.split(",")
    a_names_list = [j.strip() for j in a_names_list]

    x_names_list = x_names.split(",")
    x_names_list = [j.strip() for j in x_names_list]

    m_conc_list = c_m.split(",")
    a_conc_list = c_a.split(",")
    x_conc_list = c_x.split(",")

    c = [0]*9

    for i in range(len(m_conc_list)):
        
        c[i]=float(m_conc_list[i])
    
    for i in range(len(a_conc_list)):

        c[i+5]=float(a_conc_list[i])

    for i in range(len(x_conc_list)):

        c[i+7]=float(x_conc_list[i])

    # create vectors for all the variables (excluding concentrations that are read from input)

    z = [0]*9       #atomic number
    n = [0]*9       #number of neutrons
    m = [0]*9       #atomic mass
    n_s = [0]*9     #highest s shell number
    n_e_s = [0]*9   #number of electrons in highest s shell
    n_p = [0]*4     #highest p shell number (escludo gli elementi M che non hanno mai l'orbitale p in valenza)
    n_e_p = [0]*4   #number of electrons in highest p shell (escludo gli elementi M che non hanno mai l'orbitale p in valenza)
    n_d = [0]*7     #highest d shell number (escludo gli elementi X che non hanno mai l'orbitale d in valenza)
    n_e_d = [0]*7   #number of electrons in highest d shell (escludo gli elementi X che non hanno mai l'orbitale d in valenza)
    n_f = [0]*7     #highest f shell number (escludo gli elementi X che non hanno mai l'orbitale f in valenza)
    n_e_f = [0]*7   #number of electrons in highest f shell (escludo gli elementi X che non hanno mai l'orbitale f in valenza)
    el = [0]*9      #electronegativity
    ei = [0]*9      #ionization energy
    r = [0]*9       #atomic radius


    # M elements

    for i in range(len(m_names_list)):

        name = m_names_list[i]
        
        # get the position of chemical element in csv file "atoms_data.csv" 

        pos_name = atom_data_name.tolist().index(name)

        # take values and normalize them to the max of the M elements family

        z[i] = float(atom_data[pos_name][1])/73      
        n[i] = float(atom_data[pos_name][2])/108    
        m[i] = float(atom_data[pos_name][3])/180.9479   
        n_s[i] = float(atom_data[pos_name][5])/6
        n_e_s[i] = float(atom_data[pos_name][6])/2  
        n_d[i] = float(atom_data[pos_name][9])/5
        n_e_d[i] = float(atom_data[pos_name][10])/6
        n_f[i] = float(atom_data[pos_name][11])/4
        n_e_f[i] = float(atom_data[pos_name][12])/14
        el[i] = float(atom_data[pos_name][14])/2.16
        r[i] = float(atom_data[pos_name][15])/217
        ei[i] = float(atom_data[pos_name][16])/7.902

    # same for A elements

    for i in range(len(a_names_list)):

        name = a_names_list[i]

        pos_name = atom_data_name.tolist().index(name)


        z[i+5] = float(atom_data[pos_name][1])/83  
        n[i+5] = float(atom_data[pos_name][2])/126    
        m[i+5] = float(atom_data[pos_name][3])/208.9804   
        n_s[i+5] = float(atom_data[pos_name][5])/6
        n_e_s[i+5] = float(atom_data[pos_name][6])/2  
        n_p[i] = float(atom_data[pos_name][7])/6
        n_e_p[i] = float(atom_data[pos_name][8])/4
        n_d[i+5] = float(atom_data[pos_name][9])/5
        n_e_d[i+5] = float(atom_data[pos_name][10])/10  
        n_f[i+5] = float(atom_data[pos_name][11])/4
        n_e_f[i+5] = float(atom_data[pos_name][12])/14
        el[i+5] = float(atom_data[pos_name][14])/2.58
        r[i+5] = float(atom_data[pos_name][15])/217
        ei[i+5] = float(atom_data[pos_name][16])/10.487

    # same for X elements

    for i in range(len(x_names_list)):

        name = x_names_list[i]

        pos_name = atom_data_name.tolist().index(name)


        z[i+7] = float(atom_data[pos_name][1])/7   
        n[i+7] = float(atom_data[pos_name][2])/7    
        m[i+7] = float(atom_data[pos_name][3])/14.007      
        n_s[i+7] = float(atom_data[pos_name][5])/2  
        n_e_s[i+7] = float(atom_data[pos_name][6])/2  
        n_p[i+2] = float(atom_data[pos_name][7])/2
        n_e_p[i+2] = float(atom_data[pos_name][8])/3
        el[i+7] = float(atom_data[pos_name][14])/3.04
        r[i+7] = float(atom_data[pos_name][15])/192
        ei[i+7] = float(atom_data[pos_name][16])/14.534

    # build the final vector as horizontal stacking of vectors

    features = c + z + n + m + el + ei + r + n_s + n_e_s + n_p + n_e_p + n_d + n_e_d + n_f + n_e_f

    return features


# data augmentation function, takes as input the input dataset X and outputs y, both as numpy array

def permutations(X,y):

    row_input = X.shape[0]

    cc_m = X[:,0:5] # concentration
    cc_a = X[:,5:7]
    cc_x = X[:,7:9]

    zz_m = X[:,9:14] # atomic number
    zz_a = X[:,14:16]
    zz_x = X[:,16:18]

    nn_m = X[:,18:23] # neutron number
    nn_a = X[:,23:25]
    nn_x = X[:,25:27]

    mm_m = X[:,27:32] # mass
    mm_a = X[:,32:34]
    mm_x = X[:,34:36]

    eel_m = X[:,36:41] # electronegativity (Pauling)
    eel_a = X[:,41:43]
    eel_x = X[:,43:45]

    eei_m = X[:,45:50] # ionization energy
    eei_a = X[:,50:52]
    eei_x = X[:,52:54]

    rr_m = X[:,54:59] # vdw radius
    rr_a = X[:,59:61]
    rr_x = X[:,61:63]

    nn_s_m = X[:,63:68] # highest s shell number
    nn_s_a = X[:,68:70]
    nn_s_x = X[:,70:72]

    nn_e_s_m = X[:,72:77] # num of electron in highest s shell 
    nn_e_s_a = X[:,77:79]
    nn_e_s_x = X[:,79:81]

    nn_p_a = X[:,81:83] # highest p shell number
    nn_p_x = X[:,83:85]

    nn_e_p_a = X[:,85:87] # num of electron in highest p shell 
    nn_e_p_x = X[:,87:89]

    nn_d_m = X[:,89:94] # highest d shell number
    nn_d_a = X[:,94:96]

    nn_e_d_m = X[:,96:101] # num of electron in highest d shell 
    nn_e_d_a = X[:,101:103]

    nn_f_m = X[:,103:108] # highest f shell number
    nn_f_a = X[:,108:110]

    nn_e_f_m = X[:,110:115] # num of electron in highest f shell 
    nn_e_f_a = X[:,115:117]


    final_X = []
    final_Y = []

    for i in range(row_input):

        
        c_m = cc_m[i].tolist()
        c_a = cc_a[i].tolist()
        c_x = cc_x[i].tolist()

        z_m = zz_m[i].tolist()
        z_a = zz_a[i].tolist()
        z_x = zz_x[i].tolist()

        n_m = nn_m[i].tolist()
        n_a = nn_a[i].tolist()
        n_x = nn_x[i].tolist()

        m_m = mm_m[i].tolist()
        m_a = mm_a[i].tolist()
        m_x = mm_x[i].tolist()

        el_m = eel_m[i].tolist()
        el_a = eel_a[i].tolist()
        el_x = eel_x[i].tolist()

        ei_m = eei_m[i].tolist()
        ei_a = eei_a[i].tolist()
        ei_x = eei_x[i].tolist()

        r_m = rr_m[i].tolist()
        r_a = rr_a[i].tolist()
        r_x = rr_x[i].tolist()

        n_s_m = nn_s_m[i].tolist()
        n_s_a = nn_s_a[i].tolist()
        n_s_x = nn_s_x[i].tolist()

        n_e_s_m = nn_e_s_m[i].tolist()
        n_e_s_a = nn_e_s_a[i].tolist()
        n_e_s_x = nn_e_s_x[i].tolist()

        n_p_a = nn_p_a[i].tolist()
        n_p_x = nn_p_x[i].tolist()

        n_e_p_a = nn_e_p_a[i].tolist()
        n_e_p_x = nn_e_p_x[i].tolist()

        n_d_m = nn_d_m[i].tolist()
        n_d_a = nn_d_a[i].tolist()

        n_e_d_m = nn_e_d_m[i].tolist()
        n_e_d_a = nn_e_d_a[i].tolist()

        n_f_m = nn_f_m[i].tolist()
        n_f_a = nn_f_a[i].tolist()

        n_e_f_m = nn_e_f_m[i].tolist()
        n_e_f_a = nn_e_f_a[i].tolist()

        n_0 = z_m.count(0)             
        non_zero = len(z_m)-n_0  # the important parameter is how many M elements are there, since we have to avoid equal permutations of zeros (i.e. missing M elements in the lsit of max 5 M elements)

        n_perm = int(math.factorial(len(z_m))/math.factorial(n_0))

        non_zero_values = [e for i,e in enumerate(z_m) if e!=0]
        non_zero_indexes = [i for i,e in enumerate(z_m) if e!=0]

        perm_z = multiset_permutations(z_m) # generate permutations without repetitions of equal permutations of zeros

        list_perm_z = []
        perm_c_m = []
        perm_n_m = []
        perm_m_m = []
        perm_el_m = []
        perm_ei_m = []
        perm_r_m = []
        perm_n_s_m = []
        perm_n_e_s_m = []
        perm_n_d_m = []
        perm_n_e_d_m = []
        perm_n_f_m = []
        perm_n_e_f_m = []

        for perm in perm_z:

            list_perm_z.append(perm)

            index_0 = [i for i,e in enumerate(perm) if e==0]

            index = [[]]*non_zero

            p_c = [[]]*len(z_m)
            p_n = [[]]*len(z_m)
            p_m = [[]]*len(z_m)
            p_el = [[]]*len(z_m)
            p_ei = [[]]*len(z_m)
            p_r = [[]]*len(z_m)
            p_n_s = [[]]*len(z_m)
            p_n_e_s = [[]]*len(z_m)
            p_n_d = [[]]*len(z_m)
            p_n_e_d = [[]]*len(z_m)
            p_n_f = [[]]*len(z_m)
            p_n_e_f = [[]]*len(z_m)

            # assign zeros in same positions to other vectors

            for idx in index_0:
                p_c[idx]=0
                p_n[idx]=0
                p_m[idx]=0
                p_el[idx]=0
                p_ei[idx]=0
                p_r[idx]=0
                p_n_s[idx]=0
                p_n_e_s[idx]=0
                p_n_d[idx]=0
                p_n_e_d[idx]=0
                p_n_f[idx]=0
                p_n_e_f[idx]=0

            for t in range(non_zero):

                # take the index of where the nonzero element is now in the current permutation "perm"
                index[t] = perm.index(non_zero_values[t])

                # now I can use this index to apply the same current permutatin to the other vectors

                p_c[index[t]]= c_m[non_zero_indexes[t]]
                p_n[index[t]]= n_m[non_zero_indexes[t]]
                p_m[index[t]]= m_m[non_zero_indexes[t]]
                p_el[index[t]]= el_m[non_zero_indexes[t]]
                p_ei[index[t]]= ei_m[non_zero_indexes[t]]
                p_r[index[t]]= r_m[non_zero_indexes[t]]
                p_n_s[index[t]]= n_s_m[non_zero_indexes[t]]
                p_n_e_s[index[t]]=n_e_s_m[non_zero_indexes[t]]
                p_n_d[index[t]]= n_d_m[non_zero_indexes[t]]
                p_n_e_d[index[t]]=n_e_d_m[non_zero_indexes[t]]
                p_n_f[index[t]]= n_f_m[non_zero_indexes[t]]
                p_n_e_f[index[t]]=n_e_f_m[non_zero_indexes[t]]

            perm_c_m.append(p_c)
            perm_n_m.append(p_n)
            perm_m_m.append(p_m)
            perm_el_m.append(p_el)
            perm_ei_m.append(p_ei)
            perm_r_m.append(p_r)
            perm_n_s_m.append(p_n_s)
            perm_n_e_s_m.append(p_n_e_s)
            perm_n_d_m.append(p_n_d)
            perm_n_e_d_m.append(p_n_e_d)
            perm_n_f_m.append(p_n_f)
            perm_n_e_f_m.append(p_n_e_f)
            
        # for vectors with two entries (A and X elements) I can take simply the permutations, since I never have more than one zero

        perm_z_a = list(itertools.permutations(z_a))
        perm_z_x = list(itertools.permutations(z_x))

        perm_c_a = list(itertools.permutations(c_a))
        perm_c_x = list(itertools.permutations(c_x))

        perm_n_a = list(itertools.permutations(n_a))
        perm_n_x = list(itertools.permutations(n_x))

        perm_m_a = list(itertools.permutations(m_a))
        perm_m_x = list(itertools.permutations(m_x))

        perm_el_a = list(itertools.permutations(el_a))
        perm_el_x = list(itertools.permutations(el_x))

        perm_ei_a = list(itertools.permutations(ei_a))
        perm_ei_x = list(itertools.permutations(ei_x))

        perm_r_a = list(itertools.permutations(r_a))
        perm_r_x = list(itertools.permutations(r_x))

        perm_n_s_a = list(itertools.permutations(n_s_a))
        perm_n_s_x = list(itertools.permutations(n_s_x))

        perm_n_e_s_a = list(itertools.permutations(n_e_s_a))
        perm_n_e_s_x = list(itertools.permutations(n_e_s_x))

        perm_n_p_a = list(itertools.permutations(n_p_a))
        perm_n_p_x = list(itertools.permutations(n_p_x))

        perm_n_e_p_a = list(itertools.permutations(n_e_p_a))
        perm_n_e_p_x = list(itertools.permutations(n_e_p_x))

        perm_n_d_a = list(itertools.permutations(n_d_a))

        perm_n_e_d_a = list(itertools.permutations(n_e_d_a))

        perm_n_f_a = list(itertools.permutations(n_f_a))

        perm_n_e_f_a = list(itertools.permutations(n_e_f_a))

        output_list = []

        for l in range(n_perm):
            for j in range(len(perm_c_a)):
                for k in range(len(perm_c_x)):
                    
                    output_list = [0]*117

                    output_list[0:5]=perm_c_m[l][0:5]
                    output_list[5:7]=perm_c_a[j][0:2]
                    output_list[7:9]=perm_c_x[k][0:2]

                    output_list[9:14]=list_perm_z[l][0:5]
                    output_list[14:16]=perm_z_a[j][0:2]
                    output_list[16:18]=perm_z_x[k][0:2]

                    output_list[18:23]=perm_n_m[l][0:5]
                    output_list[23:25]=perm_n_a[j][0:2]
                    output_list[25:27]=perm_n_x[k][0:2]

                    output_list[27:32]=perm_m_m[l][0:5]
                    output_list[32:34]=perm_m_a[j][0:2]
                    output_list[34:36]=perm_m_x[k][0:2]

                    output_list[36:41]=perm_el_m[l][0:5]
                    output_list[41:43]=perm_el_a[j][0:2]
                    output_list[43:45]=perm_el_x[k][0:2]

                    output_list[45:50]=perm_ei_m[l][0:5]
                    output_list[50:52]=perm_ei_a[j][0:2]
                    output_list[52:54]=perm_ei_x[k][0:2]

                    output_list[54:59]=perm_r_m[l][0:5]
                    output_list[59:61]=perm_r_a[j][0:2]
                    output_list[61:63]=perm_r_x[k][0:2]

                    output_list[63:68]=perm_n_s_m[l][0:5]
                    output_list[68:70]=perm_n_s_a[j][0:2]
                    output_list[70:72]=perm_n_s_x[k][0:2]

                    output_list[72:77]=perm_n_e_s_m[l][0:5]
                    output_list[77:79]=perm_n_e_s_a[j][0:2]
                    output_list[79:81]=perm_n_e_s_x[k][0:2]

                    output_list[81:83]=perm_n_p_a[j][0:2]
                    output_list[83:85]=perm_n_p_x[k][0:2]

                    output_list[85:87]=perm_n_e_p_a[j][0:2]
                    output_list[87:89]=perm_n_e_p_x[k][0:2]

                    output_list[89:94]=perm_n_d_m[l][0:5]
                    output_list[94:96]=perm_n_d_a[j][0:2]

                    output_list[96:101]=perm_n_e_d_m[l][0:5]
                    output_list[101:103]=perm_n_e_d_a[j][0:2]

                    output_list[103:108]=perm_n_f_m[l][0:5]
                    output_list[108:110]=perm_n_f_a[j][0:2]

                    output_list[110:115]=perm_n_e_f_m[l][0:5]
                    output_list[115:117]=perm_n_e_f_a[j][0:2]
        
                    final_X.append(output_list)
                    final_Y.append(y[i].tolist())

    return np.asarray(final_X), np.asarray(final_Y)

# read data as in "input_data_211.csv", in which each column contains a string for chemical elements and concentrations, as well as a and c lattice parameter
# returns X input dataset as numpy array and Y outputs as numpy array, both generated by "generate_feature" function

def read_data(csv_data):


    exp_data_df = pd.read_csv(csv_data)
    exp_data = np.asarray(exp_data_df)

    X = []
    Y = []

    for data in exp_data:

        m_names = data[0]
        a_names = data[1]
        x_names = data[2]

        c_m = data[3]
        c_a = data[4]
        c_x = data[5]

        features = generate_features(m_names, a_names, x_names, c_m, c_a, c_x)

        X.append(features)
        Y.append([data[6],data[7]])

    X = np.asarray(X)
    Y = np.asarray(Y)

    return X,Y

# function to reduce features. The list of indices was calculated by looking at variables correlated in absolute value strictly more than 99%

def reduce_features(X):
    

    features_to_remove_idx = [6,8,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,43,44,50,51,52,53,54,55,56,57,58,61,62,70,71,79,80,83,84,87,88,89,90,91,92,93,110,111,112,113,114,115,116]

    X = pd.DataFrame(X)
    X = X.drop(features_to_remove_idx, axis=1)
    X = X.to_numpy()

    return X

# min-max output rescaling

def rescale_output(y, min, max):

    y = (y-min)/(max-min)

    return y


# function to make the weighted average prediction as described in the paper. X is the vector of the structure under investigation, generated by "generate_feature"
# model is either 'XGB' or 'LGBM' ans savefolder the name of the folder used to store model, as named in the train_optuna*.py main script.
# N is the number of models (i.e. N-30 in the paper, but can be modified in the train script)
# return a, err_a, c and err_c as described in the paper

def final_predict(X, model, save_folder, N):   

    X= np.asarray(X).reshape(1,-1)

    Y= np.asarray([3.0, 13.0]).reshape(1,-1) # I invent an output vector just to call the "permutations" function

    X_perm, y_perm = permutations(X,Y)

    X_perm = reduce_features(X_perm)

    a = 0
    err_a = 0
    c = 0
    err_c = 0

    pred_p_a = np.empty(0)
    pred_p_c = np.empty(0)
    pred_i_a = np.empty(0)
    pred_i_c = np.empty(0)
    pred_i_err_a = np.empty(0)
    pred_i_err_c = np.empty(0)

    if model=='LGBM':

        min_max = np.loadtxt(save_folder+'min_max.txt', skiprows=1)

        best_models_a_folder = save_folder+'best_models_a/'  
        best_models_c_folder = save_folder+'best_models_c/'

        for i in range(N):

            model_a = lgb.Booster(model_file=best_models_a_folder+"model_a_"+str(i)+".txt")
            model_c = lgb.Booster(model_file=best_models_c_folder+"model_c_"+str(i)+".txt")

            min_a = min_max[i][0]
            max_a = min_max[i][1]
            min_c = min_max[i][2]
            max_c = min_max[i][3]

            for x in X_perm:

                a_new_pred = model_a.predict(x.reshape(1,-1)) # prediction of a e c for every permutation x of new input vector
                c_new_pred = model_c.predict(x.reshape(1,-1))
                
                a_pred = (max_a-min_a)*a_new_pred+min_a   # inverse rescaling to original range of a and c
                c_pred = (max_c-min_c)*c_new_pred+min_c

                pred_p_a = np.append(pred_p_a, a_pred)
                pred_p_c = np.append(pred_p_c, c_pred)

            # save the average prediction over permutations for the i-th model
            pred_i_a = np.append(pred_i_a, pred_p_a.mean())
            pred_i_c = np.append(pred_i_c, pred_p_c.mean())

            # and the same for the error
            pred_i_err_a = np.append(pred_i_err_a, np.std(pred_p_a))
            pred_i_err_c = np.append(pred_i_err_c, np.std(pred_p_c))

        
        # calculate the weighted average for a and c

        weights_a = 1/pred_i_err_a**2
        weights_c = 1/pred_i_err_c**2

        a_w_av =  np.dot(weights_a, pred_i_a)/np.sum(weights_a)
        sigma_a_w_av = 1/math.sqrt(np.sum(weights_a))

        c_w_av =  np.dot(weights_c, pred_i_c)/np.sum(weights_c)
        sigma_c_w_av = 1/math.sqrt(np.sum(weights_c))

        a = a_w_av
        err_a = sigma_a_w_av
        c = c_w_av
        err_c = sigma_c_w_av

    # same for XGB, just different loading of models

    if model=='XGB':

        min_max = np.loadtxt(save_folder+'min_max.txt', skiprows=1)

        loaded_models_a = pickle.load(open(save_folder+'best_models_a.sav','rb'))
        loaded_models_c = pickle.load(open(save_folder+'best_models_c.sav','rb'))

        for i in range(N):

            model_a = loaded_models_a[i]
            model_c = loaded_models_c[i]

            min_a = min_max[i][0]
            max_a = min_max[i][1]
            min_c = min_max[i][2]
            max_c = min_max[i][3]

            for x in X_perm:

                a_new_pred = model_a.predict(x.reshape(1,-1)) 
                c_new_pred = model_c.predict(x.reshape(1,-1))
                
                a_pred = (max_a-min_a)*a_new_pred+min_a   
                c_pred = (max_c-min_c)*c_new_pred+min_c


                pred_p_a = np.append(pred_p_a, a_pred)
                pred_p_c = np.append(pred_p_c, c_pred)

            pred_i_a = np.append(pred_i_a, pred_p_a.mean())
            pred_i_c = np.append(pred_i_c, pred_p_c.mean())

            pred_i_err_a = np.append(pred_i_err_a, np.std(pred_p_a))
            pred_i_err_c = np.append(pred_i_err_c, np.std(pred_p_c))


        weights_a = 1/pred_i_err_a**2
        weights_c = 1/pred_i_err_c**2

        a_w_av =  np.dot(weights_a, pred_i_a)/np.sum(weights_a)
        sigma_a_w_av = 1/math.sqrt(np.sum(weights_a))

        c_w_av =  np.dot(weights_c, pred_i_c)/np.sum(weights_c)
        sigma_c_w_av = 1/math.sqrt(np.sum(weights_c))

        a = a_w_av
        err_a = sigma_a_w_av
        c = c_w_av
        err_c = sigma_c_w_av


    return a, err_a, c, err_c