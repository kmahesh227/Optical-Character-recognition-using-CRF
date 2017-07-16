import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_l_bfgs_b
import time
from math import exp,log

def parse_data(file_path, columns_file_path):
    print "Parsing the dataset from",file_path,
    with open("letter.names.txt") as f:
        columns = [_.strip() for _ in f.readlines()]
    df = pd.read_csv("letter.csv",delimiter="\t",names=columns,index_col=False)
    def convert_label(c):
        return ord(c)-ord('a')
    df.letter=df.letter.apply(convert_label)
    #print data
    data=[]
    word_id=1
    while(True):
        cur_word=df.loc[df.word_id == word_id]
        if cur_word.shape[0]==0:
            break
        data.append((np.asarray(cur_word[cur_word.columns[6:]]),np.asarray(cur_word['letter'])))        
        word_id +=1
    print "--- Done"
    np.random.shuffle(data)
    return data

def forwardalg(obs,wmatr,transmatrx):
    M=len(obs)
    alphat=np.zeros((M,26), dtype="float64")
    for i in range(0, 26):
        alphat[0, i] = exp(np.dot(wmatr[i], obs[0]))
    for m in range(1,M):
        for j in range(0,26):                    
            prod1=exp(np.dot(wmatr[j], obs[m]))
            sum1=0
            for i in range(0,26):
                sum1=sum1+exp(transmatrx[i,j])*alphat[m-1,i]
            alphat[m,j]=prod1*sum1
    return alphat

def backwardalg(obs,wmatr,transmatrx):
    M=len(obs)
    betat=np.zeros((M,26), dtype="float64")
    for i in range(0, 26):
        betat[-1, i] = exp(np.dot(wmatr[i], obs[-1]))    
    for m in range(M-2,-1,-1):
        for j in range(0,26):
            prod1=exp(np.dot(wmatr[j],obs[m]))
            sum1=0
            for i in range(0,26):
               sum1=sum1+np.exp(transmatrx[j, i])*betat[m+1,i]
            betat[m,j]=prod1*sum1
    return betat

def gradient(wmatr, transmatrx, x, y):
    p = 128
    m = len(x)
    forward = forwardalg(x, wmatr, transmatrx)
    backward = backwardalg(x, wmatr, transmatrx)
    z = sum(forward[-1])
    delW = np.empty((26, p), dtype="float64")
    for i in range(0, 26):  
        psi = x[0] * backward[0, i] + x[-1] * forward[-1, i]
        for j in range(1, m-1):
            psi += x[j] * (forward[j, i] * backward[j, i] / exp(np.dot(wmatr[i], x[j])))
        delW[i] = sum(x[j] for j in range(0, m) if y[j] == i) - psi / z
    delT = np.zeros((26, 26), dtype="float64")

    for i in range(1, m):  
        for j in range(0, 26):
            for k in range(0, 26):
                delT[j, k] -= forward[i-1, j] * backward[i, k] * exp(transmatrx[j, k])
    delT /= z
    for i in range(1, m):
        delT[y[i-1], y[i]] += 1
    return delW, delT

def log_probability(wmatr, transmatrx, x, y):
    psi = sum(np.dot(wmatr[y[i]], x[i]) for i in range(0, len(x)))
    psi += sum(transmatrx[y[i], y[i+1]] for i in range(0, len(x)-1))
    z = sum(forwardalg(x, wmatr, transmatrx)[-1])
    return log(exp(psi) / z)

def likelihood(theta, data, c):
    wmatr = np.reshape(theta[:26 * p], (26, p))
    transmatrx = np.reshape(theta[26 * p:], (26, 26))
    score = -c * sum(log_probability(wmatr, transmatrx, x, y) for x, y in data) / len(data)
    score += sum((np.linalg.norm(x)) ** 2 for x in wmatr) / 2
    score += sum(sum(x ** 2 for x in row) for row in transmatrx) / 2
    # print(score)
    return score

def likelihood_prime(theta, data, c):
    p=128
    wmatr = np.reshape(theta[:26 * p], (26, p))
    transmatrx = np.reshape(theta[26 * p:], (26, 26))
    deltaW, deltaT = [], []
    for x, y in data:
        nw, nt = gradient(wmatr, transmatrx, x, y)
        deltaW.append(nw)
        deltaT.append(nt)
    delW = -c * sum(deltaW) / len(data) + wmatr
    delT = -c * sum(deltaT) / len(data) + transmatrx
    return np.concatenate((np.reshape(delW, 26 * p), np.reshape(delT, 26 ** 2)))

def trainCRF(data, c, maxiter, log):	
    w = np.ndarray.flatten(np.zeros((26, 128)))
    t = np.ndarray.flatten(np.zeros((26, 26)))
    theta = np.concatenate([w, t])
    t0 = time.time()
    theta, fmin, _ = fmin_l_bfgs_b(likelihood, theta, fprime=likelihood_prime,
		                           args=(data, c), disp=1, maxiter=maxiter)
    t1 = time.time()
    log.write("Training time: "+str(t1-t0)+"\n")
    p = 128	
    w= np.reshape(theta[:26*p], (26, p))
    t= np.reshape(theta[26*p:], (26, 26))

    return (w,t)

def maxassn(obs,wmatr,transmatrx):
    M=len(obs)
    deltat=np.zeros((M,26), dtype="float64")
    outlabls=np.zeros(M)
    for m in range(1,M):
        for j in range(0,26):
            prod1=exp(np.dot(wmatr[j],obs[m]))         
            for i in range(0,26):
                maxprod1=prod1*exp(transmatrx[i,j])*deltat[m-1,i]
                if deltat[m,j]<maxprod1:
                    deltat[m,j]=maxprod1
    outlabls[M-1]=np.argmax(deltat[M-1,:])
    for m in range(M-2,-1,-1):
        prod1=exp(np.dot(wmatr[outlabls[m+1]],obs[m+1]))
        maxprod1=0
        marg=0
        for i in range(0,26):
            prod2=prod1*exp(transmatrx[outlabls[m+1],i])*deltat[m,i]
            if maxprod1<prod2:
                maxprod1=prod2
                marg=i
        outlabls[m]=marg
    return outlabls

def testCRF(data, wmatr, transmatrx, log):
    predictions = []
    correct_letter, correct_word = 0, 0
    start_time = time.time()
    for x, y in data:
        infer = max_sum(x, wmatr, transmatrx).tolist()
        correct_letter += sum(1 if infer[i] == y[i] else 0 for i in range(0, len(y)))
        if infer == y.tolist():
            correct_word += 1
        predictions += infer
    character_accuracy = 100 * correct_letter / len(predictions)
    word_accuracy = 100 * correct_word / len(data)
    log.write("Time taken={}\n".format(time.time() - start_time))
    log.write("letter-wise accuracy={}%\n".format(character_accuracy))
    log.write("word-wise accuracy={}%\n".format(word_accuracy))
    return predictions

if __name__ == "__main__":
    p=128    
    data=parse_data("letters.csv","letters.names.txt")
    N=len(data)
    print N
    C=1000    # Regularization constant
    maxiter = 1000  # Maximum iterations for optimization algorithm
    """
    Running multiple experiments with different data set size

    samples_counts= [ 150, 300, 450, 750, 1000, 1500, 2300, 3000, 4500, 6600];
    for samples_count in samples_counts:
        print "***********",samples_count,"**************"        
        cur_data = data[:samples_count]
        split = int(0.66*samples_count)
        data_train,data_test =cur_data[:split],cur_data[split:]
        with open("1000-iterations_results\log_"+str(samples_count)+".txt", "w") as f:
            wmatr,transmatrx = trainCRF(data_train, C, maxiter, f)
            np.savetxt("1000-iterations_results\learnt_w_"+str(samples_count)+".txt", wmatr)
            np.savetxt("1000-iterations_results\learnt_t_"+str(samples_count)+".txt", transmatrx)
            predictions_train=testCRF(data_train,w,t, f)
            predictions_test=testCRF(data_test,w,t, f) 
    """
    samples_count = 6600
    cur_data = data[:samples_count]
    split = int(0.66*samples_count)
    data_train,data_test =cur_data[:split],cur_data[split:]
    with open("log_"+str(samples_count)+".txt", "w") as f:
        wmatr,transmatrx = trainCRF(data_train, C, maxiter, f)
        np.savetxt("learnt_w_"+str(samples_count)+".txt", wmatr)
        np.savetxt("learnt_t_"+str(samples_count)+".txt", transmatrx)
        f.write("********** Evaluating on Training data **********\n")
        predictions_train=testCRF(data_train,wmatr,transmatrx, f)
        f.write("********** Evaluating on Test data  **********\n")
        predictions_test=testCRF(data_test,wmatr,transmatrx, f) 
