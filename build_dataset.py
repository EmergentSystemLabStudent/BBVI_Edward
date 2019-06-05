import numpy as np

def build_dataset_1dim_ugm(N,mean,std):
    np.random.seed(42)
    data = np.random.normal(mean, std, N)
    return data

def build_dataset_2dim_ugm(N,mean,std):
    np.random.seed(42)
    data = np.random.multivariate_normal(mean,std,N)
    return data

def build_dataset_1dim_gmm(N,mean,std,pi):
    np.random.seed(42) 
    N1=int(N*pi[0])
    N2=int(N*pi[1])
    x1 = np.random.normal(mean[0], std[0], N1)
    x2 = np.random.normal(mean[1], std[1], N2)
    data = np.concatenate([x1, x2])
    label =np.concatenate([[0.0]*N1,[1.0]*N2])
    return data,label

def build_dataset_1dim_Kclass_gmm(N,K,mean,std,pi):
    np.random.seed(42) 
    label = [0] * N;
    data = np.zeros(N)
    mix = [0] * K;
    for n in range(N):
        label[n] = np.argmax(np.random.multinomial(1,pi))
        data[n] = np.random.normal(mean[label[n]],std[label[n]])
        mix[label[n]]=mix[label[n]]+1
    return data,label

def build_dataset_2dim_Kclass_gmm(N,K,mean,cov,pi):
    np.random.seed(42)
    label = [0] * N;
    data = np.zeros((N,2))
    mix = [0] * K;
    for n in range(N):
        label[n] = np.argmax(np.random.multinomial(1,pi))
        data[n,:] = np.random.multivariate_normal(mean[label[n]],cov[label[n]])
        mix[label[n]] = mix[label[n]] + 1
    return data,label,mix