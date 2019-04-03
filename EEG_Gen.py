import Data_prep
from EEG_DCGAN import dcgan
from EEG_VAE import vae
from EEG_WGAN import wgan
import torch
import numpy as np
import time
import random
import argparse

gen_model = "DCGAN" #"WGAN" , "VAE"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed_n = np.random.randint(500)
print (seed_n)

random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)

gen_time = 0
for nclass in range (0, 3):
    
    if nclass == 0:
        data_train = Data_prep.data_class0
        label_train = Data_prep.label_class0
    if nclass == 1:
        data_train = Data_prep.data_class1
        label_train = Data_prep.label_class1
    if nclass == 2:
        data_train = Data_prep.data_class2
        label_train = Data_prep.label_class2
    
    data_train = data_train.swapaxes(1, 2)
    
    # generative model
    print ("*********Training Generative Model*********")
    start = time.time()
    if gen_model == "DCGAN":
        print(gen_model)
        gen_data = dcgan(data_train, label_train, seed_n) # DCGAN
    elif gen_model == "WGAN":
        print(gen_model)
        gen_data = wgan(data_train, label_train, seed_n) # WGAN
    elif gen_model == "VAE":  
        print(gen_model)
        gen_data = vae(data_train, label_train, seed_n) # VAE
    
    gen_time = gen_time + (time.time() - start)

    # save generated data
    if nclass == 0:
        fake_data0 = gen_data
        # np.save("~/fake_class0.npy", fake_data0)
    if nclass == 1:
        fake_data1 = gen_data
        # np.save("~/fake_class1.npy", fake_data1)
    if nclass == 2:
        fake_data2 = gen_data
        # np.save("~/fake_class2.npy", fake_data2)

print("time for generative model: %f" % gen_time)



