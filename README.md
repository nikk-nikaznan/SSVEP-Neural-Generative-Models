# SSVEP-Neural-Generative-Models

Code to accompany our International Joint Conference on Neural Networks (IJCNN) paper entitled -
[Simulating Brain Signals: Creating Synthetic EEG Data via Neural-Based Generative Models for Improved SSVEP Classification](https://arxiv.org/pdf/1901.07429.pdf).

The code is structured as follows:

- `Data_pred.py ` contains functions to pre-proposes EEG data; 
- `EEG_Gen.py ` A sample script showing how all models can be run to generate data;
- `EEG_DCGAN.py ` Our DCGAN based model for generating SSVEP-based EEG data;
- `EEG_WGAN.py ` Our Wasserstein GAN based model for generating SSVEP-based EEG data;
- `EEG_VAE.py ` Our Variational Autoencoder based model for generating SSVEP-based EEG data;

The `Sampledata` directory contains some sample SSVEP EEG data on which the models can be trained. 

## Dependencies and Requirements
The code has been designed to support python 3.6+ only. The project has the following dependencies and version requirements:

- torch=1.1.0+
- numpy=1.16++
- python=3.6.5+
- scipy=1.1.0+

## Cite

Please cite the associated papers for this work if you use this code:

```
@article{aznan2019simulating,
  title={Simulating Brain Signals: Creating Synthetic EEG Data via Neural-Based Generative Models for Improved SSVEP Classification},
  author={Aznan, Nik Khadijah Nik and Atapour-Abarghouei, Amir and Bonner, Stephen and Connolly, Jason and Moubayed, Noura Al and Breckon, Toby},
  journal={arXiv preprint arXiv:1901.07429},
  year={2019}
}
```
