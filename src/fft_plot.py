import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

T = 3.0
# frequency class
nf = 0

def fft_data(input_data):
    ffts = []
    for block in range(input_data.shape[0]):
        for channel in range(input_data.shape[-1]):
            input_data[block, :, channel] -= input_data[block,:, channel].mean()
            # print data.shape
            ffts.append(np.abs(np.fft.rfft(input_data[block,:,channel])))
    mean_fft = np.stack(ffts).mean(axis=0)
    return mean_fft

def plot_fft(x_axis, data_fft):
    
    plt.plot(x_axis, data_fft, 'tab:red')

    plt.xlabel('Frequency', fontsize=17)
    plt.ylabel('Amplitude', fontsize=17)
    plt.axis(xmin = 5, xmax = 70)
    plt.axis(ymin = 0, ymax = 20)
    plt.tight_layout()
    filename = "fft_plot_class%i.pdf" 
    plt.savefig(filename % (nf), format='PDF', bbox_inches='tight')

if __name__ == "__main__":

    # data loading
    data_files = [ f'{Path.home()}/Data/SampleData_class{nf}.npy']
    data = [np.load(f) for f in data_files]

    data = np.asarray(data)
    data = np.concatenate(data)

    data_fft = fft_data(data)
    x_axis = np.linspace(0,(data.shape[1]/T)/2, data_fft.shape[0])

    plot_fft(x_axis, data_fft)

