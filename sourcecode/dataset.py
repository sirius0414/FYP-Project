import os
import scipy.io
from os.path import dirname, join
import numpy as np
from time import time
import matplotlib.pyplot as plt


# path of dataset
dir = 'F:\data\Data\Saved'

# class dataset
class dataset:
    def __init__(self):
        self.label = []
        self.labelnames = []
        self.grandelabel = []
        self.grandelabelnames = []
        self.spectogram = []
        self.spectogram_db = []

    def add_example(self, labelnames, spectogram):
        self.labelnames.append(labelnames)
        self.spectogram.append(spectogram)

    def create_label(self):
        for i in range(len(self.labelnames)):
            if self.labelnames[i] == 'I3-D':
                self.label.append(10)
                self.grandelabel.append(1)
                self.grandelabelnames.append('Drone')
            elif self.labelnames[i] == 'MT-D':
                self.label.append(11)
                self.grandelabel.append(1)
                self.grandelabelnames.append('Drone')
            elif self.labelnames[i] == 'OpportuneBirds':
                self.label.append(12)
                self.grandelabel.append(2)
                self.grandelabelnames.append('Bird')
            elif self.labelnames[i] == 'OpportuneBirds4096':
                self.label.append(13)
                self.grandelabel.append(2)
                self.grandelabelnames.append('Bird')
            elif self.labelnames[i] == 'OpportuneBird':
                self.label.append(14)
                self.grandelabel.append(2)
                self.grandelabelnames.append('Bird')
            elif self.labelnames[i] == 'Opportune birds':
                self.label.append(15)
                self.grandelabel.append(2)
                self.grandelabelnames.append('Bird')
            else:
                print('Unknown label exists in dataset')

    def img2mag(self):
        for i in range(len(self.spectogram)):
            self.spectogram[i] = np.abs(self.spectogram[i])

    def mag2db(self):
        for i in range(len(self.spectogram)):
            self.spectogram_db.append(self.spectogram[i])
            self.spectogram_db[i] = 20*np.log10(self.spectogram_db[i])

#plot spectogram function
# def plot_gallery(title, images, n_col=3, n_row=2):
#     plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#     plt.suptitle(title, size=16)
#     for i, comp in enumerate(images):
#         plt.subplot(n_row, n_col, i + 1)
#         vmax = max(comp.max(), -comp.min())
#         plt.imshow(comp, cmap=plt.cm.gray, interpolation="nearest", vmin=-vmax, vmax=vmax)
#         # stretch the y axis to fill the plot
#         plt.ylim(-5, 1201 + 5)
#         plt.xticks(())
#         plt.yticks(())
#     plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
# # plot spectogram example
# plot_gallery("Spectogram", myDataset.spectogram[:5])

def createDataset(dir):
    # instantiate dataset
    t0 = time()
    myDataset = dataset()
    for file in os.listdir(dir):
        if file.endswith(".mat"):
            mat_fname = join(dir, file)
            test_Data = scipy.io.loadmat(mat_fname)
            myDataset.add_example(test_Data['label'], test_Data['fftData_frac'])
        else:
            pass
    myDataset.create_label()
    myDataset.img2mag()
    myDataset.mag2db()
    print("Dataset instantiation in %0.3fs" % (time() - t0))
    return myDataset
