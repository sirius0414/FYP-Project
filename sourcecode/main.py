from time import time
from without_PCA import without_PCA
from conv_PCA import conv_PCA_trail
from PCA2d import PCA_2D_trail
from L1PCA import L1_PCA_trail

dir1 = 'F:\data\Data\Real-300bin'
dir2 = 'F:\data\Data\Real-600bin'
dir3 = 'F:\data\Data\Real-1200bin'

t0 = time()

acc_without300 = without_PCA('Real-300bin',dir1)
acc_without600 = without_PCA('Real-600bin',dir2)
acc_without1200 = without_PCA('Real-1200bin',dir3)

acc_conv300 = conv_PCA_trail('Real-300bin',dir1)
acc_conv600 = conv_PCA_trail('Real-600bin',dir2)
acc_conv1200 = conv_PCA_trail('Real-1200bin',dir3)

acc_PCA2d300 = PCA_2D_trail('Real-300bin',dir1)
acc_PCA2d600 = PCA_2D_trail('Real-600bin',dir2)
acc_PCA2d1200 = PCA_2D_trail('Real-1200bin',dir3)

acc_L1PCA300 = L1_PCA_trail('Real-300bin',dir1)
acc_L1PCA600 = L1_PCA_trail('Real-600bin',dir2)
acc_L1PCA1200 = L1_PCA_trail('Real-1200bin',dir3)

print("Total time: %0.3fs" % (time() - t0))
