import os
import numpy
import sys

sys.path.append('./cnn')
from cnn.step02_testing import predict
file_path = './media/01_W_104_1b1_Ar_sc_Litt3200_W_1.npz'

predict(file_path)
