import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from tqdm import tqdm

# true = [float(line.rstrip('\n')) for line in open('data/train.txt')]
# true = np.array(true).flatten()[2:]

preds = np.load('preds.npy').flatten()

kf = KalmanFilter(initial_state_mean=preds[0], n_dim_obs=1)
smooth_preds, _ = kf.smooth(preds)
new = smooth_preds = smooth_preds.flatten()
for x in tqdm(range(200)):
    smooth_preds, _ = kf.smooth(new)
    new = smooth_preds = smooth_preds.flatten()

np.savetxt('test.txt',smooth_preds)

# plt.plot(preds,'r')
# plt.plot(smooth_preds)
# plt.show()































#
