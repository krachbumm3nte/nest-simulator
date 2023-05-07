

from src.dataset import MnistDataset
import matplotlib.pyplot as plt
import numpy as np

#orig_set = MnistDataset('train', 2, 5)
n_cls = 10 
s_sml = 10
small_set = MnistDataset('train', n_cls, 120, target_size=s_sml)
s_med = 12
med_set = MnistDataset('train', n_cls, 120, target_size=s_med)

set = MnistDataset('train', n_cls, 120)


img, label = small_set[0]

n_samples = 7
fig, axes = plt.subplots(3, n_samples)

for i in range(n_samples):
    idx = np.random.randint(0, 120)
    axes[0][i].imshow(small_set.__getitem__(idx)[0].reshape(s_sml, -1), cmap="Greys")
    axes[1][i].imshow(med_set.__getitem__(idx)[0].reshape(s_med, -1), cmap="Greys")
    axes[2][i].imshow(set.__getitem__(idx)[0].reshape(28, -1), cmap="Greys")

plt.show()

