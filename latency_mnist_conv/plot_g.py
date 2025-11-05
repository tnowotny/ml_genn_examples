import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def plot_kernels(g, ht, wd, num, caption):
    fig, ax = plt.subplots(ht,wd,sharex=True,sharey=True)
    for i in range(ht):
        for j in range(wd):
            id = i*wd+j
            if id < num:
                ax[i,j].imshow(g[id,:,:])
            else:
                ax[i,j].set_visible(False)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    fig.suptitle(caption)
    plt.tight_layout()

if len(sys.argv) != 3:
    print(f"usage: {argv[0]} <settings filename.json> <kernel filename.npy>")
    exit(1)

with open(sys.argv[1], "r") as f:
    p = json.load(f)

g = np.load(sys.argv[2])
g = g.reshape((p["NUM_EPOCHS"]+1,p["NUM_KERNELS"],p["KERNEL_SZ"],p["KERNEL_SZ"]))
wd = int(np.sqrt(p["NUM_KERNELS"]))
ht = (p["NUM_KERNELS"]-1) // wd + 1
for k in [0, p["NUM_EPOCHS"]]:
    plot_kernels(g[k,:,:,:], ht, wd, p["NUM_KERNELS"],f"kernel[Epoch == {k}]")
plot_kernels(g[p["NUM_EPOCHS"],:,:,:]-g[0,:,:,:], ht, wd, p["NUM_KERNELS"],f"kernel[Epoch == {p['NUM_EPOCHS']} - kernel[Epoch == 0]")

plt.show()
