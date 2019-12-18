from plot_pca_new import PlotEmb
import glob
import numpy as np

emb_path = '../eer_data/emb_epoch_000085.npy'
lab_path = '../eer_data/emb_epoch_000085_lab.npy'

dvectors=np.load(emb_path,allow_pickle=True)
labs= np.load(lab_path,allow_pickle=True)


num_epoch=1

plotemb=PlotEmb(dvectors,labs,num_epoch)
plotemb.plot_tsne()