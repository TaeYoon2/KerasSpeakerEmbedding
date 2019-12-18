import numpy as np
import re
import os
import argparse
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import time
import pdb
import random
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from multiprocessing import Pool


from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq




class Plot2D(object):
   def __init__(self,emb_indir,tsne_outdir,plot_outdir):

      if not os.path.exists(plot_outdir):
         os.makedirs(plot_outdir)

      if not os.path.exists(tsne_outdir):
         os.makedirs(tsne_outdir)

      self.emb_indir=emb_indir
      self.tsne_outdir=tsne_outdir
      self.plot_outdir=plot_outdir

      return

   def load_embs(self):
      emb_npy_list = glob.glob(self.emb_indir+'/*.npy')
      utt_label_list = [ i.split('.npy')[0] for i in emb_npy_list ]

      # load all emb
      emb_list=[]
      valid_utt_label_list=[]
      for emb_npy,lab in zip(emb_npy_list,utt_label_list):
         try:
            emb_list.append(np.load(emb_npy,allow_pickle=True))
            valid_utt_label_list.append(lab)
         except:
            print(f"[skipping] could not open {emb_npy}")

      print(f"read {len(emb_list)} embedding npys,{len(valid_utt_label_list)} labels")
      np.save(os.path.join(tsne_outdir,'utt_label_list.npy'),valid_utt_label_list)

      return emb_list,valid_utt_label_list
      

   def run_tsne(self,emb_list):

      #TSNE
      emb_list=np.asarray(emb_list).reshape(-1,256)
      print(emb_list.shape)

      print("tsne running ...")
      start_tsne=time.time()
      tsne = TSNE(n_components=2, random_state=0)
      emb_2d_tsne = tsne.fit_transform(np.asarray(emb_list))
      print(f"tsne done: {time.time()-start_tsne} secs")
      np.save(os.path.join(tsne_outdir,'emb_tsne_2d.npy'),emb_2d_tsne)
      print("tsne dim reduction saved as {os.path.join(tsne_outdir,'emb_tsne_2d.npy')}")

      return emb_2d_tsne

   def run_pca(self,emb_list,utt_label_list):
      #PCA

      return

   def plot_2d(self,array_2d,utt_label_list,num_spk_show,rescale=True,rescale_range=(-1,1)):

      plt.rcParams["figure.figsize"] = (30,30)

      if rescale==True:
         array_2d=np.interp(array_2d, (array_2d.min(), array_2d.max()),rescale_range)
      spk_label_list=[os.path.basename(i).split('_')[0] for i in utt_label_list]
      print(f" unique speakers: {len(set(spk_label_list))}")

      if num_spk_show == "all":
         num_spk_show = len(set(spk_label_list))
      display_spk_list=spk_label_list[:num_spk_show]

      # colors
      preset_color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange','purple']
      gencolor = cm.get_cmap('prism', 1000)
      list_rgba = np.linspace(0, 1, num_spk_show)
      rand_rgba = np.random.uniform(0, 1, num_spk_show)

      # plot axes lim
      if rescale==True:
         plt.xlim(rescale_range)
         plt.ylim(rescale_range)

      i=0
      for speaker,rgba in zip(display_spk_list, list_rgba):
         if num_spk_show <= len(preset_color_list):
            color = preset_color_list[i]
         else:
            color = gencolor(rgba)

         plt.scatter(
            x= array_2d[[speaker == x for x in spk_label_list], 0],
            y= array_2d[[speaker == x for x in spk_label_list], 1],
            c= color,#
            s=5,
            #edgecolors = 'k',
            label= speaker)
         plt.legend(loc=2,prop={'fontsize': "x-small"})
         plt.title('showing {} speakers'.format(i+1),fontdict = {'fontsize' : 20})
         plt.savefig(os.path.join(plot_outdir,'TSNE_{:03d}.png'.format(i+1)),bbox_inches='tight')
         i+=1
      plt.savefig(os.path.join(plot_outdir,'TSNE_ALL.png'),bbox_inches='tight')
      plt.close()

      return

if __name__ == "__main__":

   emb_indir = '/mnt/data1/youngsunhere/data/VCTK/embs_0828_ckpt1'
   tsne_outdir='/mnt/data1/youngsunhere/data/VCTK/embs_0828_tsneout'
   plot_outdir = '/mnt/data1/youngsunhere/data/VCTK/embs_0828_plot'
   num_spk_show = "all"

   plot2d=Plot2D(emb_indir,tsne_outdir,plot_outdir)
   emb_list,utt_label_list=plot2d.load_embs()
   emb_2d_tsne=plot2d.run_tsne(emb_list)
   plot2d.plot_2d(emb_2d_tsne,utt_label_list,num_spk_show=num_spk_show,rescale=True,rescale_range=(-1,1))

