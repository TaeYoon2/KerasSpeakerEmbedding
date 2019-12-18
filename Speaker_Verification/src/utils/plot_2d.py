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

plot_outdir = './eer_data'



class PlotEmb(object):
    def __init__(self,dvector_list,lab_list,num_epoch):

        self.dvector_list=dvector_list
        self.lab_list= lab_list
        self.num_epoch = num_epoch


        return

    def plot_tsne(self,display_spkr_num):

        # TSNE
        tsne = TSNE(n_components=2, random_state=0)

        print("running tsne...")
        timestart_tsne=time.time()
        dvector_arr_2d_tsne = tsne.fit_transform(self.dvector_list)
        #print(dvector_arr_2d_tsne)

        #normalize to -1 ~ +1
        dvector_arr_2d_tsne=np.interp(dvector_arr_2d_tsne, (dvector_arr_2d_tsne.min(), dvector_arr_2d_tsne.max()), (-1, +1))
        #print(dvector_arr_2d_tsne)


        timeend_tsne=time.time()
        print("done tsne, took {} sec".format(timeend_tsne - timestart_tsne))

        print(dvector_arr_2d_tsne)

        spkr_label_list=[]
        utt_label_list = []
        # id(\d+)_(\w+)_(\d+).npy else (\d+)-(\d+)-(\d+).npy'
        for filename in self.lab_list:
            pattern_vox = re.compile(r"(?P<spkr_vox>id\d{5})_(?P<utt_vox>(\S{11})_(\d{5}))\.npy")
            pattern_lib = re.compile(r"(?P<spkr_lib>\d+)-(?P<utt_lib>(\d+)-(\d+))\.npy")

            m_vox=re.search(pattern_vox,filename)
            m_lib=re.search(pattern_lib,filename)

            if m_vox:
                spkr_label_list.append("VOX"+m_vox.group("spkr_vox"))
                utt_label_list.append("VOX"+m_vox.group("spkr_vox") + '_' + m_vox.group("utt_vox"))


            elif m_lib:
                spkr_label_list.append("LIB"+m_lib.group("spkr_lib"))
                utt_label_list.append("LIB"+m_lib.group("spkr_lib") + '_' + m_lib.group("utt_lib"))

            else:
                spkr_label_list.append("_".join(filename.split('_')[:-1]))
                utt_label_list.append(filename.split('.npy')[0])

        gencolor = cm.get_cmap('prism', 1000)

        #rand_rgba = np.random.uniform(0, 1, num_spk_dp)
        fig = plt.figure(figsize=(10, 10))

        #display_num=100

        display_spk_list = sorted(list(set(spkr_label_list)))[:display_spkr_num]
        #list_rgba = np.linspace(0, 1, len(display_spk_list))
        color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']


        # fix xy axis scale
        plt.xlim(-1,1)
        plt.ylim(-1,1)


        for i,spkr in enumerate(display_spk_list):
            #rgba = list_rgba[i]
            #color = gencolor(rgba)
            color = color_list[i]
            plt.scatter(

                x= dvector_arr_2d_tsne[[spkr == x.split('_')[0] for x in utt_label_list], 0],
                y= dvector_arr_2d_tsne[[spkr == x.split('_')[0] for x in utt_label_list], 1],
                c= color,#
                edgecolors = 'k',
                s = 50,
                label= spkr)
            plt.legend()
        plt.title('showing {} speakers'.format(len(display_spk_list)),fontdict = {'fontsize' : 50})
        plt.savefig(os.path.join(plot_outdir,f'TSNE_eng_ge2e_{self.num_epoch:06d}.png'),bbox_inches='tight')
        plt.close()

