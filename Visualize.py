from __future__ import print_function
import time

import numpy as np
import random
import pandas as pd

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import GFK, CORAL
import MEDA, GA_MEDA, Random_MEDA

def tSNETransform(X):
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(X)
    tsne = TSNE(n_components=2, verbose=0, perplexity=100, n_iter=5000, metric='cosine')
    tsne_results = tsne.fit_transform(pca_result_50)
    return tsne_results

list_datasets = [
    ['SURFa-w'], #,'SURFd-w','SURFa-w''SURFw-c', 'AMZbooks-dvd']
                   # 'SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a', 'SURFc-d', 'SURFc-w', SURFd-w','SURFd-a', 'SURFd-c', 'SURFw-a', 'SURFw-c', 'SURFw-d'],
    # ['DECAF6d-w','DECAF6a-c','DECAF6a-d','DECAF6a-w','DECAF6c-a','DECAF6c-d','DECAF6c-w','DECAF6d-a','DECAF6d-c','DECAF6w-a','DECAF6w-c','DECAF6w-d'],
    # ['ICLEFc-i','ICLEFc-p','ICLEFi-c','ICLEFi-p','ICLEFp-c','ICLEFp-i'],
    # ['Office31amazon-dslr','Office31amazon-webcam','Office31dslr-amazon','Office31dslr-webcam','Office31webcam-amazon','Office31webcam-dslr'],
    # ['OfficeHomeArt-Clipart','OfficeHomeArt-Product','OfficeHomeArt-RealWorld','OfficeHomeClipart-Art',
    #  'OfficeHomeClipart-Product',
    #         'OfficeHomeClipart-RealWorld', 'OfficeHomeProduct-Art','OfficeHomeProduct-Clipart','OfficeHomeProduct-RealWorld','OfficeHomeRealWorld-Art',
    #         'OfficeHomeRealWorld-Clipart',
    #  'OfficeHomeRealWorld-Product'
    #  ],
    # ['AMZbooks-dvd','AMZbooks-elec','AMZbooks-kitchen','AMZdvd-books','AMZdvd-elec','AMZdvd-kitchen','AMZelec-books','AMZelec-dvd',
    #         'AMZelec-kitchen','AMZkitchen-books','AMZkitchen-dvd','AMZkitchen-elec']
                  ]
# list_dims = [20, 50, 70]
list_dims = [20]

for index, datasets in enumerate(list_datasets):
    dim = list_dims[index]
    eta = 0.1
    for dataset in datasets:
        source = np.genfromtxt("/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/" + dataset + "/Source",
                               delimiter=",")
        m = source.shape[1] - 1
        Xs = source[:, 0:m]
        Ys = source[:, m:m + 1]
        Ys = np.ravel(Ys)
        Ys = np.array([int(label) for label in Ys])
        c = np.unique(Ys).shape[0]

        target = np.genfromtxt("/home/nguyenhoai2/Grid/data/TransferLearning/UnPairs/" + dataset + "/Target",
                               delimiter=",")
        Xt = target[:, 0:m]
        Yt = target[:, m:m + 1]
        Yt = np.ravel(Yt)
        Yt = np.array([int(label) for label in Yt])
        ns, nt = Xs.shape[0], Xt.shape[0]

        C = len(np.unique(Ys))
        if C > np.max(Ys):
            Ys = Ys + 1
            Yt = Yt + 1

        run = 1
        random_seed = 1617 * run
        np.random.seed(random_seed)
        random.seed(random_seed)

        meda = MEDA.MEDA(kernel_type='rbf', dim=dim, lamb=10, rho=1.0, eta=eta, p=10, gamma=0.5, T=10, out=None)
        meda_acc, meda_target, _ = meda.fit_predict(Xs, Ys, Xt, Yt)
        print('MEDA acc: %f' %meda_acc)

        gmeda_acc, gmeda_target = GA_MEDA.evolve(Xs, Ys, Xt, Yt, None, 0.2, 1, dim, eta)
        print('G-MEDA acc: %f' %gmeda_acc)

        # r_meda = Random_MEDA.Random_MEDA(kernel_type='rbf', dim=dim, lamb=10, rho=1.0, eta=0.1, p=10, gamma=0.5, T=10,
        #                                  init_op=2, re_init_op=3, run=1, archive_size=10)
        # pmeda_target, pmeda_acc = r_meda.evolve(Xs, Ys, Xt, Yt)
        # print('P-MEDA acc: %f' %pmeda_acc)

        Ys = np.reshape(Ys, (len(Ys), 1))
        Yt = np.reshape(Yt, (len(Yt), 1))
        meda_target = np.reshape(meda_target, (len(meda_target), 1))
        gmeda_target = np.reshape(gmeda_target, (len(gmeda_target), 1))
        # pmeda_target = np.reshape(pmeda_target, (len(pmeda_target), 1))

        Y = np.vstack((Ys, Yt))
        Y_meda = np.vstack((Ys, meda_target))
        Y_gmeda = np.vstack((Ys, gmeda_target))
        # Y_pmeda = np.vstack((Ys, pmeda_target))

        data = pd.DataFrame()
        data['y'] = Y.tolist()
        data['y_meda'] = Y_meda.tolist()
        data['y_gmeda'] = Y_gmeda.tolist()
        # data['y_pmeda'] = Y_pmeda.tolist()

        for index, row in data.iterrows():
            row['y'] = row['y'][0]
            row['y_meda'] = row['y_meda'][0]
            row['y_gmeda'] = row['y_gmeda'][0]
            # row['y_pmeda'] = row['y_pmeda'][0]

        gfk = GFK.GFK(dim=dim)
        _, Xs_new, Xt_new = gfk.fit(Xs, Xt)
        Xs_new, Xt_new = Xs_new.T, Xt_new.T
        X = np.hstack((Xs_new, Xt_new))
        X /= np.linalg.norm(X, axis=0)
        X_gfk = X.T
        Xs_gfk = X[:, :ns].T
        Xt_gfk = X[:, ns:].T
        gfk_results = tSNETransform(X_gfk)
        data['gfk1'] = gfk_results[:, 0]
        data['gfk2'] = gfk_results[:, 1]


        #
        #
        #
        # coral = CORAL.CORAL()
        # Xs_new = coral.fit(Xs, Xt)
        # X = np.hstack((Xs_new.T, Xt.T))
        # X /= np.linalg.norm(X, axis=0)
        # X_cor = X.T
        # Xs_cor = X[:, :ns].T
        # Xt_cor = X[:, ns:].T
        # coral_results = tSNETransform(X_gfk)
        # data['cor1'] = coral_results[:, 0]
        # data['cor2'] = coral_results[:, 1]
        #
        # data['y'] = Y
        #
        # pos_x = -7.5
        # pos_y = -2.5
        # width = 2
        # height = 1.5
        pos_x = -10
        pos_y = -16.5
        width = 10
        height = 6.5
        plt.figure(figsize=(14, 14))
        plt.rcParams.update({'font.size': 40})
        cols = sns.color_palette("hls", c)
        plt.subplots_adjust(hspace=0.4)

        ax1 = plt.subplot(1, 1, 1)
        sns.scatterplot(
            x="gfk1", y="gfk2",
            hue="y",
            palette=cols,
            data=data.iloc[:ns, :],
            # legend="full",
            alpha=1.0,
            s=200,
            ax=ax1
        )
        s_data = data.iloc[:ns, :]
        for r_index in range(len(s_data.index)):
            x = s_data.iloc[r_index]['gfk1']
            y = s_data.iloc[r_index]['gfk2']
            label = s_data.iloc[r_index]['y']
            ax1.annotate(str(label), (x,y), color=cols[int(label)-1],fontsize= 40)
        ax1.set_title('Source')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.get_legend().remove()
        ax1.add_patch(Rectangle((pos_x, pos_y), width, height, facecolor="red", fill=False, color='red', lw=2))
        plt.savefig('Figure/'+dataset + '/Source.eps',  bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 14))
        plt.rcParams.update({'font.size': 40})
        cols = sns.color_palette("hls", c)
        plt.subplots_adjust(hspace=0.4)

        ax1 = plt.subplot(1, 1, 1)
        sns.scatterplot(
            x="gfk1", y="gfk2",
            hue="y",
            palette=cols,
            data=data.iloc[ns:, :],
            # legend="full",
            alpha=1.0,
            s=200,
            ax=ax1
        )
        t_data = data.iloc[ns:, :]
        for r_index in range(len(t_data.index)):
            x = t_data.iloc[r_index]['gfk1']
            y = t_data.iloc[r_index]['gfk2']
            label = t_data.iloc[r_index]['y']
            ax1.annotate(str(label), (x, y), color=cols[int(label) - 1], fontsize=40)
        ax1.set_title('Target')
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.get_legend().remove()
        ax1.add_patch(Rectangle((pos_x, pos_y), width, height, facecolor="red", fill=False, color='red', lw=2))
        plt.savefig('Figure/' + dataset + '/Target.eps', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 14))
        plt.rcParams.update({'font.size': 40})
        cols = sns.color_palette("hls", c)
        plt.subplots_adjust(hspace=0.4)
        ax1 = plt.subplot(1, 1, 1)
        sns.scatterplot(
            x="gfk1", y="gfk2",
            hue="y_meda",
            palette=sns.color_palette("hls", c),
            data=data.iloc[ns:, :],
            # legend="full",
            alpha=1.0,
            s=200,
            ax=ax1
        )
        for r_index in range(len(t_data.index)):
            x = t_data.iloc[r_index]['gfk1']
            y = t_data.iloc[r_index]['gfk2']
            label = t_data.iloc[r_index]['y_meda']
            ax1.annotate(str(label), (x,y), color=cols[int(label)-1],fontsize=40)
        ax1.set_title('MEDA')
        ax1.get_legend().remove()
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.add_patch(Rectangle((pos_x, pos_y), width, height, facecolor="red", fill=False, color='red', lw=2))
        plt.savefig('Figure/' + dataset + '/MEDA.eps', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 14))
        plt.rcParams.update({'font.size': 40})
        cols = sns.color_palette("hls", c)
        plt.subplots_adjust(hspace=0.4)
        ax1 = plt.subplot(1, 1, 1)
        sns.scatterplot(
            x="gfk1", y="gfk2",
            hue="y_gmeda",
            palette=sns.color_palette("hls", c),
            data=data.iloc[ns:, :],
            # legend="full",
            alpha=1.0,
            s=200,
            ax=ax1
        )
        for r_index in range(len(t_data.index)):
            x = t_data.iloc[r_index]['gfk1']
            y = t_data.iloc[r_index]['gfk2']
            label = t_data.iloc[r_index]['y_gmeda']
            ax1.annotate(str(label), (x,y), color=cols[int(label)-1],fontsize=40)
        ax1.set_title('G-MEDA')
        ax1.get_legend().remove()
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.add_patch(Rectangle((pos_x, pos_y), width, height, facecolor="red", fill=False, color='red', lw=2))
        plt.savefig('Figure/' + dataset + '/GMEDA.eps', bbox_inches='tight')
        plt.close()


        # t_data = data.iloc[ns:, :]
        # ax2 = plt.subplot(2, 2, 2)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax2
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y']
        #     ax2.annotate(str(label), (x,y), color=cols[int(label)-1], fontsize=15)
        # ax2.set_title('Target')
        # ax2.get_legend().remove()
        #
        # ax3 = plt.subplot(2, 2, 3)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y_meda",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax3
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y_meda']
        #     ax3.annotate(str(label), (x,y), color=cols[int(label)-1],fontsize=15)
        # ax3.set_title('MEDA Target')
        # ax3.get_legend().remove()
        #
        # ax4 = plt.subplot(2, 2, 4)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y_gmeda",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax4
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y_gmeda']
        #     ax4.annotate(str(label), (x,y), color=cols[int(label)-1],fontsize=15)
        # ax4.set_title('G-MEDA Target')
        # ax4.get_legend().remove()

        # ax5 = plt.subplot(3, 2, 5)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y_pmeda",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax5
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y_pmeda']
        #     ax5.annotate(str(label), (x,y), color=cols[int(label)-1])
        # ax5.set_title('P-MEDA Target')
        # ax5.get_legend().remove()

        # plt.figure(figsize=(21, 14))
        # cols = sns.color_palette("hls", c)
        # plt.subplots_adjust(hspace=0.4)
        #
        # ax1 = plt.subplot(3, 2, 1)
        # s_data = data.iloc[:ns, :]
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y",
        #     palette=cols,
        #     data=data.iloc[ns:, :],
        #     # legend="full",
        #     alpha=1.0,
        #     ax=ax1
        # )
        # for r_index in range(len(s_data.index)):
        #     x = s_data.iloc[r_index]['gfk1']
        #     y = s_data.iloc[r_index]['gfk2']
        #     label = s_data.iloc[r_index]['y']
        #     ax1.annotate(str(label), xy=(x,y), xytext=(x,y), color=cols[int(label)-1])
        # ax1.set_title('Source')
        # ax1.set_xlim(5, 10)
        # ax1.set_ylim(-8, -4)
        # ax1.get_legend().remove()
        #
        # t_data = data.iloc[ns:, :]
        # ax2 = plt.subplot(3, 2, 2)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax2
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y']
        #     ax2.annotate(str(label), (x,y), color=cols[int(label)-1])
        # ax2.set_title('Target')
        # ax2.get_legend().remove()
        #
        # ax3 = plt.subplot(3, 2, 3)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y_meda",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax3
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y_meda']
        #     ax3.annotate(str(label), (x,y), color=cols[int(label)-1])
        # ax3.set_title('MEDA Target')
        # ax3.get_legend().remove()
        #
        # ax4 = plt.subplot(3, 2, 4)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y_gmeda",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax4
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y_gmeda']
        #     ax4.annotate(str(label), (x,y), color=cols[int(label)-1])
        # ax4.set_title('G-MEDA Target')
        # ax4.get_legend().remove()
        #
        # ax5 = plt.subplot(3, 2, 5)
        # sns.scatterplot(
        #     x="gfk1", y="gfk2",
        #     hue="y_pmeda",
        #     palette=sns.color_palette("hls", c),
        #     data=data.iloc[ns:, :],
        #     legend="full",
        #     alpha=1.0,
        #     ax=ax5
        # )
        # for r_index in range(len(t_data.index)):
        #     x = t_data.iloc[r_index]['gfk1']
        #     y = t_data.iloc[r_index]['gfk2']
        #     label = t_data.iloc[r_index]['y_pmeda']
        #     ax5.annotate(str(label), (x,y), color=cols[int(label)-1])
        # ax5.set_title('P-MEDA Target')
        # ax5.get_legend().remove()
        #
        # plt.savefig('Figure/'+dataset + '_zoom.eps')



