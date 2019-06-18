import pandas as pd
import numpy as np
from collections import OrderedDict


list_datasets = [
    ['SURFd-w','SURFa-c','SURFa-d','SURFa-w','SURFc-a','SURFc-d','SURFc-w','SURFd-a','SURFd-c','SURFw-a','SURFw-c','SURFw-d'],
    ['DECAF6d-w','DECAF6a-c','DECAF6a-d','DECAF6a-w','DECAF6c-a','DECAF6c-d','DECAF6c-w','DECAF6d-a','DECAF6d-c','DECAF6w-a','DECAF6w-c','DECAF6w-d'],
    ['ICLEFc-i','ICLEFc-p','ICLEFi-c','ICLEFi-p','ICLEFp-c','ICLEFp-i'],
    ['Office31amazon-dslr','Office31amazon-webcam','Office31dslr-amazon','Office31dslr-webcam','Office31webcam-amazon','Office31webcam-dslr'],
    ['OfficeHomeArt-Clipart','OfficeHomeArt-Product','OfficeHomeArt-RealWorld','OfficeHomeClipart-Art',
     #'OfficeHomeClipart-Product',
            'OfficeHomeClipart-RealWorld', 'OfficeHomeProduct-Art','OfficeHomeProduct-Clipart','OfficeHomeProduct-RealWorld','OfficeHomeRealWorld-Art',
            'OfficeHomeRealWorld-Clipart',
     # 'OfficeHomeRealWorld-Product'
     ],
    ['AMZbooks-dvd','AMZbooks-elec','AMZbooks-kitchen','AMZdvd-books','AMZdvd-elec','AMZdvd-kitchen','AMZelec-books','AMZelec-dvd',
            'AMZelec-kitchen','AMZkitchen-books','AMZkitchen-dvd','AMZkitchen-elec'],
    ['VOC2007-ImageNet', 'ImageNet-VOC2007']
]

for datasets in list_datasets:
    acc = OrderedDict([('Datasets', []), ('1NN', []), ('MEDA',[]), ('Best',[]), ('Evolve',[]), ('10p',[]), ('Pop',[]), ('Archive',[])])

    time =OrderedDict([('Datasets', []), ('MEDA', []), ('GA-MEDA',[])])

    dir = '/home/nguyenhoai2/Grid/results/MEDA-Extend/50-10/'
    for dataset in datasets:
        dir_data = dir+dataset+'/'
        runs = 1
        nn_acc = 0
        meda_acc = 0
        best_acc = []
        evolve_acc = []
        tenp_acc = []
        pop_acc = []
        archive_acc = []
        meda_time = []
        gmeda_time = []
        for run in range(1, runs+1):
            dir_data_run = dir_data + str(run) + '.txt'
            f = open(dir_data_run, 'r')
            fl = f.readlines()
            for l in fl:
                if '1NN accuracy' in l:
                    nn_acc = float(l.split(': ')[1])
                elif 'MEDA accuracy' in l:
                    meda_acc = float(l.split(': ')[1])
                elif 'MEDA time' in l and not('GA-MEDA' in l):
                    meda_time.append(float(l.split(': ')[1]))
                elif 'Accuracy of the best individual' in l:
                    best_acc.append(float(l.split(': ')[1]))
                elif 'Accuracy of the evovled best individual' in l:
                    evolve_acc.append(float(l.split(': ')[1]))
                elif 'Accuracy of the 10% population' in l:
                    tenp_acc.append(float(l.split(': ')[1]))
                elif 'Accuracy of the population' in l:
                    pop_acc.append(float(l.split(': ')[1]))
                elif 'Accuracy of the archive' in l:
                    archive_acc.append(float(l.split(': ')[1]))
                elif 'GA-MEDA time' in l:
                    gmeda_time.append(float(l.split(': ')[1]))

        acc['Datasets'].append(dataset)
        acc['1NN'].append(np.mean(nn_acc)*100)
        acc['MEDA'].append(np.mean(meda_acc)*100)
        acc['Best'].append(np.mean(best_acc)*100)
        acc['Evolve'].append(np.mean(evolve_acc)*100)
        acc['10p'].append(np.mean(tenp_acc)*100)
        acc['Pop'].append(np.mean(pop_acc)*100)
        acc['Archive'].append(np.mean(archive_acc)*100)

        time['Datasets'].append(dataset)
        time['MEDA'].append(np.mean(meda_time))
        time['GA-MEDA'].append(np.mean(gmeda_time))

    df_acc = pd.DataFrame(acc, columns=acc.keys())
    df_acc = df_acc.round(2)
    df_time = pd.DataFrame(time, columns=time.keys())
    df_time = df_time.round(2)

    print(df_acc.to_latex(index=False))
    print(df_time.to_latex(index=False))
