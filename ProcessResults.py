import pandas as pd
import numpy as np
from collections import OrderedDict
import shutil


# import sys, os
# op_datasets = [ 'AMZbooks-dvd','Office31webcam-amazon','SURFd-c' ]
# methods = ['JDA']
# for method in methods:
#     print('************ %s ************' % method)
#     for dataset in op_datasets:
#         print('------- %s --------' %dataset)
#         paras = []
#         accs = []
#         for dim in range(10,110,10):
#             for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
#                 if os.path.isfile('/home/nguyenhoai2/Grid/results/OptimizeParameter/dim_eta/' +dataset+'/'+method+'/'
#                          + str(dim)+'_'+str(alpha)+'.txt'):
#                     f = open('/home/nguyenhoai2/Grid/results/OptimizeParameter/dim_eta/' +dataset+'/'+method+'/'
#                              + str(dim)+'_'+str(alpha)+'.txt','r')
#                     fl = f.readlines()
#                     for l in fl:
#                         # if 'Accuracy of the archive' in l:
#                         paras.append(str(dim)+'_'+str(alpha))
#                         accs.append(float(l))#l.split(': ')[1]))
#                         break
#         accs = np.array(accs)
#         args = accs.argsort()[-5:][::-1]
#         for arg in args:
#             print(paras[arg])
# sys.exit(0)

# import sys, os
# op_datasets = [ 'AMZbooks-dvd','Office31webcam-amazon','SURFd-c' ]
# methods = ['GFK']
# for method in methods:
#     print('************ %s ************' % method)
#     for dataset in op_datasets:
#         print('------- %s --------' %dataset)
#         paras = []
#         accs = []
#         for dim in range(10,110,10):
#                 if os.path.isfile('/home/nguyenhoai2/Grid/results/OptimizeParameter/dim_eta/' +dataset+'/'+method+'/'
#                          + str(dim)+'.txt'):
#                     f = open('/home/nguyenhoai2/Grid/results/OptimizeParameter/dim_eta/' +dataset+'/'+method+'/'
#                              + str(dim)+'.txt','r')
#                     fl = f.readlines()
#                     for l in fl:
#                         # if 'Accuracy of the archive' in l:
#                         paras.append(str(dim))
#                         accs.append(float(l))#l.split(': ')[1]))
#                         break
#         accs = np.array(accs)
#         args = accs.argsort()[-5:][::-1]
#         for arg in args:
#             print(paras[arg])
# sys.exit(0)


list_datasets = [
    ['AMZbooks-dvd','AMZbooks-elec','AMZbooks-kitchen','AMZdvd-books','AMZdvd-elec','AMZdvd-kitchen','AMZelec-books','AMZelec-dvd',
     'AMZelec-kitchen','AMZkitchen-books','AMZkitchen-dvd','AMZkitchen-elec'],
    ['SURFa-c', 'SURFa-d', 'SURFa-w', 'SURFc-a', 'SURFc-d', 'SURFc-w','SURFd-w', 'SURFd-a', 'SURFd-c',
     'SURFw-a', 'SURFw-c', 'SURFw-d'],
    ['Office31amazon-dslr','Office31amazon-webcam','Office31dslr-amazon','Office31dslr-webcam','Office31webcam-amazon','Office31webcam-dslr'],
    # ['DECAF6d-w','DECAF6a-c','DECAF6a-d','DECAF6a-w','DECAF6c-a','DECAF6c-d','DECAF6c-w','DECAF6d-a','DECAF6d-c','DECAF6w-a','DECAF6w-c','DECAF6w-d'],
    # ['ICLEFc-i','ICLEFc-p','ICLEFi-c','ICLEFi-p','ICLEFp-c','ICLEFp-i'],

    # ['OfficeHomeArt-Clipart','OfficeHomeArt-Product','OfficeHomeArt-RealWorld','OfficeHomeClipart-Art',
    #  'OfficeHomeClipart-Product',
    #         'OfficeHomeClipart-RealWorld', 'OfficeHomeProduct-Art','OfficeHomeProduct-Clipart','OfficeHomeProduct-RealWorld','OfficeHomeRealWorld-Art',
    #         'OfficeHomeRealWorld-Clipart',
    #  'OfficeHomeRealWorld-Product'
    #  ],

    # ['Caltech101-ImageNet', 'Caltech101-SUN09', 'ImageNet-Caltech101', 'ImageNet-SUN09', 'SUN09-ImageNet', 'SUN09-Caltech101']
    # ['VOC2007-ImageNet', 'ImageNet-VOC2007']
]

for datasets in list_datasets:
    acc = OrderedDict([('Datasets', []), ('1NN', [])])
    # ('Best',[]),
    # ('10p',[]), ('Pop',[]), ('Archive',[])]

    # traditionals = [ 'TCA', 'JDA', 'TJM', 'JGSA', 'GFK']
    traditionals = [ "RF", "LSVM", "SVM"]
    for method in traditionals:
        acc.update({method : []})

    time =OrderedDict([('Datasets', []), ('MEDA', [])])
    acc.update({'MEDA':[]})

    dir = '/home/nguyenhoai2/Grid/results/MEDA-Extend/'
    heu_methods = ['P-MEDA','GA-MEDA-Final']
    for method in heu_methods:
        acc.update({method: []})
        time.update({method: []})


    for dataset in datasets:
        time['Datasets'].append(dataset)
        acc['Datasets'].append(dataset)

        list_acc = OrderedDict()

        for method in traditionals:
            f = open(dir+dataset+'/'+method+".txt",'r')
            fl = f.readlines()
            for l in fl:
                tra_acc = float(l)
                list_acc.update({method : [tra_acc]*30})
                acc[method].append(tra_acc*100)
                break

        for index, method in enumerate(heu_methods):
            list_acc.update({method: []})
            list_acc.update({'MEDA': []})

            if 'GA-MEDA' in method:
                dir_data = dir+dataset+'/'+method+'/'
                runs = 30
                nn_acc = 0
                meda_acc = 0
                best_acc = []
                evolve_acc = []
                tenp_acc = []
                pop_acc = []
                arc_acc = []
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
                            arc_acc.append(float(l.split(': ')[1]))
                        elif 'GA-MEDA time' in l:
                            gmeda_time.append(float(l.split(': ')[1]))

                list_acc['MEDA'] = [meda_acc]*runs
                list_acc[method] = evolve_acc

                if index == len(heu_methods)-1:
                    acc['1NN'].append(np.mean(nn_acc)*100)
                    acc['MEDA'].append(np.mean(meda_acc)*100)
                    time['MEDA'].append(np.mean(meda_time))
                # acc['Best'].append(np.mean(best_acc)*100)
                acc[method].append(np.mean(evolve_acc)*100)
                time[method].append(np.mean(gmeda_time))
                #
                # acc['Best'].append(np.mean(best_acc)*100)
                # acc['10p'].append(np.mean(tenp_acc)*100)
                # acc['Pop'].append(np.mean(pop_acc)*100)
                # acc['Arc'].append(np.mean(arc_acc)*100)

                # acc['10p'].append(np.mean(tenp_acc)*100)
                # acc['Pop'].append(np.mean(pop_acc)*100)
                # acc['Archive'].append(np.mean(archive_acc)*100)
            elif method == 'P-MEDA':
                dir_data = dir+dataset+'/'+method+'/'
                runs = 30
                p_acc = []
                p_time = []
                for run in range(1, runs+1):
                    dir_data_run = dir_data + str(run) + '.txt'
                    f = open(dir_data_run, 'r')
                    fl = f.readlines()
                    for line in fl:
                        if 'Accuracy archive:' in line:
                            p_acc.append(float(line.split(':')[1]))
                        elif 'Execution time' in line:
                            p_time.append(float(line.split(':')[1]))

                list_acc[method] = p_acc

                acc[method].append(np.mean(p_acc)*100)
                time[method].append(np.mean(p_time))

        keys = list_acc.keys()
        f_out = open('SigTest/'+dataset, 'w')
        for index, key in enumerate(keys):
            if index < len(keys)-1:
                f_out.write(key+', ')
            else:
                f_out.write(key+'\n')

        for run in range(0, 30):
            for index, key in enumerate(keys):
                if index < len(keys) - 1:
                    f_out.write(str(list_acc[key][run]) + ', ')
                else:
                    f_out.write(str(list_acc[key][run]) + '\n')
        f_out.close()

    acc['Datasets'].append('Ave')
    for key in acc.keys():
        if key != 'Datasets':
            acc[key].append(np.mean(acc[key]))

    df_acc = pd.DataFrame(acc, columns=acc.keys())
    df_acc = df_acc.round(2)
    df_time = pd.DataFrame(time, columns=time.keys())
    df_time = df_time.round(2)

    print(df_acc.to_latex(index=False))
    print(df_time.to_latex(index=False))
