import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

datasets = ['SURFa-c', 'Office31amazon-dslr', 'AMZbooks-dvd']
leg_names = ['A->C', 'A->D(31)', 'B->D']
runs = 1
iterations = 10
dicts = {'Dataset':[], 'Iteration':[], 'Distance':[]}

for index, dataset in enumerate(datasets):
    dir = '/home/nguyenhoai2/Grid/results/MEDA-Extend/'+dataset+'/GA-MEDA-Final/'
    leg_name = leg_names[index]
    dis = [0.0]*iterations
    for run in range(1, runs+1):
        dir_data_run = dir + str(run) + '.txt'
        f = open(dir_data_run, 'r')
        fl = f.readlines()
        count = 0
        for l in fl:
            if 'Average distance' in l:
                dis[count] += float(l.split(': ')[1])
                count += 1
    dis = np.array(dis)
    dis = dis/runs
    for iter in range(iterations):
        dicts['Dataset'].append(leg_name)
        dicts['Iteration'].append(iter+1)
        dicts['Distance'].append(dis[iter])

data = pd.DataFrame.from_dict(dicts)

plt.figure(figsize=(21, 14))
plt.rcParams.update({'font.size': 40})
cols = sns.color_palette("hls", len(datasets))
ax = plt.subplot(1, 1, 1)
ax = sns.lineplot(
    x="Iteration", y="Distance",
    hue="Dataset",
    # palette=cols,
    data=data,
    style="Dataset",
    # markers=True,
    dashes=True,
    size= "Dataset",
    sizes = [8,8,8],
    legend="full"
    # alpha=1.0,sizes=10,
    # markers=["circle","square", "star"]
)
ax.set_xlim(left=1, right=10)
ax.set_ylim(bottom=0.001, top=1.0)
ax.set_title('Convergence')
plt.savefig('Figure/Convergence.eps', bbox_inches='tight')
plt.close()




