

import os
import sys
import csv
import math
import numpy
import scipy
import codecs
import winsound
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram




def read_dataset( dataset ):
    fpin = codecs.open('nyms_%s.txt' % dataset, 'r', 'utf-8' )
    nyms = fpin.read().splitlines()
    fpin.close()
    fpin = codecs.open('proj_sizes_%s.txt' % dataset, 'r', 'utf-8' )
    proj_sizes = fpin.read().splitlines()
    fpin.close()
    nym_proj_size = {}
    for i in range(len(proj_sizes)):
        nym_proj_size[nyms[i]] = float(proj_sizes[i])
    fpin = codecs.open('block_weights_%s.txt' % dataset, 'r', 'utf-8' )
    block_weights = fpin.read().splitlines()
    fpin.close()
    for i in range(len(block_weights)):
        if block_weights[i] != '':
            block_weights[i] = float(block_weights[i])
    fpin = codecs.open('blocks_%s.txt' % dataset, 'r', 'utf-8' )
    blocks = fpin.read().splitlines()
    fpin.close()
    for i in range(len(blocks)):
        blocks[i] = blocks[i].split()
        for j in range(len(blocks[i])):
            blocks[i][j] = int(blocks[i][j])
    fpin = codecs.open('element_weights_%s.txt' % dataset, 'r', 'utf-8' )
    element_weights = fpin.read().splitlines()
    fpin.close()
    for i in range(len(element_weights)):
        if element_weights[i] != '':
            element_weights[i] = element_weights[i].split()
        for j in range(len(element_weights[i])):
            element_weights[i][j] = float(element_weights[i][j])
    print('Dataset %s read.\nThere are %d blocks of %d nyms' % ( dataset, len(blocks), len(nyms) ))
    return nyms, blocks, block_weights, element_weights, nym_proj_size





dataset = 'single_cell_data_chosen_1-inf'
outfile = 'ea_'+dataset
treefile = 'bifurcations_'+dataset+'.csv'
figwidth = 3


( nyms, blocks, block_weights, element_weights, nym_proj_size ) = read_dataset( dataset )

print('Reading bifurcations ..')
with codecs.open(treefile, 'rb', 'utf-8') as f:
    reader = csv.reader(f)
    bifurcations = []
    for row in reader:
        bifurcations.append(row)


recurrence_base = 1.8


maxex_feature = []
clusters = []
for i in range(len(nyms)):
    clusters.append([i])
for bif in bifurcations:
    clus1 = int(bif[0])
    clus2 = int(bif[1])
    clus = clusters[clus1]+clusters[clus2]
    clusters.append(clus)
    if int(bif[3])<=999:
        maxex_feature.append(-1)
    else:
        ex = []
        for i in range(len(blocks)):
            exi = 0
            for j in range(len(blocks[i])):
                if blocks[i][j] in clus:
                    exi += element_weights[i][j]
                else:
                    exi -= element_weights[i][j]
            ex.append(exi)
        maxex = ex.index(max(ex))
        maxex_feature.append(maxex)
        print(bif[3],maxex)

feature = 9715

cfunc = {}
for i in range(len(nyms)-1):
    val = 0
    for j in blocks[feature]:
        if j in clusters[i+len(nyms)]:
            val += element_weights[feature][blocks[feature].index(j)]/3.0
    val /= len(clusters[i+len(nyms)])
    cfunc[i+len(nyms)] = '#%02x%02x%02x' % (int(min(255,255*val)),int(min(255,128*val)),int(min(255,255*val)))



overall_min_ent = 10e10
overall_max_ent = -10e10
for i in range(len(bifurcations)):
    for j in range(len(bifurcations[i])):
        bifurcations[i][j] = float(bifurcations[i][j])
    if bifurcations[i][2] < overall_min_ent:
        overall_min_ent = bifurcations[i][2]
    if bifurcations[i][2] > overall_max_ent:
        overall_max_ent = bifurcations[i][2]        

if overall_min_ent > 0:
    overall_min_ent = 0

ent_offset = 0
if overall_min_ent < 0:
    ent_offset = -overall_min_ent
    for i in range(len(bifurcations)):
        bifurcations[i][2] = ent_offset + bifurcations[i][2]

##    assert overall_min_ent>=0, 'There are negative projection entropies! Should not happen unless blocks are multisets. Are they?'

nymshort = []
nym_longest = 0
for nym in nyms:
    nym = nym.replace('_',' ')
##        if len(nym) < 35:
    nymshort.append(nym)
##        else:
##            nymshort.append(nym[0:33]+'..')
    if nym_longest < len(nym):
        nym_longest = len(nym)

print(nym_longest*0.088,figwidth)
fig = plt.figure(figsize=(nym_longest*0.088+figwidth,(len(nyms)+5)*0.20))
mpl.rc('lines', linewidth=2, color='r')

print('Producing dendrogram ..')

dendrogram(bifurcations,orientation='right',labels=nymshort,link_color_func=lambda k: cfunc[k],leaf_font_size=8)

y1,y2 = plt.ylim()
if overall_min_ent < 0:
    plt.plot([ent_offset, ent_offset], [y1, y2], '--', color="black", linewidth=1);

print('Saving dendrogram ..' )

##    plt.title('%s\n%d words in %d blocks' % (plot_title,len(nyms),len(blocks)))

ax=plt.gca() 
ticks = ax.get_xticks()
if overall_min_ent < 0:
    min_x = -(-overall_min_ent // ticks[1]-ticks[0])*(ticks[1]-ticks[0])
else:
    min_x = 0
plt.xticks(np.arange(ent_offset + min_x, ent_offset + overall_max_ent+0.00001, ticks[1]-ticks[0]), np.arange(min_x, overall_max_ent+0.00001, ticks[1]-ticks[0]))



fig.tight_layout()
##    fig.savefig(outfile+'.png')
fig.savefig(outfile+'.pdf')
##    fig.savefig(outfile+'.eps', format='eps', dpi=1000)




