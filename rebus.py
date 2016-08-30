#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# — — ———  REBUS: entropy agglomeration of words in a text  ——— — —  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# On this software implementation, please refer to:                  #
# Fidaner, I. B. & Cemgil, A. T. (2014) Clustering Words by Projec-  #
#     tion Entropy, submitted to NIPS 2014 Modern ML+NLP Workshop    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# On entropy agglomeration and related concepts, please refer to:    #
# Fidaner, I. B. & Cemgil, A. T. (2013) Summary Statistics for Par-  #
#     titionings and Feature Allocations. In Advances in Neural In-  #
#     formation Processing Systems, 26.                              #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#  — — ———  This work is under GNU General Public License  ——— — —   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import os
import sys
import csv
import math
import numpy
import scipy
import codecs
#import winsound
import traceback
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def projection_entropy( nym_count, blocks, block_weights, subset ):
    assert sorted(list(set(subset))) == sorted(subset), 'This cluster is not a set: %r' % subset
    pe = 0
    proj = [filter(lambda x: x in subset, sublist) for sublist in blocks]
    for i in range(len(list(proj))):
        if proj[i] != [] and block_weights[i] > 0:
            p = len(list(proj[i]))/float(len(subset))
            if p>0:
                pe += - p * math.log(p) * block_weights[i]
    return pe

def cumulative_occurences( nym_count, blocks, block_weights, subset ):
    assert sorted(list(set(subset))) == sorted(subset), 'This cluster is not a set: %r' % subset
    cod = []
    for i in range(len(subset)+1):
        cod.append(0)
    proj = [filter(lambda x: x in subset, sublist) for sublist in blocks]
    for i in range(len(list(proj))):
        for k in range(len(list(proj[i]))+1):
            cod[k] += 1
    return cod


def save_dataset( dataset_name, nyms, nym_proj_size, blocks ):
    fpin = codecs.open('nyms_%s.txt' % dataset_name, 'w', 'utf-8' )
    for i in range(len(nyms)):
        fpin.write('%s\n' % nyms[i])
    fpin.close()
    fpin = codecs.open('nym_proj_size_%s.csv' % dataset_name, 'w', 'utf-8' )
    for i in range(len(nyms)):
        fpin.write('"%s", %d\n' % ( nyms[i], nym_proj_size[ nyms[i] ] ) )
    fpin.close()
    fpin = codecs.open('proj_sizes_%s.txt' % dataset_name, 'w', 'utf-8' )
    for i in range(len(nyms)):
        fpin.write('%d\n' % nym_proj_size[ nyms[i] ])
    fpin.close()
    fpin = codecs.open('blocks_%s.txt' % dataset_name, 'w', 'utf-8' )
    for block in blocks:
        for elm in block:
            fpin.write('%d ' % elm)
        fpin.write('\n')
    fpin.close()
    fpin = codecs.open('weights_%s.txt' % dataset_name, 'w', 'utf-8' )
    block_weight = 1.0 / float( len(blocks) )
    for block in blocks:
        if block == []:
            fpin.write('\n')
        else:
            fpin.write('%g\n' % block_weight)
    fpin.close()
    print('Dataset %s saved.' % dataset_name)
    

def read_dataset( dataset ):
    fpin = codecs.open('nyms_%s.txt' % dataset, 'r', 'utf-8' )
    nyms = fpin.read().splitlines()
    fpin.close()
    fpin = codecs.open('proj_sizes_%s.txt' % dataset, 'r', 'utf-8' )
    proj_sizes = fpin.read().splitlines()
    fpin.close()
    nym_proj_size = {}
    for i in range(len(proj_sizes)):
        nym_proj_size[nyms[i]] = int(proj_sizes[i])
    fpin = codecs.open('weights_%s.txt' % dataset, 'r', 'utf-8' )
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
    print('Dataset %s read.\nThere are %d blocks of %d nyms' % ( dataset, len(blocks), len(nyms) ))
    return nyms, blocks, block_weights, nym_proj_size


def prepare_text_all( text_name_all, text_filename, merge_lines ):

    print('Preparing text %s ..' % text_name_all)

    fpin = codecs.open(text_filename, 'r', 'utf-8' )
    lines = fpin.read().splitlines()
    fpin.close()
    
    for i in range(len(lines)):
        lines[i]=lines[i].lower()
        lines[i]=lines[i].replace(u"–",u" ")
        lines[i]=lines[i].replace(u"—",u" ")
        lines[i]=lines[i].replace(u"`",u"'")
        lines[i]=lines[i].replace(u"´",u"'")
        lines[i]=lines[i].replace(u"’",u"'")
        lines[i]=lines[i].replace(u"‘",u"'")
        lines[i]=lines[i].split()
    i=0
    while i<len(lines):
        if merge_lines==1 and len(lines[i])>0 and i<len(lines)-1 and len(lines[i+1])>0:
            if lines[i][len(lines[i])-1][-1:]=='-':
                lines[i][len(lines[i])-1] = lines[i][len(lines[i])-1][:-1] + lines[i+1][0]
                lines[i] = lines[i] + lines[i+1][1:]
            else:
                lines[i] = lines[i] + lines[i+1]
            del lines[i+1]
            i-=1

        if len(lines[i])==0:
            del lines[i]
            i-=1
        for j in range(len(lines[i])):
            j2 = 0
            while j2<len(lines[i][j]):
                if not (lines[i][j][j2].isalpha() or lines[i][j][j2]==u"'" or lines[i][j][j2]==u"-"):
                    newstr = ''
                    if j2>0:
                        newstr = lines[i][j][:j2] + newstr
                    if j2<len(lines[i][j])-1:
                        newstr = newstr + lines[i][j][j2+1:]
                    lines[i][j] = newstr
                    j2-=1
                j2+=1
            if lines[i][j]==u"'":
                lines[i][j]=''
            else:
                if lines[i][j]!='':
                    while lines[i][j][0]==u"'":
                        lines[i][j] = lines[i][j][1:]
                    while lines[i][j][-1]==u"'":
                        lines[i][j] = lines[i][j][0:-1]
        i+=1
        
    blocks = []
    nym_id = {}
    nyms = []
    nym_count = 0
    for i in range(len(lines)):
        block = []
        for j in range(len(lines[i])):
            if len(lines[i][j])>0:
                if lines[i][j] in nym_id:
                    block.append(nym_id[lines[i][j]])
                else:
                    nym_count += 1
                    nym_id[lines[i][j]] = nym_count
                    nyms.append(lines[i][j])
                    block.append(nym_count)
        blocks.append(block)

    nym_proj_size = {}
    for i in range(len(nyms)):
        nym_proj_size[nyms[i]]=0
    for i in range(len(blocks)):
        blocks[i] = list(set(blocks[i]))
        for j in range(len(blocks[i])):
            nym_proj_size[nyms[blocks[i][j]-1]] += 1

    save_dataset( text_name_all, nyms, nym_proj_size, blocks )

    return len(blocks)


def prepare_text_chosen( min_proj_size, max_proj_size, text_name_all, text_name_chosen, text_filename, merge_lines ):

    print('Preparing text %s ..' % text_name_chosen)

    ( nyms, blocks, block_weights, nym_proj_size ) = read_dataset( text_name_all )


    id_map = []
    chosen_nym_count = 0
    chosen_nym_id = {}
    chosen_nyms = []
    for i in range(len(nyms)):
        id_map.append(-1)
        if nym_proj_size[ nyms[i] ] >= min_proj_size and nym_proj_size[ nyms[i] ] <= max_proj_size:
            chosen_nym_count += 1
            chosen_nym_id[ nyms[i] ] = chosen_nym_count
            id_map[i] = chosen_nym_count
            chosen_nyms.append( nyms[i] )
    i=0
    while i<len(blocks):
        j=0
        while j<len(blocks[i]):
            if id_map[blocks[i][j]-1]!=-1:
                blocks[i][j] = id_map[blocks[i][j]-1]
            else:
                del blocks[i][j]
                j-=1
            j+=1
        if len(blocks[i])==0:
            del blocks[i]
            i-=1
        i+=1    
            

    save_dataset( text_name_chosen, chosen_nyms, nym_proj_size,blocks)


	
def log_cluster_pair( log_file, min_pe, clusters, nyms, cod ):
    #log_file.write('%5d,%5d : ( ' % (min_pe[1], min_pe[2]) )
    log_file.write('COD: [ ')
    for i in range(len(cod)):
        log_file.write('%5d ' % cod[i])
    log_file.write('] : ( ')
    for i in range(len(clusters[min_pe[1]])):
        log_file.write('%s ' % nyms[clusters[min_pe[1]][i]-1])
    log_file.write(')+( ')
    for i in range(len(clusters[min_pe[2]])):
        log_file.write('%s ' % nyms[clusters[min_pe[2]][i]-1])
    log_file.write(') : %.30f\n' % min_pe[0])

    log_file.flush()
    

def agglomerate_dataset(dataset, log_file):

    # read the nyms and blocks of the dataset
    ( nyms, blocks, block_weights, nym_proj_size ) = read_dataset( dataset )

    # compute total weight for != 1
    total_weight=0
    for i in range(len(block_weights)):
        if block_weights[i] != '':
            total_weight += block_weights[i]
    #if total_weight != 1:
    #    print('Total weight appears to be “%g” but is not exactly One.\nTo be handled nevertheless.' % total_weight)

    print('Begin entropy agglomeration ..')

    # begin with singleton clusters
    clusters = []
    bfc_inds = []
    bifurcations = []
    for i in range(len(nyms)):
        clusters.append( [i+1] )
        bfc_inds.append( i )
    bfc_count = len(nyms)
    merge_entropies = []

    # initialize the cache of minimum projection entropies
    min_pe_cache = []
    min_pe_cache.append( [float('inf'), 0, 0 ] )

    # compute the initial matrix and cache the cluster pairs with minimum projection entropy
    for i in range(len(nyms)):
        entropies = []
        for j in range(i):
            pe = projection_entropy( len(nyms), blocks, block_weights, clusters[j] + clusters[i] )
            entropies.append( pe / float(total_weight) )
            if entropies[j] < min_pe_cache[-1][0]:
                min_pe_cache.append( [entropies[j], j, i] )
                #log_cluster_pair( log_file, min_pe_cache[-1], clusters, nyms)
        merge_entropies.append( entropies )

    # merge the best cluster pair and update the matrix
    # continue until only one cluster remains
    while len(merge_entropies) > 1:

        cod = cumulative_occurences( len(nyms), blocks, block_weights, clusters[min_pe_cache[-1][1]] + clusters[min_pe_cache[-1][2]] )

        # record the bifurcation on the dendrogram
        bifurcation = [ bfc_inds[min_pe_cache[-1][1]], bfc_inds[min_pe_cache[-1][2]] ]
        bifurcation.append( min_pe_cache[-1][0] )
        bifurcation.append( len( clusters[min_pe_cache[-1][1]] + clusters[min_pe_cache[-1][2]]) )
        bifurcations.append( bifurcation )
        del bfc_inds[min_pe_cache[-1][2]]
        bfc_inds[min_pe_cache[-1][1]] = bfc_count
        bfc_count += 1

        # log the event of merging
        #log_file.write('%5d,%5d Merged!\n' % (min_pe_cache[-1][1],min_pe_cache[-1][2]))
        #log_file.flush()
        log_cluster_pair( log_file, min_pe_cache[-1], clusters, nyms, cod)
        log_cluster_pair( sys.stdout, min_pe_cache[-1], clusters, nyms, cod)

        # delete invalidated cache entries
        for i in range(len(min_pe_cache)-2,-1,-1):
            if min_pe_cache[i][2] >= min_pe_cache[-1][1]:
                del min_pe_cache[i] 

        # remove the second cluster from the matrix
        del merge_entropies[min_pe_cache[-1][2]]
        for i in range(min_pe_cache[-1][2],len(merge_entropies)):
            del merge_entropies[i][min_pe_cache[-1][2]]

        # merge the cluster pair to the first cluster
        clusters[min_pe_cache[-1][1]] += clusters[min_pe_cache[-1][2]]
        del clusters[min_pe_cache[-1][2]]

        # recompute the first cluster on the matrix
        j = min_pe_cache[-1][1]
        for i in range(j+1,len(merge_entropies)):
            pe = projection_entropy( len(nyms), blocks, block_weights, clusters[j] + clusters[i] )
            merge_entropies[i][j] = pe / float(total_weight)
        i = min_pe_cache[-1][1]
        for j in range(i):
            pe = projection_entropy( len(nyms), blocks, block_weights, clusters[j] + clusters[i] )
            merge_entropies[i][j] = pe / float(total_weight)

        # delete the cache entry for the merged pair
        del min_pe_cache[-1]
        if len(min_pe_cache)==0:
            min_pe_cache.append( [ inf, 0, 0 ] )
                   
        # begin from the next entry
        i0 = min_pe_cache[-1][2]
                   
        # cache the cluster pairs with minimum projection entropy
        for i in range(i0,len(merge_entropies)):
            for j in range(i):
                if merge_entropies[i][j] < min_pe_cache[-1][0]:
                    min_pe_cache.append( [merge_entropies[i][j], j, i] )
                    #log_cluster_pair( log_file, min_pe_cache[-1], clusters, nyms )

    log_file.close()

    print('Writing bifurcations ..')
    with codecs.open('bifurcations_'+dataset+'.csv', 'wb', 'utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(bifurcations)


def draw_dendrogram( plot_title, treefile, dataset, outfile ):

    def cfunc(k):
        return '#dd0000'

    ( nyms, blocks, block_weights, nym_proj_size ) = read_dataset( dataset )

    print('Reading bifurcations ..')
    with codecs.open(treefile, 'rb', 'utf-8') as f:
        reader = csv.reader(f)
        bifurcations = []
        for row in reader:
            bifurcations.append(row)

    overall_min_ent = 10e10
    overall_max_ent = -10e10
    for i in range(len(bifurcations)):
        for j in range(len(bifurcations[i])):
            bifurcations[i][j] = float(bifurcations[i][j])
        if bifurcations[i][2] < overall_min_ent:
            overall_min_ent = bifurcations[i][2]
        if bifurcations[i][2] > overall_max_ent:
            overall_max_ent = bifurcations[i][2]

    assert overall_min_ent>=0, 'There are negative projection entropies! Should not happen unless blocks are multisets. Are they?'
    
    nymshort = []
    for nym in nyms:
        if len(nym) < 25:
            nymshort.append(nym)
        else:
            nymshort.append(nym[1:23]+'..')

    fig = plt.figure(figsize=(4,len(nyms)*0.22))
    mpl.rc('lines', linewidth=3, color='r')
    dendrogram(bifurcations,orientation='left',labels=nymshort,link_color_func=cfunc,leaf_font_size=10)

    print('Saving dendrogram for %s ..' % plot_title)

    plt.title('%s\n%d words in %d blocks' % (plot_title,len(nyms),len(blocks)))
    if overall_max_ent<0.06:
        plt.xticks(np.arange(0, overall_max_ent+0.01, 0.01))
    else:
        if overall_max_ent<0.10:
            plt.xticks(np.arange(0, overall_max_ent+0.01, 0.02))
        else:
            plt.xticks(np.arange(0, overall_max_ent+0.01, 0.05))
    
    fig.tight_layout()
    fig.savefig(outfile+'.png')
    fig.savefig(outfile+'.pdf')
    fig.savefig(outfile+'.eps', format='eps', dpi=1000)




#~~~~~~~~~~~~~~~~~~~~~~~#
#    REBUS main code    #
#~~~~~~~~~~~~~~~~~~~~~~~#

text_names = [ 'ulysses' ]
# put several text names to process several inputs in sequence

merge_lines = 0
# 0: put each line in a separate paragraph (separated by single newlines)
# 1: put consequent lines in the same paragraph (separated by double newlines)

min_proj_sizes = [10, 11, 12, 15, 20, 30, 40, 60,  150]
max_proj_sizes = [10, 11, 13, 17, 25, 39, 59, 149, inf]
# for each of the ranges, minimum and maximum number of paragraphs that the words are allowed to occur in

for text_name in text_names:
    assert os.path.exists(text_name)==0, 'You will need to remove this directory: %s.' % text_name

for text_name in text_names:

    os.mkdir(text_name)
    print('Directory created: %s' % text_name)

    text_filename = '%s.txt' % text_name
    text_name_all = '%s_all' % text_name
    
    outfiles0=[]
    outfiles0.append('nyms_%s.txt' % text_name_all)
    outfiles0.append('nym_proj_size_%s.csv' % text_name_all)
    outfiles0.append('proj_sizes_%s.txt' % text_name_all)
    outfiles0.append('blocks_%s.txt' % text_name_all)
    outfiles0.append('weights_%s.txt' % text_name_all)
    
    print('---')

    try:
            
        block_count = prepare_text_all(text_name_all,text_filename,merge_lines)

        for i in range(len(max_proj_sizes)):
            if max_proj_sizes[i] > block_count:
                max_proj_sizes[i] = block_count

        for i in range(len(max_proj_sizes)):
            min_proj_size = min_proj_sizes[i]
            max_proj_size = max_proj_sizes[i]

            assert(min_proj_size <= max_proj_size)

            text_name_chosen = '%s_chosen_%g-%g' % ( text_name, min_proj_size, max_proj_size )

            prepare_text_chosen(min_proj_size,max_proj_size,text_name_all,text_name_chosen,text_filename,merge_lines)

            plot_title = 'Projection sizes %d-%d' % (min_proj_size,max_proj_size)

            dataset = text_name_chosen

            
            outfiles=[]
            outfiles.append('nyms_%s.txt' % dataset)
            outfiles.append('nym_proj_size_%s.csv' % dataset)
            outfiles.append('proj_sizes_%s.txt' % dataset)
            outfiles.append('blocks_%s.txt' % dataset)
            outfiles.append('weights_%s.txt' % dataset)
            outfiles.append('ea_%s.pdf' % dataset)
            outfiles.append('ea_%s.png' % dataset)
            outfiles.append('ea_%s.eps' % dataset)
            outfiles.append('bifurcations_%s.csv' % dataset)
            outfiles.append('agglomerate_%s.log' % dataset)
            
            log_file = codecs.open('agglomerate_%s.log' % dataset, 'w', 'utf-8' )

            print('---')
        
            try:
                agglomerate_dataset( dataset, log_file )

                dataset = text_name_chosen
                outfile = 'ea_'+dataset
                treefile = 'bifurcations_'+dataset+'.csv'

                draw_dendrogram( plot_title, treefile, dataset, outfile )

                for outfile in outfiles:
                    os.rename(outfile,text_name+'/'+outfile)

            except Exception as e:
                print(e)
                for frame in traceback.extract_tb(sys.exc_info()[2]):
                    fname,lineno,fn,text = frame
                    print("Error in %s on line %d" % (fname, lineno))
                print('~ Exception! ~')
                log_file.close()
                for outfile in outfiles:
                    if os.path.isfile(outfile):
                        print('Removing %s ..' % outfile) 
                        os.remove(outfile)
                print('~~~')
                   
        for outfile in outfiles0:
            os.rename(outfile,text_name+'/'+outfile)

        #winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)

    except Exception as e:
        print(e)
        for frame in traceback.extract_tb(sys.exc_info()[2]):
            fname,lineno,fn,text = frame
            print("Error in %s on line %d" % (fname, lineno))
        print('~ Exception! ~')
        for outfile in outfiles0:
            if os.path.isfile(outfile):
                print('Removing %s ..' % outfile)
                os.remove(outfile)
        print('~~~')
        raise
        


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#  — — ———  This work is under GNU General Public License  ——— — —   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# The dream-thoughts and the dream-content are presented to us like  #
# two versions of the same subject-matter in two different langua-   #
# ges.  Or, more properly, the dream-content seems like a transcript #
# of the dream-thoughts into another mode of expression, whose cha-  #
# racters and syntactic laws it is our business to discover by com-  #
# paring the original and the translation.  The dream-thoughts are   #
# immediately comprehensible, as soon as we have learnt them.  The   #
# dream-content, on the other hand, is expressed as it were in a     #
# pictographic script, the characters of which have to be transposed #
# individually into the language of the dream-thoughts.  If we at-   #
# tempted to read these characters according to their pictorial va-  #
# lue instead of according to their symbolic relation, we should c-  #
# learly be led into error.  Suppose I have a picture-puzzle, a re-  #
# bus, in front of me.  It depicts a house with a boat on its roof,  #
# a single letter of the alphabet, the figure of a running man whose #
# head has been conjured away, and so on.  Now I might be misled in- #
# to raising objections and declaring that the picture as a whole    #
# and its component parts are nonsensical.  A boat has no business   #
# to be on the roof of a house, and a headless man cannot run.  Mo-  #
# reover, the man is bigger than the house; and if the whole picture #
# is intended to represent a landscape, letters of the alphabet are  #
# out of place in it since such objects do not occur in nature.  But #
# obviously we can only form a proper judgement of the rebus if we   #
# put aside criticisms such as these of the whole composition and    #
# its parts and if, instead, we try to replace each separate element #
# by a syllable or word that can be represented by that element in   #
# some way or other.  The words which are put together in this way   #
# are no longer nonsensical but may form a poetical phrase of the g- #
# reatest beauty and significance.  A dream is a picture puzzle of   #
# this sort and our predecessors in the field of dream interpreta-   #
# tion have made the mistake of treating the rebus as a pictorial    #
# composition: and as such it has seemed to them nonsensical and     #
# worthless.                                                         #
#      Sigmund Freud (1899) The Interpretation of Dreams, Chapter VI #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# — — ———  REBUS: entropy agglomeration of words in a text  ——— — —  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# -*- coding: utf-8 -*-
#––––––––––✂–––––––––––––––––––
#٦1OФ٢ÇΒي态ΛΖ٤BYT6鮞拻AXΛ忖3ΔΜY粴J3٧
#IU峧ÇЩFMن峹ΤثWΣΠزع7MD3٤騺WΔKL辷KPO
#ΗKENCOUNTED7٦ཆO鍐٤قPΛ蹋٧瑸3١椿X9
#⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋⪌⪋
