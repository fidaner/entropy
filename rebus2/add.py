﻿#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# — —— ————  REBUS 2.0: entropy agglomeration of elements  ——— —— —  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# On this version of the software implementation, please refer to:   #
# Fidaner, I. B. (2017) Generalized Entropy Agglomeration, arxiv.org #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# On the previous version of the software implementation, refer to:  #
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
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def projection_entropy( nym_count, blocks, block_weights, element_weights, subset, recurrence_base ):
##    assert sorted(list(set(subset))) == sorted(subset), 'This cluster is not a set: %r' % subset
    pe = 0
    proj_indices = []    
    for i in range(len(blocks)):
        proj_indices.append([ind+1 for ind,val in enumerate(blocks[i]) if val in subset])
    for i in range(len(blocks)):
        if proj_indices[i] != [] and block_weights[i] > 0:
            p = 0
            for j in proj_indices[i]:
                if element_weights == []:
                    p += 1
                else:
                    p += element_weights[i][j-1]
            p = p / ( float(len(subset)) * recurrence_base )
            if p>0:
                pe += - p * math.log(p) * block_weights[i]

    return pe

def cumulative_occurences( nym_count, blocks, subset ):
##    assert sorted(list(set(subset))) == sorted(subset), 'This cluster is not a set: %r' % subset
    cod = []
    for i in range(len(subset)+1):
        cod.append(0)
    proj_indices = []    
    for i in range(len(blocks)):
        proj_indices.append([ind+1 for ind,val in enumerate(blocks[i]) if val in subset])
    for i in range(len(blocks)):
        while len(cod) < len(proj_indices[i])+1:
            cod.append(0) # longer COD for repeated elements
        for k in range(len(proj_indices[i])+1):
            cod[k] += 1
    return cod


def save_dataset( dataset_name, nyms, nym_proj_size, blocks, block_weights, element_weights ):
    assert len(blocks) == len(block_weights), 'Number of blocks %d, number of block weights %d!' % (len(blocks),len(block_weights))
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
        fpin.write('%g\n' % nym_proj_size[ nyms[i] ])
    fpin.close()
    fpin = codecs.open('blocks_%s.txt' % dataset_name, 'w', 'utf-8' )
    for block in blocks:
        for elm in block:
            fpin.write('%d ' % elm)
        fpin.write('\n')
    fpin.close()
    fpin = codecs.open('block_weights_%s.txt' % dataset_name, 'w', 'utf-8' )
    sum_bw = sum(block_weights)
    for ind,block in enumerate(blocks):
        if block == []:
            fpin.write('\n')
        else:
            fpin.write('%g\n' % (block_weights[ind] / sum_bw))
    fpin.close()
    fpin = codecs.open('element_weights_%s.txt' % dataset_name, 'w', 'utf-8' )
    for element_weight_list in element_weights:
        for element_weight in element_weight_list:
            fpin.write('%g ' % element_weight)
        fpin.write('\n')
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


def prepare_text_all( text_name_all, text_filename, block_weights_filename, element_weights_filename, merge_lines, element_weight_power ):

    print('Preparing text %s ..' % text_name_all)

    fpin = codecs.open(text_filename, 'r', 'utf-8' )
    lines = fpin.read().splitlines()
    fpin.close()
    
    for i in range(len(lines)):
        # CONVERT LETTERS TO LOWERCASE
        #lines[i]=lines[i].lower()
        # CONVERT NON-STANDARD DASHES TO SPACES
        lines[i]=lines[i].replace(u"–",u" ")
        lines[i]=lines[i].replace(u"—",u" ")
        # CONVERT SINGLE QUOTATIONS TO A STANDARD MARK
        lines[i]=lines[i].replace(u"`",u"'")
        lines[i]=lines[i].replace(u"´",u"'")
        lines[i]=lines[i].replace(u"’",u"'")
        lines[i]=lines[i].replace(u"‘",u"'")
        # split line into words
        lines[i]=lines[i].split()
    i=0

    print('preparing lines...')
    while i<len(lines):
        if merge_lines==1 and len(lines[i])>0 and i<len(lines)-1 and len(lines[i+1])>0:
            # MERGE THIS LINE WITH THE PREVIOUS ONE
            if lines[i][len(lines[i])-1][-1:]=='-':
                lines[i][len(lines[i])-1] = lines[i][len(lines[i])-1][:-1] + lines[i+1][0]
                lines[i] = lines[i] + lines[i+1][1:]
            else:
                lines[i] = lines[i] + lines[i+1]
            del lines[i+1]
            i-=1

        # disregard empty lines
        if len(lines[i])==0:
            del lines[i]
            i-=1
        for j in range(len(lines[i])):
            j2 = 0
            while j2<len(lines[i][j]):
                # ELIMINATE NO CHARS #
                if 0:
                # ELIMINATE NON-ALPHANUMERIC+++ CHARS # if not (lines[i][j][j2].isalnum() or lines[i][j][j2]==u"." or lines[i][j][j2]==u"'" or lines[i][j][j2]==u"-" or lines[i][j][j2]==u"_" or lines[i][j][j2]==u"#" or lines[i][j][j2]==u"@" or lines[i][j][j2]==u":" or lines[i][j][j2]==u"/"):
                # ELIMINATE NON-ALPHANUMERIC+ CHARS # if not (lines[i][j][j2].isalnum() or lines[i][j][j2]==u"." or lines[i][j][j2]==u"," or lines[i][j][j2]==u"'" or lines[i][j][j2]==u"-" or lines[i][j][j2]==u"+" or lines[i][j][j2]==u"_" or lines[i][j][j2]==u"(" or lines[i][j][j2]==u")"):
                # ELIMINATE NON-ALPHA+ CHARS # if not (lines[i][j][j2].isalpha() or lines[i][j][j2]==u"'" or lines[i][j][j2]==u"-" or lines[i][j][j2]==u"_"):
                    newstr = ''
                    if j2>0:
                        newstr = lines[i][j][:j2] + newstr
                    if j2<len(lines[i][j])-1:
                        newstr = newstr + lines[i][j][j2+1:]
                    lines[i][j] = newstr
                    j2-=1
                j2+=1
##            if lines[i][j]==u"'":
##                lines[i][j]=''
##            else:
##                if lines[i][j]!='':
##                    # eliminate quotation marks from beginning & end
##                    while lines[i][j][0]==u"'":
##                        lines[i][j] = lines[i][j][1:]
##                    while lines[i][j][-1]==u"'":
##                        lines[i][j] = lines[i][j][0:-1]
        i+=1
        
    print('preparing blocks...')

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
        if block != []:
            blocks.append(block)


    if os.path.exists(block_weights_filename)==1:
        fpin = codecs.open(block_weights_filename, 'r', 'utf-8' )
        block_weights = fpin.read().splitlines()
        fpin.close()
        for i in range(len(block_weights)):
            if block_weights[i] != '':
                block_weights[i] = float(block_weights[i])
    else:
        block_weights = []
        for block in blocks:
            block_weights.append( float(1.0 / float( len(blocks) )) )

    if os.path.exists(element_weights_filename)==1:
        fpin = codecs.open(element_weights_filename, 'r', 'utf-8' )
        element_weights = fpin.read().splitlines()
        fpin.close()
        for i in range(len(element_weights)):
            if element_weights[i] != '':
                element_weights[i] = element_weights[i].split()
            for j in range(len(element_weights[i])):
                element_weights[i][j] = float(element_weights[i][j])
    else:
        element_weights = []        
        for i in range(len(blocks)):
            element_weight = []
            for j in range(len(blocks[i])):
                element_weight.append(1)
            element_weights.append(element_weight)

    for i in range(len(blocks)):
        j = 1
        while j < len(blocks[i]):
            if blocks[i][j] in blocks[i][0:j]:
                ind = blocks[i][0:j].index(blocks[i][j])
                element_weights[i][ind] += element_weights[i][j]
                del element_weights[i][j]
                del blocks[i][j]
                j -= 1
            j += 1

    for i in range(len(element_weights)):
        for j in range(len(element_weights[i])):
            element_weights[i][j] = element_weights[i][j]**element_weight_power

    nym_proj_size = {}
    for i in range(len(nyms)):
        nym_proj_size[nyms[i]]=0
    for i in range(len(blocks)):
        # ELIMINATE DUPLICATE ELEMENTS IN BLOCKS # blocks[i] = list(set(blocks[i]))
        for j in range(len(blocks[i])):
            if element_weights == []:
                nym_proj_size[nyms[blocks[i][j]-1]] += 1
            else:
                nym_proj_size[nyms[blocks[i][j]-1]] += element_weights[i][j]

    save_dataset( text_name_all, nyms, nym_proj_size, blocks, block_weights, element_weights )

    return len(blocks)


def prepare_text_chosen( min_proj_size, max_proj_size, text_name_all, text_name_chosen, text_filename, merge_lines, take_with_max_proj_size ):

    print('Preparing text %s ..' % text_name_chosen)

    ( nyms, blocks, block_weights, element_weights, nym_proj_size ) = read_dataset( text_name_all )

    if take_with_max_proj_size>0 and take_with_max_proj_size<=len(nyms):
        nym_proj_size_sorted = sorted(nym_proj_size.values())
        max_proj_size = inf
        min_proj_size = nym_proj_size_sorted[len(nyms)-take_with_max_proj_size]

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
            del block_weights[i]
            if element_weights != []:
                del element_weights[i]
            i-=1
        i+=1    

            
    save_dataset( text_name_chosen, chosen_nyms, nym_proj_size, blocks, block_weights, element_weights )


	
def log_cluster_pair( log_file, min_pe, clusters, nyms, cod ):
    #log_file.write('%5d,%5d : ( ' % (min_pe[1], min_pe[2]) )
    #log_file.write('COD: [ ')
    #for i in range(len(cod)):
    #    log_file.write('%5d ' % cod[i])
    #log_file.write('] : ')
    log_file.write('( ')
    for i in range(len(clusters[min_pe[1]])):
        log_file.write('%s ' % nyms[clusters[min_pe[1]][i]-1])
    log_file.write(')+( ')
    for i in range(len(clusters[min_pe[2]])):
        log_file.write('%s ' % nyms[clusters[min_pe[2]][i]-1])
    log_file.write(') : %.30f\n' % min_pe[0])

    log_file.flush()
    
	
def log_cluster_pair2( log_file, min_pe, clusters, nyms, cod ):
    log_file.write('( ')
    log_file.write('%d ' % len(clusters[min_pe[1]]))
    log_file.write(')+( ')
    log_file.write('%s ' % len(clusters[min_pe[2]]))
    log_file.write(') : %.30f\n' % min_pe[0])

    log_file.flush()
    

def add(dataset, log_file, recurrence_base, addsets):

    # read the nyms and blocks of the dataset
    ( nyms, blocks, block_weights, element_weights, nym_proj_size ) = read_dataset( dataset )

    i=1
    for addset in addsets:
        print("addset:",addset)
        subset = []
        pes = []
        for elm in addset:
            print("adding",elm)
            subset.append(nyms.index(elm)+1)
            pe = projection_entropy( len(nyms), blocks, block_weights, element_weights, subset, recurrence_base )
            print(pe)
            pes.append(pe)
        plt.figure(figsize=(3.5,2.5))
        plt.plot(pes)
        plt.ylabel('entropy')
        plt.xticks([0,1,2,3],addset)
        plt.savefig('test%d-pes.png' % i)
        clf()

        subset = []
        plt.figure(figsize=(4,2.5))
        j=1
        for elm in addset:
            print("adding",elm)
            subset.append(nyms.index(elm)+1)
            cod = cumulative_occurences( len(nyms), blocks, subset )
            print(cod)
            del cod[0]
            plt.bar(range(1,j+1),cod)
            plt.ylabel('cod')
            plt.ylim(0,55)
            plt.xticks([1,2,3,4],[1,2,3,4])
            plt.savefig('test%d-cod%d.png' % (i,j))
            clf()
            j+=1

        i+=1



#~~~~~~~~~~~~~~~~~~~~~~~#
#    REBUS main code    #
#~~~~~~~~~~~~~~~~~~~~~~~#

# put several text names to process several inputs in sequence


# The following alternative settings are for running the examples on this page:
# https://fidaner.wordpress.com/2017/05/18/entropy-as-a-measure-of-irrelevance/

element_weight_power = 0
take_with_max_proj_size = 0

text_names = [ 'pubchem_data' ]
take_with_max_proj_size = 0
recurrence_base = 1
figwidth = 3

addsets = []
addsets.append(['glucose','sucrose','fructose','methanol'])
addsets.append(['methanol','ethanol','propanol','glucose'])



##text_names = [ 'myco-p' ]
##min_proj_sizes = [5]
##max_proj_sizes = [inf]
##recurrence_base = 30

##text_names = [ 'myco-f' ]
##min_proj_sizes = [3]
##max_proj_sizes = [inf]
##recurrence_base = 10

##text_names = [ 'dino' ]
##min_proj_sizes = [1]
##max_proj_sizes = [inf]
##recurrence_base = 1

##text_names = [ 'music' ]
##min_proj_sizes = [20000]
##max_proj_sizes = [ inf]
##recurrence_base = 40

##text_names = [ 'tweets' ]
##min_proj_sizes = [3000,2000,1000]
##max_proj_sizes = [ inf, inf, inf]
##recurrence_base = 1

##text_names = [ 'example' ]
##min_proj_sizes = [1]
##max_proj_sizes = [inf]
##recurrence_base = 1

merge_lines = 0
# 0: put each line in a separate paragraph (separated by single newlines)
# 1: put consequent lines in the same paragraph (separated by double newlines)


# for each of the ranges, minimum and maximum number of paragraphs that the words are allowed to occur in

for text_name in text_names:
    assert os.path.exists(text_name)==0, 'You will need to remove this directory: %s.' % text_name

for text_name in text_names:

    assert os.path.exists(text_name+'.txt')==1, 'Where is my input? (%s.txt ?)' % text_name
    
    os.mkdir(text_name)
    print('Directory created: %s' % text_name)

    text_filename = '%s.txt' % text_name
    block_weights_filename = '%s-block-weights.txt' % text_name
    element_weights_filename = '%s-element-weights.txt' % text_name
    text_name_all = '%s_all' % text_name
    
    outfiles0=[]
    outfiles0.append('nyms_%s.txt' % text_name_all)
    outfiles0.append('nym_proj_size_%s.csv' % text_name_all)
    outfiles0.append('proj_sizes_%s.txt' % text_name_all)
    outfiles0.append('blocks_%s.txt' % text_name_all)
    outfiles0.append('block_weights_%s.txt' % text_name_all)
    outfiles0.append('element_weights_%s.txt' % text_name_all)
    
    print('---')

    try:
            
        block_count = prepare_text_all(text_name_all,text_filename,block_weights_filename,element_weights_filename,merge_lines,element_weight_power)


        dataset = text_name_all

            
        log_file = codecs.open('add_%s.log' % dataset, 'w', 'utf-8' )

        print('---')




        add( dataset, log_file, recurrence_base, addsets )
                





        
##            try:
##                agglomerate_dataset( dataset, log_file, recurrence_base )
##
##                dataset = text_name_chosen
##                outfile = 'ea_'+dataset
##                treefile = 'bifurcations_'+dataset+'.csv'
##
##                draw_dendrogram( plot_title, treefile, dataset, outfile, figwidth )
##
##                for outfile in outfiles:
##                    os.rename(outfile,text_name+'/'+outfile)
##
##            except:
##                print('~ Exception! ~')
##                log_file.close()
##                print("Error:", sys.exc_info())
####                for outfile in outfiles:
####                    if os.path.isfile(outfile):
####                        print 'Removing %s ..' % outfile 
####                        os.remove(outfile)
##                print('~~~')
##                   
        for outfile in outfiles0:
            os.rename(outfile,text_name+'/'+outfile)

        

    except:
        print('~ Exception! ~')
        print("Error:", sys.exc_info())
##        for outfile in outfiles0:
##            if os.path.isfile(outfile):
##                print 'Removing %s ..' % outfile 
##                os.remove(outfile)
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
