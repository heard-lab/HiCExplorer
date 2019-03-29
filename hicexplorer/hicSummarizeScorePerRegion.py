from __future__ import division

import argparse
import numpy as np
import scipy as scp
import pandas as pd
from future.utils import iteritems

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import hicexplorer.HiCMatrix as hm
import hicexplorer.utilities
from .utilities import toString
from .utilities import check_chrom_str_bytes

import logging
log = logging.getLogger(__name__)
from collections import OrderedDict


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(add_help=False,
                                     description='Takes a matrix and a list of regions, and compute the average of matrix scores in each regions')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--matrix', '-m',
                                help='Path of the Hi-C matrix to plot.',
                                required=True)

    parserRequired.add_argument('--outFileName', '-out',
                                help='File name to save the image. ',
                                type=argparse.FileType('w'),
                                required=True)

    parserOpt = parser.add_argument_group('Optional arguments')

    parserOpt.add_argument('--BED',
                            help='Intervals defining the upper triangle from the diagonal in which scores are summarized.',
                            type=argparse.FileType('r'))

    parserOpt.add_argument('--PAIRS',
                            help='Pairs of intervals defining the rectangle/square in which scores are summarized. Only pairs in cis will be considered.',
                            type=argparse.FileType('r'))

    parserOpt.add_argument('--summarizeType',
                           help='Type of operation applied on bins score per regions. Options are: mean, median, sum. Default is mean.',
                           choices=['mean', 'median', 'sum'],
                           default='mean')

    parserOpt.add_argument('--uThr',
                           help='Upper score threshold: only region with a score LOWER than this value will be reported in the output.',
                           type=float,
                           default=float('Inf'))

    parserOpt.add_argument('--lThr',
                           help='Lower score threshold: only region with a score HIGHER than this value will be reported in the output.',
                           type=float,
                           default=float('-Inf'))

    parserOpt.add_argument('--rmDiag',
                           help='Remove the first n diagonals of the matrix. default=1 (ie remove only the diagonal y=x)',
                           type=float,
                           default=1)

    parserOpt.add_argument('--scalingFactor',
                           help='summarized score will be multiplied by this value. Default: 1.',
                           type=float,
                           default=1)
    
    parserOpt.add_argument("--help", "-h", action="help", help="show this help message and exit")

    parserOut = parser.add_argument_group('Output options')

    return parser


def read_bed_per_chrom(fh):
    """
    Reads the given BED file returning
    a dictionary that contains, per each chromosome
    a list of start, end
    """
    interval = {}
    for line in fh:
        if line[0] == "#":
            continue
        fields = line.strip().split()
        if fields[0] not in interval:
            interval[fields[0]] = []
        
        interval[fields[0]].append((int(fields[1]), int(fields[2])))
    
    return interval

def read_pairs_per_chrom(fh):
    """
    Reads the given BED file returning
    a dictionary that contains, per each chromosome
    a list of start, end
    """
    interval = {}
    for line in fh:
        if line[0] == "#":
            continue
        fields = line.strip().split()
        if fields[0] not in interval:
            interval[fields[0]] = []
        if fields[3] == fields[0]:
            interval[fields[0]].append((int(fields[1]), int(fields[2]), int(fields[4]), int(fields[5])))
    
    return interval

def summarize_bed(args):
    matrixFile=args.matrix
    regionsFile=args.BED
    rmDiag=args.rmDiag
    summarizeType=args.summarizeType
    uThr=args.uThr
    lThr=args.lThr
    scalingFactor=args.scalingFactor

    #matrixFile="/data/users/scollomb/Lab_Heard/project_Katia_scHiC/HiC/HicPro/scHiC_data_embryo/EHP_182/cooler/EHP_182_G1.10kb.cool"
    #regionsFile=open("/data/users/scollomb/Lab_Heard/project_Katia_scHiC/HiC/3dnetmod/TADs_merge/3dnetmod.mergeTADs_scoreMin05_XX.mergedOlap70.bed", "r")
    #rmDiag=5
    #summarizeType="sum"
    #uThr=float("Inf")
    #lThr=float("-Inf")
    #scalingFactor=1

    ma = hm.hiCMatrix(matrixFile)
    bed_intervals = read_bed_per_chrom(regionsFile)
    ma.maskBins(ma.nan_bins)
    #ma.matrix.data[np.isnan(ma.matrix.data)] = 0
    
    bin_size = ma.getBinSize()
    ma.maskBins(ma.nan_bins)
    ma.matrix.data = ma.matrix.data
    new_intervals = hicexplorer.utilities.enlarge_bins(ma.cut_intervals)
    ma.setCutIntervals(new_intervals)
    
    chrom_list = list(ma.chrBinBoundaries)
    
    # read and sort bedgraph.
    
    chrom_matrix = OrderedDict()
    chrom_total = {}
    chrom_diagonals = OrderedDict()
    
    seen = {}
    
    center_values = []
    
    chrom_list = check_chrom_str_bytes(bed_intervals, chrom_list)
    
    
    counter = 0
    for chrom in chrom_list:
        if chrom not in bed_intervals:
            continue
        chrom_matrix[chrom] = []
        chrom_total[chrom] = 1
        chrom_diagonals[chrom] = []
        seen[chrom] = set()
        over_1_5 = 0
        empty_mat = 0
        chrom_bin_range = ma.getChrBinRange(toString(chrom))
        
        log.info("processing {}".format(chrom))
        
        for start, end in bed_intervals[chrom]:
            # check all other regions that may interact with the
            # current interval at the given depth range
            # 
            dataStart=start
            dataEnd=end
            ### if the region start/end before/after the last value in the matrix, there is no interval with this cooridnates
            ### if the region is completely out of the data, score =NA
            if ma.interval_trees[chrom][dataStart]==set() and ma.interval_trees[chrom][dataEnd]==set() :
                score=0
            ### if region starts brefor the data but ends inside, move the start to the begining of data
            if dataStart < min(ma.interval_trees[chrom]).begin and ma.interval_trees[chrom][dataEnd]!=set() :
                dataStart=min(ma.interval_trees[chrom]).begin+1
            ### if region ends brefor the data but starts inside, move the end to the end of data
            if dataEnd > max(ma.interval_trees[chrom]).end and ma.interval_trees[chrom][dataStart]!=set() :
                dataEnd = max(ma.interval_trees[chrom]).end-1
            ### average score
            if ma.interval_trees[chrom][dataStart]!=set() and ma.interval_trees[chrom][dataEnd]!=set() :
                bin_id = ma.getRegionBinRange(toString(chrom), dataStart, dataEnd)
                submatrix = scp.sparse.triu(ma.matrix[bin_id[0]:bin_id[1], :][:, bin_id[0]:bin_id[1]], k=rmDiag)
                if summarizeType == 'mean':
                    score= np.mean(submatrix.data) * scalingFactor
                if summarizeType == 'median':
                    score= np.median(submatrix.data) * scalingFactor
                if summarizeType == 'sum':
                    score= np.sum(submatrix.data) * scalingFactor
            if score > uThr:
                    continue
            if score < lThr:
                    continue
            
            if counter==0 :
                scoresArray = [[chrom, start, end, chrom + "_" + str(start) + "_" + str(end), score, "+"]]
            else:
                scoresArray = np.append(scoresArray, [[chrom, start, end, chrom + "_" + str(start) + "_" + str(end), score, "+"]], axis=0)
            counter+=1
        
    scoresArray=pd.DataFrame(scoresArray)
    scoresArray.to_csv(path_or_buf=args.outFileName, sep='\t', index=False, header=False)

def summarize_pairs(args):
    matrixFile=args.matrix
    regionsFile=args.PAIRS
    rmDiag=args.rmDiag
    summarizeType=args.summarizeType
    uThr=args.uThr
    lThr=args.lThr
    scalingFactor=args.scalingFactor

    #matrixFile="/data/users/scollomb/Lab_Heard/project_Katia_scHiC/HiC/HicPro/scHiC_data_embryo/EHP_182/cooler/EHP_182_G1.10kb.cool"
    #regionsFile=open("/data/users/scollomb/Lab_Heard/project_Katia_scHiC/HiC/3dnetmod/TADs_merge/3dnetmod.mergeTADs_scoreMin05_XX.mergedOlap70.pairs", "r")
    #rmDiag=5
    #summarizeType="sum"
    #uThr=float("Inf")
    #lThr=float("-Inf")
    #scalingFactor=1

    ma = hm.hiCMatrix(matrixFile)
    pairs_intervals = read_pairs_per_chrom(regionsFile)
    ma.maskBins(ma.nan_bins)
    #ma.matrix.data[np.isnan(ma.matrix.data)] = 0
    
    bin_size = ma.getBinSize()
    ma.maskBins(ma.nan_bins)
    ma.matrix.data = ma.matrix.data
    new_intervals = hicexplorer.utilities.enlarge_bins(ma.cut_intervals)
    ma.setCutIntervals(new_intervals)
    
    chrom_list = list(ma.chrBinBoundaries)
    
    # read and sort pairsgraph.
    
    chrom_matrix = OrderedDict()
    chrom_total = {}
    chrom_diagonals = OrderedDict()
    
    seen = {}
    
    center_values = []
    
    chrom_list = check_chrom_str_bytes(pairs_intervals, chrom_list)
    
    
    counter = 0
    for chrom in chrom_list:
        if chrom not in pairs_intervals:
            continue
        chrom_matrix[chrom] = []
        chrom_total[chrom] = 1
        chrom_diagonals[chrom] = []
        seen[chrom] = set()
        over_1_5 = 0
        empty_mat = 0
        chrom_bin_range = ma.getChrBinRange(toString(chrom))
        
        log.info("processing {}".format(chrom))
        
        for start1, end1, start2, end2 in pairs_intervals[chrom]:
            # check all other regions that may interact with the
            # current interval at the given depth range
            # 
            dataStart1=start1
            dataEnd1=end1
            ### if the region start/end before/after the last value in the matrix, there is no interval with this cooridnates
            ### if the region is completely out of the data, score =NA
            if ma.interval_trees[chrom][dataStart1]==set() and ma.interval_trees[chrom][dataEnd1]==set() :
                score=0
            ### if region starts brefor the data but ends inside, move the start to the begining of data
            if dataStart1 < min(ma.interval_trees[chrom]).begin and ma.interval_trees[chrom][dataEnd1]!=set() :
                dataStart1=min(ma.interval_trees[chrom]).begin+1
            ### if region ends brefor the data but starts inside, move the end to the end of data
            if dataEnd1 > max(ma.interval_trees[chrom]).end and ma.interval_trees[chrom][dataStart1]!=set() :
                dataEnd1 = max(ma.interval_trees[chrom]).end-1

            dataStart2=start2
            dataEnd2=end2
            ### if the region start/end before/after the last value in the matrix, there is no interval with this cooridnates
            ### if the region is completely out of the data, score =NA
            if ma.interval_trees[chrom][dataStart2]==set() and ma.interval_trees[chrom][dataEnd2]==set() :
                score=0
            ### if region starts brefor the data but ends inside, move the start to the begining of data
            if dataStart2 < min(ma.interval_trees[chrom]).begin and ma.interval_trees[chrom][dataEnd2]!=set() :
                dataStart2=min(ma.interval_trees[chrom]).begin+1
            ### if region ends brefor the data but starts inside, move the end to the end of data
            if dataEnd2 > max(ma.interval_trees[chrom]).end and ma.interval_trees[chrom][dataStart2]!=set() :
                dataEnd2 = max(ma.interval_trees[chrom]).end-1

            ### average score
            if ma.interval_trees[chrom][dataStart1]!=set() and ma.interval_trees[chrom][dataEnd1]!=set() and ma.interval_trees[chrom][dataStart2]!=set() and ma.interval_trees[chrom][dataEnd2]!=set():
                bin_id1 = ma.getRegionBinRange(toString(chrom), dataStart1, dataEnd1)
                bin_id2 = ma.getRegionBinRange(toString(chrom), dataStart2, dataEnd2)
                submatrix = ma.matrix[bin_id1[0]:bin_id1[1], :][:, bin_id2[0]:bin_id2[1]]
                if summarizeType == 'mean':
                    score= np.mean(submatrix.data) * scalingFactor
                if summarizeType == 'median':
                    score= np.median(submatrix.data) * scalingFactor
                if summarizeType == 'sum':
                    score= np.sum(submatrix.data) * scalingFactor
            if score > uThr:
                    continue
            if score < lThr:
                    continue
            
            if counter==0 :
                scoresArray = [[chrom, start1, end1, start2, end2, score]]
            else:
                scoresArray = np.append(scoresArray, [[chrom, start1, end1, start2, end2, score]], axis=0)
            counter+=1
        
    scoresArray=pd.DataFrame(scoresArray)
    scoresArray.to_csv(path_or_buf=args.outFileName, sep='\t', index=False, header=False)

def main(args=None):
    args = parse_arguments().parse_args(args)
    if args.BED : 
        summarize_bed(args)
    if args.PAIRS : 
        summarize_pairs(args)

#def main(args=None):
    #args = parse_arguments().parse_args(args)
    #matrixFile=args.matrix
    #regionsFile=args.BED
    #rmDiag=args.rmDiag
    #summarizeType=args.summarizeType
    #uThr=args.uThr
    #lThr=args.lThr
    #scalingFactor=args.scalingFactor

    ##matrixFile="/data/users/scollomb/Lab_Heard/project_Katia_scHiC/HiC/HicPro/scHiC_data_embryo/EHP_182/cooler/EHP_182_G1.10kb.cool"
    ##regionsFile=open("/data/users/scollomb/Lab_Heard/project_Katia_scHiC/HiC/3dnetmod/TADs_merge/3dnetmod.mergeTADs_scoreMin05_XX.mergedOlap70.bed", "r")
    ##rmDiag=5
    ##summarizeType="sum"
    ##uThr=float("Inf")
    ##lThr=float("-Inf")
    ##scalingFactor=1

    #ma = hm.hiCMatrix(matrixFile)
    #bed_intervals = read_bed_per_chrom(regionsFile)
    #ma.maskBins(ma.nan_bins)
    ##ma.matrix.data[np.isnan(ma.matrix.data)] = 0
    
    #bin_size = ma.getBinSize()
    #ma.maskBins(ma.nan_bins)
    #ma.matrix.data = ma.matrix.data
    #new_intervals = hicexplorer.utilities.enlarge_bins(ma.cut_intervals)
    #ma.setCutIntervals(new_intervals)
    
    #chrom_list = list(ma.chrBinBoundaries)
    
    ## read and sort bedgraph.
    
    #chrom_matrix = OrderedDict()
    #chrom_total = {}
    #chrom_diagonals = OrderedDict()
    
    #seen = {}
    
    #center_values = []
    
    #chrom_list = check_chrom_str_bytes(bed_intervals, chrom_list)
    
    
    #counter = 0
    #for chrom in chrom_list:
        #if chrom not in bed_intervals:
            #continue
        #chrom_matrix[chrom] = []
        #chrom_total[chrom] = 1
        #chrom_diagonals[chrom] = []
        #seen[chrom] = set()
        #over_1_5 = 0
        #empty_mat = 0
        #chrom_bin_range = ma.getChrBinRange(toString(chrom))
        
        #log.info("processing {}".format(chrom))
        
        #for start, end in bed_intervals[chrom]:
            ## check all other regions that may interact with the
            ## current interval at the given depth range
            ## 
            #dataStart=start
            #dataEnd=end
            #### if the region start/end before/after the last value in the matrix, there is no interval with this cooridnates
            #### if the region is completely out of the data, score =NA
            #if ma.interval_trees[chrom][dataStart]==set() and ma.interval_trees[chrom][dataEnd]==set() :
                #score=0
            #### if region starts brefor the data but ends inside, move the start to the begining of data
            #if dataStart < min(ma.interval_trees[chrom]).begin and ma.interval_trees[chrom][dataEnd]!=set() :
                #dataStart=min(ma.interval_trees[chrom]).begin+1
            #### if region ends brefor the data but starts inside, move the end to the end of data
            #if dataEnd > max(ma.interval_trees[chrom]).end and ma.interval_trees[chrom][dataStart]!=set() :
                #dataEnd = max(ma.interval_trees[chrom]).end-1
            #### average score
            #if ma.interval_trees[chrom][dataStart]!=set() and ma.interval_trees[chrom][dataEnd]!=set() :
                #bin_id = ma.getRegionBinRange(toString(chrom), dataStart, dataEnd)
                #submatrix = scp.sparse.triu(ma.matrix[bin_id[0]:bin_id[1], :][:, bin_id[0]:bin_id[1]], k=rmDiag)
                #if summarizeType == 'mean':
                    #score= np.mean(submatrix.data) * scalingFactor
                #if summarizeType == 'median':
                    #score= np.median(submatrix.data) * scalingFactor
                #if summarizeType == 'sum':
                    #score= np.sum(submatrix.data) * scalingFactor
            #if score > uThr:
                    #continue
            #if score < lThr:
                    #continue
            
            #if counter==0 :
                #scoresArray = [[chrom, start, end, chrom + "_" + str(start) + "_" + str(end), score, "+"]]
            #else:
                #scoresArray = np.append(scoresArray, [[chrom, start, end, chrom + "_" + str(start) + "_" + str(end), score, "+"]], axis=0)
            #counter+=1
        
    #scoresArray=pd.DataFrame(scoresArray)
    #scoresArray.to_csv(path_or_buf=args.outFileName, sep='\t', index=False, header=False)


