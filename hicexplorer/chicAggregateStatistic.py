import argparse
import sys
import numpy as np
import hicmatrix.HiCMatrix as hm
from hicexplorer import utilities

from hicexplorer._version import __version__
from .lib import Viewpoint

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

import math
import logging
log = logging.getLogger(__name__)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(add_help=False,
                                     description='Plots the number of interactions around a given reference point in a region.')

    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--interactionFile', '-if',
                                help='path to the interaction files which should be used for plotting',
                                required=True,
                                nargs='+')

   


    parserRequired.add_argument('--range',
                           help='Defines the region upstream and downstream of a reference point which should be included. '
                           'Format is --region upstream downstream',
                           required=True,
                           type=int,
                           nargs=2)
    parserRequired.add_argument('--acceptThreshold', '-at',
                           help='Detect all bins with threshold above this value as significant',
                           type=float,
                           default=1.96,
                           required=True)
    parserOpt = parser.add_argument_group('Optional arguments')

    parserOpt.add_argument('--outFileNameSuffix', '-o',
                            help='File name suffix to save the result.',
                            required=False,
                            default='z_score_merged.bed')
    
    
    parserOpt.add_argument("--mergeBins", "-mb", action='store_true', help="Merge neighboring significant interactions to one. The value is averaged.")


    parserOpt.add_argument("--help", "-h", action="help", help="show this help message and exit")

    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser


def filter_scores(pScoresDictionary, pThreshold, pRange):

    accepted_scores = {}
    for key in pScoresDictionary:
        if key < -pRange[0] or key > pRange[1]:
            continue
        if pScoresDictionary[key][1] >= pThreshold:
            accepted_scores[key] = pScoresDictionary[key]
    return accepted_scores


def merge_neighbors(pScoresDictionary, pMergeThreshold = 1000):

    key_list = list(pScoresDictionary.keys())
    # log.debug('key_list {}'.format(key_list))

    # [[start, ..., end]]
    neighborhoods = []
    neighborhoods.append([key_list[0], key_list[0]])
    scores = [pScoresDictionary[key_list[0]]]
    
    for key in key_list[1:]:
        
        if np.absolute(key - neighborhoods[-1][0]) <= pMergeThreshold or np.absolute(key - neighborhoods[-1][1]) <= pMergeThreshold:
            neighborhoods[-1][-1] = key
            scores[-1] += pScoresDictionary[key]
        else:
            neighborhoods.append([key, key])
            scores.append(pScoresDictionary[key])

    for i in range(len(neighborhoods)):
        scores[i] /= len(neighborhoods[i])

    return neighborhoods, scores

def write (pOutFileName, pNeighborhoods, pScores, pInteractionLines, pThreshold):

    with open(pOutFileName, 'w') as file:
        file.write('# Significant regions with rbz-score higher as ' + str(pThreshold) + '\n')
        file.write('#ChrViewpoint\tStart\tEnd\tGene\tChrInteraction\tStart\tEnd\tRel Inter viewpoint\trbz-score viewpoint\tRaw viewpoint\tRel Inter target\trbz-score target\tRaw target')
        file.write('\n')
        for i in range(len(pNeighborhoods)):
            start = pNeighborhoods[i][0]
            end = pNeighborhoods[i][1]
            pInteractionLines[start]
            pInteractionLines[end][-3:]
            new_end = pInteractionLines[end][6]

            pInteractionLines[0][-3:]

            new_line = '\t'.join(pInteractionLines[start][:6])
            new_line += '\t' + new_end
            # log.debug('pInteractionLines[0][8:] {}'.format(pInteractionLines[0][8:]))
            # "\t".join(format(x, ".5f") for x in pInteractionLines[0][8:])
            # new_line += '\t' + '\t'.join(pInteractionLines[0][8:])
            new_line += '\t' + '\t'.join(format(float(x), "10.5f") for x in pInteractionLines[0][8:])

            new_line += '\t' + format(pScores[i][0], '10.5f') + '\t' + format(pScores[i][1], '10.5f') + '\t' + format(pScores[i][2], '10.5f')

            new_line += '\n'
            file.write(new_line)


def main(args=None):
    args = parse_arguments().parse_args(args)
    log.debug('muh')
    viewpointObj = Viewpoint()
    background_data = None
    # log.debug("sdfghjkoihgfghjkl;kjhfvghjkl")
    relative_interaction = False
    rbz_score = True
    # read all interaction files.
    for interactionFile in args.interactionFile:
        header, interaction_data, interaction_file_data = viewpointObj.readInteractionFileForAggregateStatistics(interactionFile)
        # log.debug('interaction_data {}'.format(interaction_data))
        accepted_scores = filter_scores(interaction_data, args.acceptThreshold, args.range)
        # log.debug('accepted_scores {}'.format(accepted_scores))
        merged_neighborhood = merge_neighbors(accepted_scores)
        # log.debug('accepted_scores {}'.format(accepted_scores))
        # log.debug('merge_neighbored {}'.format(merged_neighborhood))
        outFileName = interactionFile.split('.')[0] + '_' + args.outFileNameSuffix
        write(outFileName, merged_neighborhood[0], merged_neighborhood[1], interaction_file_data, args.acceptThreshold)
        # log.debug('header {}'.format(header))
        # log.debug('interaction_data {}'.format(interaction_data))
        # log.debug('z_score {}'.format(z_score))


    