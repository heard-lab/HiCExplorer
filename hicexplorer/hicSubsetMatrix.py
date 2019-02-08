from __future__ import division
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import argparse
from hicmatrix import HiCMatrix as hm
from hicexplorer._version import __version__
from hicmatrix.HiCMatrix import check_cooler
import numpy as np
import logging
log = logging.getLogger(__name__)

def parse_arguments(args=None):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        description="""
        Subset a Hi-C matrix to only retain the portions of the matrix that are interacting between two bed files.
""")

    parserRequired = parser.add_argument_group('Required arguments')

    parserRequired.add_argument('--matrix', '-m',
                                help='The matrix to adjust. '
                                'HiCExplorer supports the following file formats: .h5 (native HiCExplorer format) '
                                'and .cool.',
                                required=True,
                                metavar='matrix.h5')
    parserRequired.add_argument('--outFileName', '-o',
                                help='File name to save the subsetted matrix.',
                                required=True,
                                metavar='subsetted_matrix.h5')
    parserRequired.add_argument('--BED1', '-b1',
                           help='First BED file which stores a list of regions.',
                           required=True,
                           metavar='regions1.bed')
    parserRequired.add_argument('--BED2', '-b2',
                           help='Second BED file which stores a list of regions.',
                           required=True,
                           metavar='regions2.bed')

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--help', '-h', action='help', help='show this help message and exit')
    parserOpt.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser

def main(args=None):

    args = parse_arguments().parse_args(args)
    hic_ma = hm.hiCMatrix(args.matrix)
    genomic_regions1 = []
    with open(args.BED1, 'r') as file:
		for line in file.readlines():
				_line = line.strip().split('\t')
				if len(line) == 0:
						continue
				if len(_line) == 3:
						chrom, start, end = _line[0], _line[1], int(_line[2]) - 1

				genomic_regions1.append((chrom, start, end))

    genomic_regions2 = []
    with open(args.BED2, 'r') as file:
		for line in file.readlines():
				_line = line.strip().split('\t')
				if len(line) == 0:
						continue
				if len(_line) == 3:
						chrom, start, end = _line[0], _line[1], int(_line[2]) - 1

				genomic_regions2.append((chrom, start, end))


    matrix_indices_regions1 = []
    for region in genomic_regions1:
		_regionBinRange = hic_ma.getRegionBinRange(region[0], region[1], region[2])
		if _regionBinRange is not None:
		    start, end = _regionBinRange
		    matrix_indices_regions1.extend(list(range(start, end)))

    matrix_indices_regions2 = []
    for region in genomic_regions2:
		_regionBinRange = hic_ma.getRegionBinRange(region[0], region[1], region[2])
		if _regionBinRange is not None:
		    start, end = _regionBinRange
		    matrix_indices_regions2.extend(list(range(start, end)))

    x_values_submatrix = matrix_indices_regions1
    y_values_submatrix = matrix_indices_regions2
    instances, features = hic_ma.matrix.nonzero()
    mask_x = np.isin(instances, x_values_submatrix)
    mask_y = np.isin(features, y_values_submatrix)
    mask = np.logical_and(mask_x, mask_y)
    mask = np.logical_not(mask)
    hic_ma.matrix.data[mask] = 0
    hic_ma.matrix.eliminate_zeros()
    hic_ma.save(args.outFileName)
