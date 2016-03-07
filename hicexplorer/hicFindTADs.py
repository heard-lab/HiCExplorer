#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import division
import sys
import argparse
from hicexplorer import HiCMatrix as hm
from hicexplorer.utilities import enlarge_bins
from scipy import sparse
import numpy as np
import multiprocessing


def parse_arguments(args=None):
    """
    get command line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Uses a measure called TAD score to identify the separation between '
left and right regions for a given position. This is done for a '
running window of different sizes. Then, TADs are called as those '
positions having a local minimum.

To actually find the TADs, the program  needs to compute first the
TAD scores at different window sizes. Then, the results of that computation
are used to call the TADs. An simple example usage is:

$ hicFindTads TAD_score -m hic_matrix.npz -o TAD_score.txt
$ hicFindTads find_TADs -f TAD_score.txt --outPrefix TADs

For detailed help:

 hicFindTADs TAD_score -h
  or
 hicFindTADs find_TADs -h

""")

    subparsers = parser.add_subparsers(dest='command')

    tad_score_subparser = subparsers.add_parser('TAD_score',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # define the arguments
    tad_score_subparser.add_argument('--matrix', '-m',
                                     help='Hi-C matrix to use for the computations',
                                     metavar='.npz file format',
                                     required=True)

    tad_score_subparser.add_argument('--outFileName', '-o',
                                     help='File name to store the computation of the TAD score',
                                     required=True)

    tad_score_subparser.add_argument('--minDepth',
                                     help='window length to be considered left and right '
                                          'of the cut point in bp. This number should be at least 2 times '
                                          'as large as the bin size of the Hi-C matrix.',
                                     metavar='INT bp',
                                     type=int,
                                     default=20000)

    tad_score_subparser.add_argument('--maxDepth',
                                     help='window length to be considered left and right '
                                          'of the cut point in bp. This number should around 6 times '
                                          'as large as the bin size of the Hi-C matrix.',
                                     metavar='INT bp',
                                     type=int,
                                     default=60000)

    tad_score_subparser.add_argument('--step',
                                     help='step size when moving from --minDepth to --maxDepth',
                                     metavar='INT bp',
                                     type=int,
                                     default=10000
                                     )

    tad_score_subparser.add_argument('--useLogValues',
                                     help='If set, the log of the matrix values are'
                                          'used.',
                                     action='store_true')


    tad_score_subparser.add_argument('--numberOfProcessors',  '-p',
                                     help='Number of processors to use ',
                                     type=int,
                                     default=1)


    find_tads_subparser = subparsers.add_parser('find_TADs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    find_tads_subparser.add_argument('--tadScoreFile', '-f',
                                     help='file containing the TAD scores (generated by running hicFindTADs TAD_score)',
                                     required=True)

    find_tads_subparser.add_argument('--lookahead',
                                     help='number of bins ahead to look for before deciding '
                                          'if a local minimum is a boundary.'
                                          'points.',
                                     type=int,
                                     default=2
                                     )

    find_tads_subparser.add_argument('--delta',
                                     help='minimum difference between a peak and following'
                                          'points.',
                                     type=float,
                                     default=0.001
                                     )

    find_tads_subparser.add_argument('--maxThreshold',
                                     help='only call boundaries that have an score below this threshold',
                                     type=float,
                                     default=0.3
                                     )

    find_tads_subparser.add_argument('--outPrefix',
                                     help='File prefix to save the resulting files: boundary positions, '
                                          'bedgraph matrix containing the multi-scale TAD scores, '
                                          'the BED files for the TAD clusters and the linkage BED file that '
                                          'can be used with hicPlotTADs.',
                                     required=True)

    return parser


def get_cut_weight(matrix, cut, depth):
    """
    Get inter cluster edges sum.
    Computes the sum of the counts
    between the left and right regions of a cut

    >>> matrix = np.array([
    ... [ 0,  0,  0,  0,  0],
    ... [10,  0,  0,  0,  0],
    ... [ 5, 15,  0,  0,  0],
    ... [ 3,  5,  7,  0,  0],
    ... [ 0,  1,  3,  1,  0]])

    Test a cut at position 2, depth 2.
    The values in the matrix correspond
    to:
          [[ 5, 15],
           [ 3,  5]]
    >>> get_cut_weight(matrix, 2, 2)
    28

    For the next test the expected
    submatrix is [[10],
                  [5]]
    >>> get_cut_weight(matrix, 1, 2)
    15
    >>> get_cut_weight(matrix, 4, 2)
    4
    >>> get_cut_weight(matrix, 5, 2)
    0
    """
    # the range [start:i] should have running window
    # length elements (i is excluded from the range)
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    # the idea is to evaluate the interactions
    # between the upstream neighbors with the
    # down stream neighbors. In other words
    # the inter-domain interactions
    return matrix[cut:end, :][:, start:cut].sum()


def get_min_volume(matrix, cut, depth):
    """
    The volume is the weight of the edges
    from a region to all other.

    In this case what I compute is
    a submatrix that goes from
    cut - depth to cut + depth
    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    left_region = matrix[start:end, :][:, start:cut].sum()
    right_region = matrix[cut:end, :][:, start:end].sum()

    return min(left_region, right_region)

def get_conductance(matrix, cut, depth):
    """
    Computes the conductance measure for
    a matrix at a given cut position and
    up to a given depth.

    If int = inter-domain counts

    then the conductance is defined as

    conductance = int / min(int + left counts, int + right counts)

    The matrix has to be lower or uppper to avoid
    double counting

    In the following example the conductance is to be
    computed for a cut at index position 2 (between column 2 and 3)
    >>> matrix = np.array([
    ... [ 0,  0,  0,  0,  0],
    ... [10,  0,  0,  0,  0],
    ... [ 5, 15,  0,  0,  0],
    ... [ 3,  5,  7,  0,  0],
    ... [ 0,  1,  3,  1,  0]])

    The lower left intra counts are [0,10,0]',
    The lower right intra counts are [0, 7 0],
    The inter counts are:
          [[ 5, 15],
           [ 3,  5]], sum = 28

    The min of left and right is min(28+7, 28+10) = 35
    >>> res = get_conductance(matrix, 2, 2)
    >>> res == 28.0 / 35
    True
    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    inter_edges = get_cut_weight(matrix, cut, depth)
    edges_left = matrix[start:cut, :][:, start:cut].sum()
    edges_right = matrix[cut:end, :][:, cut:end].sum()

    return float(inter_edges) / min(edges_left + inter_edges, edges_right + inter_edges)


def get_coverage_norm(matrix, cut, depth):
    """
    Similar to the coverage but instead of dividing
    by total count, it divides but the sum of left and right counts.
    This measure has the advantage of being closer to the 0-1 range.
    Where 0 occurs when there are no contacts between left and right and
    1 occurs when the intra-counts are equal to the sum of the left and
    right counts.

    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    inter_edges = get_cut_weight(matrix, cut, depth)
    edges_left = matrix[start:cut, :][:, start:cut].sum()
    edges_right = matrix[cut:end, :][:, cut:end].sum()
    return float(inter_edges) / sum([edges_left, edges_right])

def get_coverage(matrix, cut, depth):
    """
    The coverage is defined as the
    intra-domain edges / all edges

    It is only computed for a small running window
    of length 2*depth

    The matrix has to be lower or upper to avoid
    double counting
    """
    start = max(0, cut - depth)
    # same for range [i+1:end] (i is excluded from the range)
    end = min(matrix.shape[0], cut + depth)

    cut_weight = get_cut_weight(matrix, cut, depth)
    total_edges = matrix[start:end, :][:, start:end].sum()
    return cut_weight / total_edges


def compute_matrix_wrapper(args):
    return compute_matrix(*args)


def get_incremental_step_size(min_win_size, max_win_size, start_step_len):
    """
    generates a list of incremental windows sizes (measured in bins)

    :param min_win_size: starting window size
    :param max_win_size: end window size
    :param start_step_len: start length
    :return: incremental_step list of bin lengths
    """
    incremental_step = []
    step = -1
    while 1:
        step += 1
        inc_step = min_win_size + int(start_step_len * (step**1.5))
        if step > 1 and inc_step == incremental_step[-1]:
            continue
        if inc_step > max_win_size:
            break
        incremental_step.append(inc_step)
    return incremental_step


def compute_matrix(bins_list, min_win_size=8, max_win_size=50, step_len=2, outfile=None):
    """
    Iterates over the Hi-C matrix computing at each bin
    interface the conductance at different window lengths
    :param hic_ma: Hi-C matrix object from HiCMatrix
    :param outfile: String, path of a file to save the conductance
                matrix in *bedgraph matrix* format
    :return: (chrom, start, end, matrix)
    """
    global hic_ma
    positions_array = []
    cond_matrix = []
    chrom, start, end, __ = hic_ma.cut_intervals[0]
    for cut in bins_list:


        chrom, chr_start, chr_end, _ = hic_ma.cut_intervals[cut]

        # get conductance
        # for multiple window lengths at a time
        incremental_step = get_incremental_step_size(min_win_size, max_win_size, step_len)
        mult_matrix = [get_coverage_norm(hic_ma.matrix, cut, x) for x in incremental_step]
        #mult_matrix = [get_coverage(hic_ma.matrix, cut, x) for x in incremental_step]

        """
        mult_matrix = [get_coverage(hic_ma.matrix, cut, x)
                       for x in range(min_win_size, max_win_size, step_len)]
        """

        cond_matrix.append(mult_matrix)

        positions_array.append((chrom, chr_start, chr_end))

    chrom, chr_start, chr_end = zip(*positions_array)
    cond_matrix = np.vstack(cond_matrix)

    return chrom, chr_start, chr_end, cond_matrix


def peakdetect(y_axis, x_axis=None, lookahead=3, delta=0):
    """
    Based on the MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximum and minimum in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximum and minimum respectively

    keyword arguments:
    :param: y_axis -- A list containig the signal over which to find peaks
    :param: x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    :param: lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak
    :param: delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.


    :return: -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false

    # check input data
    if x_axis is None:
        x_axis = np.arange(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise (ValueError,
                'Input vectors y_axis and x_axis must have same length')

    # store data length for later use
    length = len(y_axis)

    # perform some checks
    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError, "delta must be a positive number"

    # maximum and minimum candidates are temporarily stored in
    # mx and mn respectively
    min_y, max_y = np.Inf, -np.Inf
    max_pos, min_pos = None, None
    search_for = None
    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > max_y:
            max_y = y
            max_pos = x
        if y < min_y:
            min_y = y
            min_pos = x

        # look for max
        if y < max_y - delta and max_y != np.Inf and search_for != 'min':
            # Maximum peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < max_y:
                max_peaks.append([max_pos, max_y])
                dump.append(True)
                # set algorithm to only find minimum now
                max_y = y
                min_y = y
                search_for = 'min'
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue

        # look for min
        if y > min_y + delta and min_y != -np.Inf and search_for != 'max':
            # Minimum peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > min_y:
                min_peaks.append([min_pos, min_y])
                dump.append(False)
                # set algorithm to only find maximum now
                min_y = y
                max_y = y
                search_for = 'max'
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def find_consensus_minima(matrix, lookahead=3, delta=0, max_threshold=0.3):
    """
    Finds the minimum over the average values per column
    :param matrix:
    :param max_threshold maximum value allowed for a local minimum to be called
    :return:
    """


    # use the matrix transpose such that each row
    # represents the conductance at each genomic
    # position

    _max, _min= peakdetect(matrix.mean(axis=1), lookahead=lookahead, delta=delta)
    # filter all mimimum that are over a value of max_threshold
    min_indices = [idx for idx, value in _min if value <= max_threshold]
    return np.unique(min_indices)


def hierarchical_clustering(boundary_list, clusters_cutoff=[]):
    """
    :param boundary_list: is a list of tuples each containing
    the location of a boundary. The order should be sorted
    and contain the following values:
        (chrom, start, value)
    :param clusters_cutoff: List of values to separate clusters. The
        clusters found at those value thresholds are returned.


    :return: z_value, clusters

    For z_value, the format used is similar as the scipy.cluster.hierarchy.linkage() function
    which is described as follows:

    A 4 by :math:`(n-1)` matrix ``z_value`` is returned. At the
    :math:`i`-th iteration, clusters with indices ``z_value[i, 0]`` and
    ``z_value[i, 1]`` are combined to form cluster :math:`n + i`. A
    cluster with an index less than :math:`n` corresponds to one of
    the :math:`n` original observations. The distance between
    clusters ``z_value[i, 0]`` and ``z_value[i, 1]`` is given by ``z_value[i, 2]``. The
    fourth value ``z_value[i, 3]`` represents the number of original
    observations in the newly formed cluster.

    The difference is that instead of a 4 times n-1 array, a
    6 times n-1 array is returned. Where positions 4, and 5
    correspond to the genomic coordinates of ``z_value[i, 0]`` and ``z_value[i, 1]``

    """
    # run the hierarchical clustering per chromosome
    if clusters_cutoff:
        # sort in reverse order
        clusters_cutoff = np.sort(np.unique(clusters_cutoff))[::-1]

    chrom, start, value = zip(*boundary_list)

    unique_chr, indices = np.unique(chrom, return_index=True)
    indices = indices[1:]  # the first element is not needed
    start_per_chr = np.split(start, indices)
    value_per_chr = np.split(value, indices)
    z_value = {}

    def get_domain_positions(boundary_position):
        """
        returns for each boundary a start,end position
        corresponding to each TAD
        :param boundary_position: list of boundary chromosomal positions
        :return: list of (start, end) tuples.
        """
        start_ = None
        domain_list = []
        for position in boundary_position:
            if start_ is None:
                start_ = position
                continue
            domain_list.append((start_, position))
            start_ = position

        return domain_list

    def find_in_clusters(clusters_, search_id):
        """
        Given a list of clusters (each cluster defined as as set,
        the function returns the position in which an id is found
        :param clusters_:
        :param search_id:
        :return:
        """
        for set_idx, set_of_ids in enumerate(clusters_):
            if search_id in set_of_ids:
                return set_idx

    def cluster_to_regions(clusters_, chrom_name):
        """
        Transforms a list of sets of ids from the hierarchical
        clustering to genomic positions
        :param clusters_: cluster ids
        :param chrom_name: chromosome name
        :return: list of tuples with (chrom_name, start, end)

        Example:

        clusters = [set(1,2,3), set(4,5,10)]

        """
        start_list = []
        end_list = []
        for set_ in clusters_:
            if len(set_) == 0:
                continue

            # the ids in the sets are created in such a
            # that the min id is the one with the smaller start position
            start_list.append(domains[min(set_)][0])
            end_list.append(domains[max(set_)][1])

        start_list = np.array(start_list)
        end_list = np.array(end_list)
        order = np.argsort(start_list)

        return zip([chrom_name] * len(order), start_list[order], end_list[order])

    return_clusters = {} # collects the genomic positions of the clusters per chromosome
                         # The values are a list, one for each cutoff.
    for chrom_idx, chrom_name in enumerate(unique_chr):
        z_value[chrom_name] = []
        return_clusters[chrom_name] = []
        clust_cutoff = clusters_cutoff[:]
        domains = get_domain_positions(start_per_chr[chrom_idx])
        clusters = [{x} for x in range(len(domains))]

        # initialize the cluster_x with the genomic position of domain centers
        cluster_x = [int(d_start + float(d_end - d_start) / 2) for d_start, d_end in domains]
        # number of domains should be equal to the number of values minus 1
        assert len(domains) == len(value_per_chr[chrom_idx]) - 1, "error"

        """
        domain:id
             0            1               2            3
         |---------|---------------|----------------|----|
        values:id
         0         1               3                3    4
        values id after removing flanks
                   0               1                2
         """
        values = value_per_chr[chrom_idx][1:-1] # remove flanking values that do not join TADs
        start_trimmed = start_per_chr[chrom_idx][1:-1]
        # from highest to lowest merge neighboring domains
        order = np.argsort(values)[::-1]
        for idx, order_idx in enumerate(order):
            if len(clust_cutoff) and idx + 1 < len(order) and \
                    values[order_idx] >= clust_cutoff[0] > values[order[idx + 1]]:
                clust_cutoff = clust_cutoff[1:] # remove first element
                return_clusters[chrom_name].append(cluster_to_regions(clusters, chrom_name))
            # merge domains order_idx - 1 and order_idx
            left = find_in_clusters(clusters, order_idx)
            right = find_in_clusters(clusters, order_idx + 1)
            z_value[chrom_name].append((left, right, values[order_idx],
                                  len(clusters[left]) + len(clusters[right]),
                                  cluster_x[left], cluster_x[right]))

            # set as new cluster position the center between the two merged
            # clusters
            gen_dist = int(float(abs(cluster_x[left] - cluster_x[right]))/2)
            cluster_x.append(min(cluster_x[left], cluster_x[right]) + gen_dist)

            clusters.append(clusters[left].union(clusters[right]))
            clusters[left] = set()
            clusters[right] = set()

    # convert return_clusters from a per chromosome dictionary to
    # a per cut_off dictionary merging all chromosomes in to one list.
    ret_ = {}  # dictionary to hold the clusters per cutoff. The key of
               # each item is the str(cutoff)

    for idx, cutoff in enumerate(clusters_cutoff):
        cutoff = str(cutoff)
        ret_[cutoff] = []
        for chr_name in return_clusters:
            try:
                ret_[cutoff].extend(return_clusters[chr_name][idx])
            except IndexError:
                pass

    return z_value, ret_


def save_linkage(Z, file_name):
    """

    :param Z: Z has a format similar to the scipy.cluster.linkage matrix (see function
                hierarchical_clustering).
    :param file_name: File name to save the results
    :return: None
    """

    try:
        file_h = open(file_name, 'w')
    except IOError:
        sys.stderr.write("Can't save linkage file:\n{}".format(file_name))
        return

    count = 0
    for chrom, values in Z.iteritems():
        for id_a, id_b, distance, num_clusters, pos_a, pos_b in values:
            count += 1
            file_h.write('{}\t{}\t{}\tclust_{}'
                         '\t{}\t.\t{}\t{}\t{}\n'.format(chrom,
                                                        int(pos_a),
                                                        int(pos_b),
                                                        count,
                                                        distance,
                                                        id_a, id_b,
                                                        num_clusters))


def get_domains(boundary_list):
    """
    returns for each boundary a chrom, start,end position
    corresponding to each TAD
    :param boundary_position: list of boundary chromosomal positions
    :return: list of (chrom, start, end, value) tuples.
    """
    prev_start = None
    prev_chrom = boundary_list[0][0]
    domain_list = []
    for chrom, start, value in boundary_list:
        if start is None:
            prev_start = start
            prev_chrom = chrom
            continue
        if prev_chrom != chrom:
            prev_chrom = chrom
            prev_start = None
            continue
        domain_list.append((chrom, prev_start, start, value))
        prev_start = start
        prev_chrom = chrom

    return domain_list


def save_bedgraph_matrix(outfile, chrom, chr_start, chr_end, score_matrix):
    """
    Save matrix as chrom, start, end ,row, values separated by tab
    I call this a bedgraph matrix (bm)

    :param outfile: string file name
    :param chrom: list of chrom names
    :param chr_start: list of start positions
    :param chr_end: list of end positions
    :param score_matrix: list of lists
    :return: None
    """

    with open(outfile, 'w') as f:
        for idx in range(len(chrom)):
            matrix_values = "\t".join(
                    np.char.mod('%f', score_matrix[idx, :]))
            f.write("{}\t{}\t{}\t{}\n".format(chrom[idx], chr_start[idx],
                                              chr_end[idx], matrix_values))


def save_clusters(clusters, file_prefix):
    """

    :param clusters: is a dictionary whose key is the cut of used to create it.
                     the value is a list of tuples, each representing
                      a genomec interval as ('chr', start, end).
    :param file_prefix: file prefix to save the resulting bed files
    :return: list of file names created
    """
    for cutoff, intervals in clusters.iteritems():
        fileh = open("{}_{}.bed".format(file_prefix, cutoff), 'w')
        for chrom, start, end in intervals:
            fileh.write("{}\t{}\t{}\t.\t0\t.\n".format(chrom, start, end))


def save_domains_and_boundaries(chrom, chr_start, chr_end, matrix, min_idx, args):


    prev_chrom = chrom[0]
    chrom_sizes = {}
    chr_bin_range = []
    for idx, chr_name in enumerate(chrom):
        if prev_chrom != chr_name:
            chrom_sizes[prev_chrom] = chr_end[idx-1]
            chr_bin_range.append(idx)
        prev_chrom = chr_name

    chrom_sizes[chr_name] = chr_end[idx]
    chr_bin_range.append(idx)

    new_min_idx = [0]
    for idx in min_idx:
        # for each chromosome, add position 0 and end position as boundaries
        # 'chr_bin_range'contains the indices values where the chromosome name changes in the list

        if len(chr_bin_range) and idx > chr_bin_range[0]:
            new_min_idx.append(chr_bin_range[0] - 1)
            new_min_idx.append(chr_bin_range[0])
            chr_bin_range = chr_bin_range[1:]
        new_min_idx.append(idx)

    new_min_idx.append(len(chr_start - 1))

    # get min_idx per chromosome
    chrom_boundary = chrom[min_idx]
    boundaries = np.array([chr_start[idx] for idx in min_idx])
    mean_mat_all = matrix.mean(axis=1)
    mean_mat = mean_mat_all[min_idx]
    count = 0
    with open(args.outPrefix + '_boundaries.bed', 'w') as file_boundaries, open(args.outPrefix + '_domains.bed', 'w') as file_domains:
        for idx in range(len(boundaries)):
            # save boundaries at 1bp position
            file_boundaries.write("{}\t{}\t{}\tmin\t{}\t.\n".format(chrom_boundary[idx], boundaries[idx],
                                                                    boundaries[idx] + 1,
                                                                    mean_mat[idx]))

            start = boundaries[idx]
            if start == chrom_sizes[chrom[idx]]:
                continue
            if idx + 1 == len(boundaries) or boundaries[idx + 1] < start:
                end = chrom_sizes[chrom_boundary[idx]]
            else:
                end = boundaries[idx + 1]

            # 2. save domain intervals
            if count % 2 == 0:
                rgb = '51,160,44'
            else:
                rgb = '31,120,180'

            file_domains.write("{0}\t{1}\t{2}\tID_{3}\t{4}\t."
                        "\t{1}\t{2}\t{5}\n".format(chrom_boundary[idx],
                                                   start,
                                                   end,
                                                   min_idx[idx],
                                                   mean_mat[idx],
                                                   rgb))

            count += 1

    # save track with mean values in bedgraph format
    with open(args.outPrefix + '_score.bg', 'w') as tad_score:
        for idx in range(len(chrom)):
            tad_score.write("{}\t{}\t{}\t{}\n".format(chrom[idx], chr_start[idx], chr_end[idx], mean_mat_all[idx]))


def compute_spectra_matrix(args):
    if args.maxDepth <= args.minDepth:
        exit("Please check that maxDepth is larger than minDepth.")

    global hic_ma
    hic_ma = hm.hiCMatrix(args.matrix)
    # remove self counts
    hic_ma.diagflat(value=0)
    sys.stderr.write('removing diagonal values\n')
    if args.useLogValues is True:
        # use log values for the computations
        hic_ma.matrix.data = np.log(hic_ma.matrix.data)
        sys.stderr.write('using log matrix values\n')

    # mask bins without any information
    hic_ma.maskBins(hic_ma.nan_bins)
    orig_intervals = hic_ma.cut_intervals

    # extend remaining bins to remove gaps in
    # the matrix
    new_intervals = enlarge_bins(hic_ma.cut_intervals)

    # rebuilt bin positions if necessary
    if new_intervals != orig_intervals:
        hic_ma.interval_trees, hic_ma.chrBinBoundaries = \
            hic_ma.intervalListToIntervalTree(new_intervals)

    if args.minDepth % hic_ma.getBinSize() != 0:
        sys.stderr.write('Warning. specified depth is not multiple of the '
             'hi-c matrix bin size ({})\n'.format(hic_ma.getBinSize()))
    if args.step % hic_ma.getBinSize() != 0:
        sys.stderr.write('Warning. Epecified step is not multiple of the '
                         'hi-c matrix bin size ({})\n'.format(hic_ma.getBinSize()))

    binsize = hic_ma.getBinSize()

    min_depth_in_bins = int(args.minDepth / binsize)
    max_depth_in_bins = int(args.maxDepth / binsize)
    step_in_bins = int(args.step / binsize)
    if step_in_bins == 0:
        exit("Please select a step size larger than {}".format(binsize))

    step_len = binsize * step_in_bins
    min_win_size = binsize * min_depth_in_bins
    max_win_size = binsize * max_depth_in_bins
    incremental_step = []
    step = -1
    while 1:
        step += 1
        inc_step = min_win_size + (step_len * int(step**1.5))
        if step > 1 and inc_step == incremental_step[-1]:
            continue
        if inc_step > max_win_size:
            break
        incremental_step.append(inc_step)

    print incremental_step

    sys.stderr.write("computing spectrum for window sizes between {} ({} bp)"
                     "and {} ({} bp) at the following window sizes {} {}\n".format(min_depth_in_bins,
                                                                   binsize * min_depth_in_bins,
                                                                   max_depth_in_bins,
                                                                   binsize * max_depth_in_bins,
                                                                   step_in_bins, incremental_step))
    if min_depth_in_bins <= 1:
        sys.stderr.write('ERROR\nminDepth length too small. Use a value that is at least'
                         'twice as large as the bin size which is: {}\n'.format(binsize))
        exit()

    if max_depth_in_bins <= 1:
        sys.stderr.write('ERROR\nmaxDepth length too small. Use a value that is larger '
                         'than the bin size which is: {}\n'.format(binsize))
        exit()

    # work only with the lower matrix
    # and remove all pixels that are beyond
    # 2 * max_depth_in_bis which are not required
    # (this is done by subtracting a second sparse matrix
    # that contains only the lower matrix that wants to be removed.
    limit = -2 * max_depth_in_bins
    hic_ma.matrix = sparse.tril(hic_ma.matrix, k=0, format='csr') - sparse.tril(hic_ma.matrix, k=limit, format='csr')
    hic_ma.matrix.eliminate_zeros()

    num_processors = args.numberOfProcessors
    pool = multiprocessing.Pool(num_processors)
    func = compute_matrix_wrapper
    TASKS = []
    bins_to_consider = []
    for chrom in hic_ma.chrBinBoundaries.keys():
        bins_to_consider.extend(range(*hic_ma.chrBinBoundaries[chrom]))

    for idx_array in np.array_split(bins_to_consider, num_processors):
        TASKS.append((idx_array, min_depth_in_bins, max_depth_in_bins, step_in_bins))

    if num_processors > 1:
        sys.stderr.write("Using {} processors\n".format(num_processors))
        res = pool.map_async(func, TASKS).get(9999999)
    else:
        res = map(func, TASKS)

    chrom = []
    chr_start = []
    chr_end = []
    matrix = []
    for _chrom, _chr_start, _chr_end, _matrix in res:
        chrom.extend(_chrom)
        chr_start.extend(_chr_start)
        chr_end.extend(_chr_end)
        matrix.append(_matrix)

    matrix = np.vstack(matrix)
    return np.array(chrom), np.array(chr_start), np.array(chr_end), matrix


def load_spectrum_matrix(file):
    # load spectrum matrix:
    matrix = []
    chrom_list = []
    start_list = []
    end_list = []
    with open(file, 'r') as fh:
        for line in fh:
            fields = line.strip().split('\t')
            chrom, start, end = fields[0:3]
            chrom_list.append(chrom)
            start_list.append(int(start))
            end_list.append(int(end))
            matrix.append(map(float, fields[3:]))

    matrix = np.vstack(matrix)
    chrom = np.array(chrom_list)
    start = np.array(start_list)
    end = np.array(end_list)
    return chrom, start, end, matrix


def main(args=None):

    args = parse_arguments().parse_args(args)
    if args.command == 'TAD_score':
        chrom, chr_start, chr_end, matrix = compute_spectra_matrix(args)
        save_bedgraph_matrix(args.outFileName, chrom, chr_start, chr_end, matrix)
        return

    chrom, chr_start, chr_end, matrix = load_spectrum_matrix(args.tadScoreFile)

    min_idx = find_consensus_minima(matrix, lookahead=args.lookahead, delta=args.delta, max_threshold=args.maxThreshold)

    save_domains_and_boundaries(chrom, chr_start, chr_end, matrix, min_idx, args)

    # turn of hierarchical clustering which is apparently not working.
    if 2==1:
        boundary_list = [(hic_ma.cut_intervals[min_][0], hic_ma.cut_intervals[min_][2], mean_mat[min_]) for min_ in min_idx]

        Z, clusters = hierarchical_clustering(boundary_list, clusters_cutoff=[0.4, 0.3, 0.2])

        save_linkage(Z, args.outPrefix + '_linkage.bed')
        save_clusters(clusters, args.outPrefix)
