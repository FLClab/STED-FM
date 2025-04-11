import numpy
import pickle
import os
import itertools
import random
import csv
import json
import pandas
import scikit_posthocs

from tqdm.auto import tqdm
from matplotlib import pyplot
from scipy import stats
from collections import defaultdict

random.seed(42)
numpy.random.seed(42)

class Combinations:
    def __init__(self, samples, r=2, possible_combinations=[]):
        self.samples = samples
        if possible_combinations:
            self.possible_combinations = possible_combinations
        else:
            self.possible_combinations = list(itertools.combinations(range(len(samples)), r=r))
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.current >= len(self.possible_combinations):
            raise StopIteration
        self.current += 1
        return tuple(self.samples[c] for c in self.possible_combinations[self.current - 1])

    def __len__(self):
        return len(self.possible_combinations)

def permute(samples, group_indexes):
    """
    Permutes a raveled sampled array and returns the new smaples
    :param samples: A `numpy.ndarray` with shape (N, )
    :param group_indexes: A `list` of group indexes
    :returns : A `numpy.ndarray` of the permuted samples
    """
    numpy.random.shuffle(samples)
    return [samples[index] for index in group_indexes]

def resampling_F(samples, raveled_samples, group_indexes, permutations=10000):
    """
    Computes the F statistics using a resampling of samples
    :param samples: A `list` of sample
    :param permutations: The number of permutations to test
    """
    gt_fstat, _ = stats.f_oneway(*samples)
    p_fstat = []
    for _ in range(permutations):
        tmp_samples = permute(raveled_samples, group_indexes)
        statistic, _  = stats.f_oneway(*tmp_samples)
        p_fstat.append(statistic)
    p_fstat = numpy.array(p_fstat)
    p_value = numpy.sum(p_fstat >= gt_fstat, axis=0) / permutations
    return p_value

def resampling_stats(samples, labels, raveled_samples=None, group_indexes=None, permutations=10000, show_ci=False,
                        bin_edges=None, possible_combinations=[]):
    """
    Computes the pair-wise comparisons of each sample in the list using a resampling
    statistical test
    
    :param samples: A `list` of sample
    :param raveled_samples: A `list` of all available samples
    :param group_indexes: A `list` of associated groups 
    :param labels: A `list` of label 
    :param permutations: An `int` of the number of permutations to do 
    :param show_ci: Wheter to plot the condifence interval 

    :returns : A `list` of p-values for each comparisons
    """
    # Make sure that the samples are numpy arrays
    samples = [numpy.array(sample) for sample in samples]

    if isinstance(raveled_samples, type(None)):
        raveled_samples, group_indexes = [], []
        current_count = 0
        for i, samp in enumerate(samples):
            raveled_samples.extend(samp)
            group_indexes.append(current_count + numpy.arange(len(samp)))
            current_count += len(samp)

    raveled_samples = numpy.array(raveled_samples)
        
    p_values = pandas.DataFrame(data=-1 * numpy.ones((len(labels), len(labels))), index=labels, columns=labels)

    # Resampled anova
    F_p_value = None
    if len(samples) > 2:
        F_p_value = resampling_F(samples, raveled_samples, group_indexes, permutations=permutations)
        if numpy.all(F_p_value > 0.05):
            print("Resampling F (pvalue : {})".format(F_p_value))
            return p_values, F_p_value
        else:
            pass

    possible_treatments = []
    for t1, t2 in Combinations(labels, r=2, possible_combinations=possible_combinations):
        possible_treatments.append([t1, t2])
    possible_treatments = set(map(tuple, possible_treatments))

    if show_ci:
        plot_info = {
            key : {
                "figax" : pyplot.subplots(tight_layout=True, figsize=(12, 3)),
                "current_count" : 0,
                "treatments" : []
            } for key in possible_treatments
        }
    for j, ((sample1, sample2), (treatment1, treatment2)) in enumerate(zip(tqdm(Combinations(samples, r=2, possible_combinations=possible_combinations)),\
                                                                    Combinations(labels, r=2, possible_combinations=possible_combinations))):
        # if i % 9 == 0:
        #     treatments = []
        gt_abs_diff = numpy.abs(numpy.mean(sample1, axis=0) - numpy.mean(sample2, axis=0))
        concatenated = numpy.concatenate((sample1, sample2), axis=0)
        p_abs_diff = []
        for _ in range(permutations):
            numpy.random.shuffle(concatenated)
            p_abs_diff.append(numpy.abs(numpy.mean(concatenated[:len(sample1)], axis=0) - numpy.mean(concatenated[len(sample1):], axis=0)))
        p_abs_diff = numpy.array(p_abs_diff)
        p_value = numpy.sum(p_abs_diff >= gt_abs_diff, axis=0) / permutations

        # p_values.append(p_value)
        p_values.loc[treatment1, treatment2] = p_value
        p_values.loc[treatment2, treatment1] = p_value

        if show_ci:
            key = tuple([treatment1, treatment2])
            fig, ax = plot_info[key]["figax"]
            i = plot_info[key]["current_count"]
            plot_info[key]["treatments"].append(f"{treatment1}, {treatment2}")

            ax.bar(i, numpy.quantile(p_abs_diff, q=0.95), color="grey", alpha=0.7, width=1, zorder=3)
            ax.bar(i, gt_abs_diff, alpha=0.7, color="tab:blue", width=1, zorder=1)
            # output = f"STATS-{treatment1}-{treatment2}"
            # ax.set_title(output)
            ax.set_xticks(numpy.arange(len(plot_info[key]["treatments"])))
            ax.set_xticklabels(plot_info[key]["treatments"], rotation=45, horizontalalignment="right")
            ax.set_ylim(0, 0.2)

            plot_info[key]["current_count"] += 1

    return p_values, F_p_value

def create_latex_table(pvalues, scores, formatted_labels, output_file=None, group_name=None):
    """
    Creates a latex table by using pandas as a backend 
    
    :param pvalues: A list of pvalues 
    :param scores: A list of all scores 
    :param formatted_labels: A list of formatted labesl 
    :param output_file: (Optional) A string path to the output_file 
    """
    def formatter(x):
        try:
            x = float(x)
            if MINVAL == abs(x):
                formatted_x = "<\\SI{1.0000e-4}{}"
            else:
                formatted_x = "\\SI{{{:0.4e}}}{{}}".format(abs(x))
            if x < 0:
                if (x > -0.05): 
                    return "\\textcolor[rgb]{{0.93,0.26,0.18}}{{{}}}".format(formatted_x)
                else :
                    return formatted_x
            else:
                if (x < 0.05):
                    return "\\textcolor[rgb]{{0.39,0.68,0.75}}{{{}}}".format(formatted_x)
                else:
                    return formatted_x
        except ValueError:
            return "-"
    MINVAL = 1e-9
    df = pandas.DataFrame(index=formatted_labels, columns=formatted_labels)
    for (row, col), (s1, s2), val in zip(Combinations(formatted_labels, r=2), Combinations(scores, r=2), pvalues):
        is_smaller = (numpy.mean(s1) - numpy.mean(s2)) < 0
        if is_smaller:
            df.loc[row, col] = "{}".format(-1 * max(val, MINVAL))
        else:
            df.loc[row, col] = "{}".format(max(val, MINVAL))

        is_smaller = (numpy.mean(s2) - numpy.mean(s1)) < 0
        if is_smaller:
            df.loc[[col], row] = "{}".format(-1 * max(val, MINVAL))
        else:
            df.loc[col, row] = "{}".format(max(val, MINVAL))

    with pandas.option_context("max_colwidth", 1000):
        out = df.to_latex(open(output_file, "w") if isinstance(output_file, str) else output_file, 
                          formatters=[formatter] * len(formatted_labels),
                          na_rep="-", column_format="c" * (len(formatted_labels) + 1),
                          escape=False)
    return out

def plot_p_values(p_values):

    fig, ax = pyplot.subplots(figsize=(3,3))
    cmap = ['1', '#aaaaaa',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {
        'cmap': cmap, 'linewidths': 1.0, 'linecolor': '0.', 
        'clip_on': False, 'square': True, 
    }
    scikit_posthocs.sign_plot(p_values, ax=ax, **heatmap_args)
    return fig, ax