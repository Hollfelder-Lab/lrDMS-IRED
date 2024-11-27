# Plotting functions for project
import itertools
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from analysis_utils import call_aa_simple


def plot_obs_fitness_heatmap(reference_seq, double_mut_seqs: list, obs_fitness_scores: list):
    """
    Plots a heatmap of observed fitness on a len(reference_seq) x len(reference_seq) grid

    :param reference_seq: string of amino acids of reference protein sequence
    :param double_mut_seqs: list of sequences of mutants with double mutations
    :param obs_fitness_scores: list of observed fitness scores
    :return: None
    """
    obs_fitness_heatmap = np.ones((len(reference_seq), len(reference_seq))) * -6

    obs_fitness_pos_score_list = []

    for double_mut in range(len(double_mut_seqs)):
        obs_fitness_pos_score_list_element = []
        sequence = double_mut_seqs[double_mut]
        _, pos, _ = call_aa_simple(reference_seq, sequence)
        obs_fitness = obs_fitness_scores[double_mut]

        # Append list element with double mutation positions and corresponding epistatic score / observed fitness
        obs_fitness_pos_score_list_element.append(pos)
        obs_fitness_pos_score_list_element.append(obs_fitness)

        # Append total epistatic score list with element
        obs_fitness_pos_score_list.append(obs_fitness_pos_score_list_element)

    for double_mut in range(len(obs_fitness_pos_score_list)):
        index_pair = obs_fitness_pos_score_list[double_mut][0]
        score_of_index_pair = obs_fitness_pos_score_list[double_mut][1]
        obs_fitness_heatmap[index_pair[0] - 1, index_pair[1] - 1] = score_of_index_pair
        obs_fitness_heatmap[index_pair[1] - 1, index_pair[0] - 1] = score_of_index_pair

    plt.imshow(obs_fitness_heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Fitness Heatmap")
    plt.xlabel("Mutated Amino Acid Position")
    plt.ylabel("Mutated Amino Acid Position")
    plt.show()


def plot_node_degree_distribution(epistasis_graph: nx.Graph, frequency: bool = False, sequences: list = [], reference=[]):
    """
    Given an epistasis graph, plot the node degree distribution (# epistatic interaction per amino acid position) for
    each node (amino acid position)

    :param reference: string of amino acids of the reference sequence
    :param sequences: list of mutant sequences
    :param frequency: Node degree distribution by frequency instead of counts
    :param epistasis_graph: Networkx graph
    :return: None
    """

    node_degree_list = np.array(list(map(list, sorted(epistasis_graph.degree, key=lambda x: x[1], reverse=True))),
                                dtype=float)

    if frequency:
        full_mut_pos_list = []
        full_mut_aa_list = []

        # Create lists of double mutation positions dependent on positiveness and combinability
        for seq in range(len(sequences)):
            sequence = sequences[seq]
            _, mut_pos, mut_aa = call_aa_simple(reference, sequence)

            full_mut_pos_list.append(mut_pos)
            full_mut_aa_list.append(mut_aa)

        full_mut_pos_dist_list = list(itertools.chain.from_iterable(full_mut_pos_list))

        unique_dist_pos, unique_dist_count = np.unique(np.array(full_mut_pos_dist_list), return_counts=True)

        for pos_node_idx in range(0, len(node_degree_list)):
            for pos in range(0, len(unique_dist_pos)):
                if unique_dist_pos[pos] == node_degree_list[pos_node_idx, 0]:
                    node_degree_list[pos_node_idx, 1] = node_degree_list[pos_node_idx, 1] / unique_dist_count[pos]

    # Plot node degree list as bar chart
    plt.figure(figsize=[15, 3])
    plt.bar(node_degree_list[:, 0].astype(int), node_degree_list[:, 1], color="tomato", linewidth=0, alpha=0.8)
    plt.xlim(1, 291)
    for a in range(10, 290, 10):
        plt.axvline(x=a, color="k", alpha=0.6, linewidth=0.3)
    # plt.legend(loc="lower right")
    plt.locator_params(axis="x", nbins=29)
    plt.locator_params(axis="y", nbins=10)
    plt.title("Node Degree Distribution")
    plt.xlabel("Amino Acid Position")
    plt.ylabel("Number of Epistatic Interactions")
    plt.show()


def plot_node_degree_aa_distribution(mut_aa: np.ndarray, protein_length: int, node_degree_type: str) -> dict:
    """
    Given a n x 2 dimensional numpy array of mutated amino acids, plot the node degree and amino acid distribution as
    bar chart and return a dictionary of amino acids per positions

    :param mut_aa:
    :param protein_length:
    :param node_degree_type:
    :return: pos_per_aa_dict
    """
    if node_degree_type=="Epistasis":
        ylabel_title = "Number of Epistatic Interactions"
    elif node_degree_type=="Combinability":
        ylabel_title = "Number of Combinable Interactions"
    else:
        raise TypeError("Only 'Combinability' or 'Epistasis' is implemented")

    # Return the unique rows
    unique_pos, unique_pos_counts = np.unique(mut_aa, return_counts=True, axis=0)

    # check for present AA in all data
    unique_aa = np.unique(mut_aa[:, 1]).tolist()

    # Create dict for each AA, for each dict the counts
    pos_per_aa_dict = dict.fromkeys(unique_aa)

    # Fill each entry with a 291 long list containing the positional counts

    for sel_aa in unique_aa:
        aa_count_list = np.zeros(protein_length + 1)
        for pos in range(0, len(unique_pos)):
            if unique_pos[pos, 1] == sel_aa:
                aa_count_list[unique_pos[pos, 0].astype(int)] = unique_pos_counts[pos]
        pos_per_aa_dict[sel_aa] = aa_count_list

    x = np.arange(0, protein_length + 1)
    plt.figure(figsize=[15, 3])
    stored_value = np.zeros(protein_length + 1)
    idx = 0
    cmap = plt.get_cmap('nipy_spectral')
    slicedCM = cmap(np.linspace(0, 1, len(unique_aa)))
    for a in range(10, protein_length, 10): # for SrIred 290 protein length -1
        plt.axvline(x=a, color="k", alpha=0.6, linewidth=0.3)
    for sp_aa in unique_aa:
        plt.bar(x, pos_per_aa_dict[sp_aa], bottom=stored_value, color=slicedCM[idx], label=unique_aa[idx])
        stored_value = stored_value + pos_per_aa_dict[sp_aa]
        idx = idx + 1
    plt.xlim(1, protein_length)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.locator_params(axis="x", nbins=29)
    plt.locator_params(axis="y", nbins=10)
    plt.title("Node Degree Distribution")
    plt.xlabel("Amino Acid Position")
    plt.ylabel(ylabel_title)
    plt.show()

    return pos_per_aa_dict


def plot_mutation_distribution(sequences: list, reference: str):
    """
    Takes the full list of all mutants and plots the mutation distribution (number of mutations per position)

    :param sequences: list of all strings of mutants
    :param reference: string of amino acids of the reference protein
    :return: None
    """
    full_mut_pos_list = []
    full_mut_aa_list = []

    # Create lists of double mutation positions dependent on positiveness and combinability
    for seq in range(len(sequences)):
        sequence = sequences[seq]
        _, mut_pos, mut_aa = call_aa_simple(reference, sequence)

        full_mut_pos_list.append(mut_pos)
        full_mut_aa_list.append(mut_aa)

    full_mut_pos_dist_list = list(itertools.chain.from_iterable(full_mut_pos_list))

    unique_dist_pos, unique_dist_count = np.unique(np.array(full_mut_pos_dist_list), return_counts=True)

    plt.figure(figsize=[15, 3])
    for a in range(10, 290, 10):
        plt.axvline(x=a, color="k", alpha=0.6, linewidth=0.3)
    plt.bar(np.array(unique_dist_pos), np.array(unique_dist_count))
    plt.xlim(1, 291)
    plt.locator_params(axis="x", nbins=29)
    plt.locator_params(axis="y", nbins=10)
    plt.title("Mutation Distribution")
    plt.xlabel("Amino Acid Position")
    plt.ylabel("Number of Mutations")
    plt.show()


# ln_W_axb_list -> W_expected_list
# ln_W_a_b_list -> W_observed_list
# e -> epistatic_score_list

def plot_epistasis_model(exp_fitness_scores: list, obs_fitness_scores: list, epistatic_scores: list):
    """
    Plots the epistasis model for all mutants as scatter plot

    :param exp_fitness_scores: list of expected fitness scores for each mutant
    :param obs_fitness_scores: list of observed fitness scores for each mutant
    :param epistatic_scores: list of epistatic scores for each mutant
    :return: None
    """

    plt.figure()  # default figsize 6.4 4.8
    plt.axhline(0, c="grey", linewidth=1)
    plt.axvline(0, c="grey", linewidth=1)
    # plt.plot([-1,1], [0.3, 0.3], c="grey", linewidth=1)
    # plt.plot([0,0], [0.3, 1], c="grey", linewidth=1)
    plt.scatter(exp_fitness_scores, obs_fitness_scores, s=10, c=epistatic_scores, cmap="coolwarm")
    plt.xlabel("Expected fitness scores")
    plt.ylabel("Observed fitness scores")
    # plt.ylim([-0.5,0.9])
    # plt.xlim([-0.95,0.55])
    plt.colorbar(ticks=np.arange(-2.5, 2.5, 0.2), label=u'\u03B5');
    plt.clim(-2.5, 2.5)
    # plt.savefig("correlation_calculated_double_double_epsilon.pdf", bbox_inches='tight')
    plt.show()
