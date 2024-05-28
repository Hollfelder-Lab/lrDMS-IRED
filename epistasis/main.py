import csv
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from analysis_utils import preprocessing, double_mut_pos, epistasis_graph, epistatic_triangles, comb_pos_mut, construct_structural_epistasis_graph
from plotting_utils import plot_node_degree_distribution, plot_node_degree_aa_distribution, \
    plot_mutation_distribution, plot_epistasis_model
import numpy as np
import itertools
import scipy as sp

# Upload input files into panda data frame
data_frame = pd.read_csv('MePy_merge_filtered_260923.csv')

# Specify sequence of reference protein
reference = "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPS*"

# Specify maximum order of mutations to be analysed i.e. mutations up to that order will be analysed
num_mut = 5

# Preprocess the data
preprocessed_data = preprocessing(data_frame, num_mut, reference)

############# ALL MUTATIONS
for i in range(2, num_mut + 1):
    locals()["mut_" + str(i) + "_sequence_list"] = preprocessed_data[str(i) + " Mutation"]["Sequence of mutants"]
    locals()["mut_" + str(i) + "_W_observed_list"] = preprocessed_data[str(i) + " Mutation"]["Observed fitness"]
    locals()["mut_" + str(i) + "_W_observed_std_list"] = preprocessed_data[str(i) + " Mutation"]["Observed std of " \
                                                                                                 "fitness"]
    locals()["mut_" + str(i) + "_W_expected_list"] = preprocessed_data[str(i) + " Mutation"]["Expected fitness"]
    locals()["mut_" + str(i) + "_W_expected_std_list"] = preprocessed_data[str(i) + " Mutation"]["Expected std of " \
                                                                                                 "fitness"]
    locals()["mut_" + str(i) + "_epistatic_score_list"] = preprocessed_data[str(i) + " Mutation"]["Epistatic score"]

full_mut_sequence_list = []
full_mut_W_observed_list = []
full_mut_W_observed_std_list = []
full_mut_W_expected_list = []
full_mut_W_expected_std_list = []
full_mut_epistatic_score_list = []

for mut_num_i in range(2, num_mut + 1):
    full_mut_sequence_list = full_mut_sequence_list + locals()["mut_" + str(mut_num_i) + "_sequence_list"]
    full_mut_W_observed_list = full_mut_W_observed_list + locals()["mut_" + str(mut_num_i) + "_W_observed_list"]
    full_mut_W_observed_std_list = full_mut_W_observed_std_list + locals()[
        "mut_" + str(mut_num_i) + "_W_observed_std_list"]
    full_mut_W_expected_list = full_mut_W_expected_list + locals()["mut_" + str(mut_num_i) + "_W_expected_list"]
    full_mut_W_expected_std_list = full_mut_W_expected_std_list + locals()[
        "mut_" + str(mut_num_i) + "_W_expected_std_list"]
    full_mut_epistatic_score_list = full_mut_epistatic_score_list + locals()[
        "mut_" + str(mut_num_i) + "_epistatic_score_list"]

# Plot mutation distribution for double mutations
plot_mutation_distribution(mut_2_sequence_list, reference)

# Plot mutation distribution for mutations 2 - 5
plot_mutation_distribution(full_mut_sequence_list, reference)

# Plot mutation distribution for mutations 3 - 5
mut_3_5_sequence_list = []
mut_3_5_W_observed_list = []
mut_3_5_W_expected_list = []
mut_3_5_epistatic_score_list = []

for mut_num_i in range(3, num_mut + 1):
    mut_3_5_sequence_list = mut_3_5_sequence_list + locals()["mut_" + str(mut_num_i) + "_sequence_list"]
    mut_3_5_W_observed_list = mut_3_5_W_observed_list + locals()["mut_" + str(mut_num_i) + "_W_observed_list"]
    mut_3_5_W_expected_list = mut_3_5_W_expected_list + locals()["mut_" + str(mut_num_i) + "_W_expected_list"]
    mut_3_5_epistatic_score_list = mut_3_5_epistatic_score_list + locals()["mut_" + str(mut_num_i) +
                                                                           "_epistatic_score_list"]

plot_mutation_distribution(mut_3_5_sequence_list, reference)


# Plot epistasis model for double mutations
r_d_s, p_d_s = sp.stats.pearsonr(mut_2_W_expected_list, mut_2_W_observed_list)
print(f"correlation calculated double mutations fitness / double mutation fitness: pearson r = {r_d_s} with p = {p_d_s}")
plot_epistasis_model(mut_2_W_expected_list, mut_2_W_observed_list, mut_2_epistatic_score_list)

# Plot epistasis model for mutations 3 - 5
r_d_s, p_d_s = sp.stats.pearsonr(mut_3_5_W_expected_list, mut_3_5_W_observed_list)
print(f"correlation calculated double mutations fitness / double mutation fitness: pearson r = {r_d_s} with p = {p_d_s}")
plot_epistasis_model(mut_3_5_W_expected_list, mut_3_5_W_observed_list, mut_3_5_epistatic_score_list)

# Plot epistasis model for mutations 2 - 5
r_d_s, p_d_s = sp.stats.pearsonr(full_mut_W_expected_list, full_mut_W_observed_list)
print(f"correlation calculated double mutations fitness / double mutation fitness: pearson r = {r_d_s} with p = {p_d_s}")
plot_epistasis_model(full_mut_W_expected_list, full_mut_W_observed_list, full_mut_epistatic_score_list)

# Combine two lists into combined list of double mutation positions
comb_pos_mut_pos_list, comb_pos_mut_aa_list = comb_pos_mut(full_mut_epistatic_score_list, full_mut_W_observed_list,
                                                           full_mut_W_expected_list, full_mut_W_observed_std_list,
                                                           full_mut_sequence_list, reference, 1, 1)

# Unpack list of into pairs
pos_comb_mut_edges = []
pos_comb_mut_aa = []
for higher_ord_mut in range(0, len(comb_pos_mut_pos_list)):
    higher_order_mut_list = (list(map(list, itertools.combinations(comb_pos_mut_pos_list[higher_ord_mut], 2))))
    higher_order_mut_aa_list = (list(map(list, itertools.combinations(comb_pos_mut_aa_list[higher_ord_mut], 2))))
    if len(higher_order_mut_list) == 2:
        pos_comb_mut_edges.append(higher_order_mut_list)
        pos_comb_mut_aa.append(higher_order_mut_aa_list)
    else:
        for higher_order_mut_list_ele in range(0, len(higher_order_mut_list)):
            pos_comb_mut_edges.append(higher_order_mut_list[higher_order_mut_list_ele])
            pos_comb_mut_aa.append(higher_order_mut_aa_list[higher_order_mut_list_ele])

# Epistasis graph for all higher order mutants
higher_order_mut_epistasis_graph = epistasis_graph(pos_comb_mut_edges)
nx.draw(higher_order_mut_epistasis_graph, with_labels=True, font_weight='bold')
plt.show()

# Load distance matrix
dist_matrix = np.load("min_dimer_distances.npy")

# Structural epistasis graph for all higher order mutants
structural_epistasis_graph = construct_structural_epistasis_graph(pos_comb_mut_edges, 5, dist_matrix)
pos = nx.get_node_attributes(structural_epistasis_graph, 'pos')

plt.figure()
nx.draw(structural_epistasis_graph, pos, with_labels=True, font_weight='bold')
plt.show()

# Find largest cliques in largest component of graph
largest_cc = max(nx.connected_components(higher_order_mut_epistasis_graph), key=len)
higher_order_mut_epistasis_subgraph = higher_order_mut_epistasis_graph.subgraph(largest_cc)
#nx.draw(higher_order_mut_epistasis_subgraph, with_labels=True, font_weight='bold')
largest_cliques = sorted(nx.find_cliques(higher_order_mut_epistasis_subgraph), key=len, reverse=True)
print(largest_cliques)

# Node degree for each position
mut_2_5_node_degree_list = np.array(list(map(list, sorted(higher_order_mut_epistasis_graph.degree, key=lambda x: x[1], reverse=True))),
                                dtype=int)
fields = ["Amino Acid Position", "Node Degree"]
rows = mut_2_5_node_degree_list.tolist()

with open('mut_2_5_cut_1_-1.csv', 'w') as mut_2_5_cut_1_1:
    # using csv.writer method from CSV package
    write = csv.writer(mut_2_5_cut_1_1)

    write.writerow(fields)
    write.writerows(rows)

# Node degree analysis (node, degree) in descending order
plot_node_degree_distribution(higher_order_mut_epistasis_graph)

plot_node_degree_distribution(higher_order_mut_epistasis_graph, frequency=True, sequences=full_mut_sequence_list,
                              reference=reference)
# Node degree and amino acid distribution
pos_comb_higher_mut_pos = np.concatenate(
    (np.array(pos_comb_mut_edges)[:, 0], np.array(pos_comb_mut_edges)[:, 1].astype(int)), axis=0)
pos_comb_higher_mut_mut_aa = np.concatenate((np.array(pos_comb_mut_aa)[:, 0], np.array(pos_comb_mut_aa)[:, 1]), axis=0)

pos_comb_higher_mut_pos_aa = np.stack((pos_comb_higher_mut_pos, pos_comb_higher_mut_mut_aa), axis=1)

pos_per_aa_dict = plot_node_degree_aa_distribution(pos_comb_higher_mut_pos_aa)

############# DOUBLE MUTATIONs
# Unpack the preprocessed data
single_mut_W_observed_std = preprocessed_data["1 Mutation"]["Observed std of fitness"]
sequence_double_list = preprocessed_data["2 Mutation"]["Sequence of mutants"]
W_observed_list = preprocessed_data["2 Mutation"]["Observed fitness"]
W_observed_std_list = preprocessed_data["2 Mutation"]["Observed std of fitness"]
W_expected_list = preprocessed_data["2 Mutation"]["Expected fitness"]
W_expected_std_list = preprocessed_data["2 Mutation"]["Expected std of fitness"]
epistatic_score_list = preprocessed_data["2 Mutation"]["Epistatic score"]

# Create a list of positive and combinable positions of double mutations
pos_comb_double_mut_list_full = double_mut_pos(epistatic_score_list, W_observed_list, W_expected_std_list,
                                               W_observed_std_list, sequence_double_list, reference, 1, -1)
pos_comb_double_mut_list = pos_comb_double_mut_list_full[:, :2].astype(int)

# Determine all epistatic triangles for all AA positions
epistatic_triangle_list = epistatic_triangles(pos_comb_double_mut_list)
print(" Epistatic triangles: ", epistatic_triangle_list)

# Create epistasis double_mut_epistasis_graph given list of double mutation positions
double_mut_epistasis_graph = epistasis_graph(pos_comb_double_mut_list)

# Plot epistasis double_mut_epistasis_graph
nx.draw(double_mut_epistasis_graph, with_labels=True, font_weight='bold')
plt.show()

# Node degree for each position
mut_2_node_degree_list = np.array(list(map(list, sorted(double_mut_epistasis_graph.degree, key=lambda x: x[1], reverse=True))),
                                dtype=int)
fields = ["Amino Acid Position", "Node Degree"]
rows = mut_2_node_degree_list.tolist()

with open('mut_2_cut_1_-1.csv', 'w') as mut_2_cut_1_1:
    # using csv.writer method from CSV package
    write = csv.writer(mut_2_cut_1_1)

    write.writerow(fields)
    write.writerows(rows)

# Node degree analysis (node, degree) in descending order
#plot_node_degree_distribution(double_mut_epistasis_graph)

# Node degree + amino acid distribution
pos_comb_double_mut_pos = np.concatenate(
    (pos_comb_double_mut_list_full[:, 0].astype(int), pos_comb_double_mut_list_full[:, 1].astype(int)), axis=0)
pos_comb_double_mut_aa = np.concatenate((pos_comb_double_mut_list_full[:, 2], pos_comb_double_mut_list_full[:, 3]),
                                        axis=0)
pos_comb_double_mut_pos_aa = np.stack((pos_comb_double_mut_pos, pos_comb_double_mut_aa), axis=1)

pos_per_aa_dict = plot_node_degree_aa_distribution(pos_comb_double_mut_pos_aa)

