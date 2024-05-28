import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import pandas as pd
from typing import Tuple, Literal


def call_aa_simple(reference_seq: str, target_seq: str) -> Tuple[list, list, list]:
    """
    Takes a reference protein sequence and compares it with a possible mutant protein sequence to extract lists of
    wildtype amino acids, mutated amino acid positions, and mutated amino acid

    :param reference_seq: amino acid sequence of reference protein (string)
    :param target_seq: amino acid sequence of mutated protein (string)
    :return: mut_list_wt, mut_list_pos, mut_list_mut: List of mutated wild type amino acids, list of mutated amino acid
    positions, list of mutated amino acids
    """
    # Check for indels
    if len(reference_seq) != len(str(target_seq)):
        #print("Indel detected")
        return "indel", "indel", "indel"

    # call mutations in 3 lists if only substitutions -> dataframe would be better
    mut_list_wt = []
    mut_list_pos = []
    mut_list_mut = []
    for seq in range(0, len(reference_seq)):
        if reference_seq[seq] != target_seq[seq]:
            mut_list_wt.append(reference_seq[seq])
            mut_list_pos.append(seq + 1)
            mut_list_mut.append(target_seq[seq])

    return mut_list_wt, mut_list_pos, mut_list_mut


def preprocessing(data_frame: pd.DataFrame, num_mut: int, reference_seq: str) -> dict[dict]:
    """
    Pre-processing of the data to determine fitness scores, standard deviations of the fitness scores, and mutated amino
    acid positions for single mutations and the specified higher order mutation type

    :param data_frame: panda data frame containing all raw data information about the mutants
    :param num_mut: number of mutations to analyse
    :param reference_seq: string of amino acids of reference protein
    :return: preprocessed_data: nested dictionary of preprocessed data subdivided into single mutations and higher order
    mutations
    """

    # Save columns of data frame into lists
    seq_raw_list = data_frame["aa_seq"].tolist()
    seq_ids_list = data_frame.iloc[:, 0].tolist()
    nhams_list = data_frame["Nham_aa"].tolist()
    fitness_list = data_frame["fitness"].tolist()
    observed_std_list = data_frame["sigma"].tolist()

    for i in range(1, num_mut + 1):
        locals()["mutation_dict" + str(i)] = {
            "Positions": [],
            "Fitness": [],
            "Mutated AA": [],
            "Observed std of fitness": [],
            "Pre sequence list": []
        }

    # Empty lists of preprocessed data
    preprocessed_data = {}

    for i in range(1, num_mut + 1):
        preprocessed_data[str(i) + " Mutation"] = locals()["mutation_dict" + str(i)]

    for i in range(len(seq_raw_list)):

        wt, pos, mut = call_aa_simple(reference_seq, seq_raw_list[i])
        if mut != "indel":
            n_mut = len(mut)
        else:
            n_mut = "NA"

        for mutation_num in np.arange(1, len(preprocessed_data) + 1):
            if n_mut == mutation_num:
                preprocessed_data[str(n_mut) + " Mutation"]["Positions"].append(pos)
                preprocessed_data[str(n_mut) + " Mutation"]["Fitness"].append(fitness_list[i])
                preprocessed_data[str(n_mut) + " Mutation"]["Mutated AA"].append(mut)
                preprocessed_data[str(n_mut) + " Mutation"]["Observed std of fitness"].append(observed_std_list[i])
                preprocessed_data[str(n_mut) + " Mutation"]["Pre sequence list"].append(seq_raw_list[i])

    for num_mut_analysed in range(1, num_mut + 1):
        num_variants_analysed = len(preprocessed_data[str(num_mut_analysed) + " Mutation"]["Fitness"])
        print(f"number of variants with mutations of order {num_mut_analysed} analyzed: {num_variants_analysed}")

    for higher_ord_mut in range(2, num_mut + 1):

        number_possible_epistatic_events = 0

        W_expected_list = []
        W_observed_list = []
        W_expected_std_list = []
        W_observed_std_list = []
        epistatic_score_list = []
        W_expected_list_non_log = []
        W_observed_list_non_log = []

        # for counting epistatic hotspots
        positive_positions = []

        higher_ord_seq_list = []

        for i in range(len(preprocessed_data[str(higher_ord_mut) + " Mutation"]["Positions"])):
            # get positions of the higher order mutation
            positions = preprocessed_data[str(higher_ord_mut) + " Mutation"]["Positions"][i]
            mut_aas = preprocessed_data[str(higher_ord_mut) + " Mutation"]["Mutated AA"][i]
            W_observed = preprocessed_data[str(higher_ord_mut) + " Mutation"]["Fitness"][i]
            W_observed_std = preprocessed_data[str(higher_ord_mut) + " Mutation"]["Observed std of fitness"][i]

            # for every position found +1
            num_found = 0

            # list with all the single fitness_list values for this mutation
            single_fitness_list = []
            single_observed_std_list = []
            # for QC also the identity of single mutations:
            single_pos_list = []
            single_mut_aa_list = []

            for sub_pos in range(len(positions)):

                # loop through single mutant information
                for a in range(len(preprocessed_data["1 Mutation"]["Positions"])):
                    # if match for subpositions -> append lists and numb found += 1
                    if preprocessed_data["1 Mutation"]["Positions"][a][0] == positions[sub_pos] and \
                            preprocessed_data["1 Mutation"]["Mutated AA"][a][0] == mut_aas[sub_pos]:
                        num_found += 1
                        single_fitness_list.append(preprocessed_data["1 Mutation"]["Fitness"][a])
                        single_observed_std_list.append(preprocessed_data["1 Mutation"]["Observed std of fitness"][a])
                        # only for control
                        single_pos_list.append(preprocessed_data["1 Mutation"]["Positions"][a])
                        single_mut_aa_list.append(preprocessed_data["1 Mutation"]["Mutated AA"][a])

            # only if all positions found: proceed
            if num_found == len(positions):
                number_possible_epistatic_events += 1
                single_observed_std_list.append(
                    W_observed_std)  # Contains now full observed single and double mutations stdv
                single_observed_std_list = np.array(single_observed_std_list)

                # product model of expected fitness:
                W_expected = sum(single_fitness_list)
                W_expected_std = np.sqrt((single_observed_std_list ** 2).sum())  # Add W_observed_std
                epistatic_score = W_observed - W_expected

                W_expected_list.append(W_expected)
                W_expected_std_list.append(W_expected_std)
                W_observed_list.append(W_observed)
                W_observed_std_list.append(W_observed_std)

                epistatic_score_list.append(epistatic_score)
                higher_ord_seq_list.append(preprocessed_data[str(higher_ord_mut) + " Mutation"]["Pre sequence list"][i])
                # for model QC:
                W_observed_list_non_log.append(np.exp(W_observed))
                W_expected_list_non_log.append(np.exp(W_expected))

                """
                #additive model: 
                non_log_W_list = []
                for W in range(len(single_fitness_list)):
                    non_log_W_list.append(np.exp(single_fitness_list[W]))
    
                W_expected = np.log(sum(non_log_W_list) - 1)
                epistatic_score = W_observed - W_expected
    
                W_expected_list.append(W_expected)
                W_observed_list.append(W_observed)
                epistatic_score_list.append(epistatic_score)
    
                """

        preprocessed_data[str(higher_ord_mut) + " Mutation"]["Sequence of mutants"] = higher_ord_seq_list
        preprocessed_data[str(higher_ord_mut) + " Mutation"]["Observed fitness"] = W_observed_list
        preprocessed_data[str(higher_ord_mut) + " Mutation"]["Observed std of fitness"] = W_observed_std_list
        preprocessed_data[str(higher_ord_mut) + " Mutation"]["Expected fitness"] = W_expected_list
        preprocessed_data[str(higher_ord_mut) + " Mutation"]["Expected std of fitness"] = W_expected_std_list
        preprocessed_data[str(higher_ord_mut) + " Mutation"]["Epistatic score"] = epistatic_score_list

    return preprocessed_data


def comb_pos_mut(epistatic_scores: list, obs_fitness_scores: list, exp_fitness_stdvs: list, obs_fitness_stdvs: list,
                 mut_seqs: list,
                 reference_seq: str, sig_std_obs: int, sig_std_exp: int) -> Tuple[list, list]:
    """
    Creates a list of at least double mutation positions based on positive epistatic effect and combinability (
    positive epistatic effect and additive single mutations)

    :param epistatic_scores: list of expected fitness scores for each mutant
    :param obs_fitness_scores: list of observed fitness scores for each mutant
    :param exp_fitness_stdvs: list of standard deviations of expected fitness score for list of mutants
    :param obs_fitness_stdvs: list of standard deviations of observed fitness score for list of mutants
    :param mut_seqs: list of sequences containing at least double mutations (with sufficient information)
    :param reference_seq: reference sequence of protein
    :param sig_std_obs: user defined multiple of standard deviation of observed fitness
    :param sig_std_exp: user defined multiple of standard deviation of expected fitness
    :return: comb_pos_mut_list: list of positive and combinable (at least double) mutation positions
    """

    mut_pos_list = []
    mut_aa_list = []

    # Create lists of double mutation positions dependent on positiveness and combinability
    for seq in range(len(mut_seqs)):
        # Ensure positive fitness effect and combinability
        if obs_fitness_scores[seq] > sig_std_obs * obs_fitness_stdvs[seq] and epistatic_scores[
            seq] > sig_std_exp * exp_fitness_stdvs[seq]:
            sequence = mut_seqs[seq]
            _, mut_pos, mut_aa = call_aa_simple(reference_seq, sequence)

            mut_pos_list.append(mut_pos)
            mut_aa_list.append(mut_aa)

    # Combine two lists into combined list of double mutation positions
    comb_pos_mut_pos_list = np.array(mut_pos_list, dtype=object)
    comb_pos_mut_aa_list = np.array(mut_aa_list, dtype=object)

    return comb_pos_mut_pos_list, comb_pos_mut_aa_list


def double_mut_pos(epistatic_scores: list, obs_fitness_scores: list, exp_fitness_stdvs: list, obs_fitness_stdvs: list,
                   double_mut_seqs: list, reference_seq: str, sig_std_obs: int, sig_std_exp: int) -> list:
    """
    Creates a list of double mutation positions based on a positive epistatic effect and combinability (positive
    epistatic effect and additive single mutations)

    :param epistatic_scores: list of expected fitness scores for each double mutant
    :param obs_fitness_scores: list of observed fitness scores for each double mutant
    :param exp_fitness_stdvs: list of standard deviations of expected fitness score for list of double mutations
    :param obs_fitness_stdvs: list of standard deviations of observed fitness score for list of double mutations
    :param double_mut_seqs: list of sequences containing double mutations (with sufficient information)
    :param reference_seq: reference sequence of protein
    :param sig_std_obs: user defined multiple of standard deviation of observed fitness
    :param sig_std_exp: user defined multiple of standard deviation of expected fitness
    :return: pos_comb_double_mut_list: list of positive and combinable double mutation positions
    """

    candidates_tria_1 = []
    candidates_tria_2 = []
    mut_aa_list_1 = []
    mut_aa_list_2 = []

    # Create lists of double mutation positions dependent on positiveness and combinability
    for seq in range(len(double_mut_seqs)):
        # Ensure positive fitness effect and combinability
        if obs_fitness_scores[seq] > sig_std_obs * obs_fitness_stdvs[seq] and epistatic_scores[
            seq] > sig_std_exp * exp_fitness_stdvs[seq]:
            sequence = double_mut_seqs[seq]
            _, mut_pos, mut_aa = call_aa_simple(reference_seq, sequence)

            candidates_tria_1.append(mut_pos[0])
            candidates_tria_2.append(mut_pos[1])
            mut_aa_list_1.append(mut_aa[0])
            mut_aa_list_2.append(mut_aa[1])

    # Combine two lists into combined list of double mutation positions
    candidates_tria_1_np = np.array(candidates_tria_1)
    candidates_tria_2_np = np.array(candidates_tria_2)
    mut_aa_list_1_np = np.array(mut_aa_list_1)
    mut_aa_list_2_np = np.array(mut_aa_list_2)

    pos_comb_double_mut_list = np.stack(
        (candidates_tria_1_np, candidates_tria_2_np, mut_aa_list_1_np, mut_aa_list_2_np), axis=1)  # .tolist()

    return pos_comb_double_mut_list


def double_mut_pos_eps(exp_fitness_scores: list, obs_fitness_scores: list, double_mut_seq: list, ref_seq: str,
                       epistasis: Literal["positive", "negative", "none"], add_threshold: float,
                       pos_threshold: float) -> list:
    """
    Analyses the list of sequences of double mutants for positive, negative, or no epistasis and returns a list of
    double mutation positions for each sequence

    :param exp_fitness_scores: list of expected fitness scores of double mutant
    :param obs_fitness_scores: list of observed fitness scores of double mutant
    :param double_mut_seq: list of sequences of double mutants
    :param ref_seq: reference sequence of protein
    :param epistasis: 'positive', 'negative, 'none'
    :param add_threshold: threshold of additive effect
    :param pos_threshold: threshold of positive fitness
    :return: filtered_double_mut_pos_list: list of positions of double mutations filtered according to type and
    thresholds
    """

    if epistasis != 'positive' and epistasis != 'negative' and epistasis != 'none':
        raise ValueError("Only 'positive', 'negative', or 'none' allowed as input for :param epistasis")

    if add_threshold < 0:
        raise ValueError("The value of :param add_threshold must be greater equal 0")

    if pos_threshold < 0:
        raise ValueError("The value of :param pos_threshold must be greater equal 0")

    candidates_tria_1 = []
    candidates_tria_2 = []

    # Create lists of double mutation positions dependent on epistatic or additive effect
    for k in range(len(double_mut_seq)):
        # Positive epistasis associated with positive outcome
        if epistasis == 'positive' and pos_threshold > 0:
            if obs_fitness_scores[k] > (exp_fitness_scores[k] + add_threshold) and obs_fitness_scores[
                k] > pos_threshold:
                seq_i = double_mut_seq[k][0]
                _, mut_pos, _ = call_aa_simple(ref_seq, seq_i)

                candidates_tria_1.append(mut_pos[0])
                candidates_tria_2.append(mut_pos[1])
        # Positive epistasis
        elif epistasis == 'positive' and pos_threshold == 0:
            if obs_fitness_scores[k] > (exp_fitness_scores[k] + add_threshold):
                seq_i = double_mut_seq[k][0]
                _, mut_pos, _ = call_aa_simple(ref_seq, seq_i)

                candidates_tria_1.append(mut_pos[0])
                candidates_tria_2.append(mut_pos[1])
        # Negative epistasis
        elif epistasis == 'negative':
            if obs_fitness_scores[k] < (exp_fitness_scores[k] - add_threshold):
                seq_i = double_mut_seq[k][0]
                _, mut_pos, _ = call_aa_simple(ref_seq, seq_i)

                candidates_tria_1.append(mut_pos[0])
                candidates_tria_2.append(mut_pos[1])
        # No epistasis / additive effect
        elif epistasis == 'none':
            if (exp_fitness_scores[k] - add_threshold) <= obs_fitness_scores[k] <= (
                    exp_fitness_scores[k] + add_threshold):
                seq_i = double_mut_seq[k][0]
                _, mut_pos, _ = call_aa_simple(ref_seq, seq_i)

                candidates_tria_1.append(mut_pos[0])
                candidates_tria_2.append(mut_pos[1])

    # Combine two lists into combined list of double mutation positions
    candidates_tria_1_np = np.array(candidates_tria_1)
    candidates_tria_2_np = np.array(candidates_tria_2)

    filtered_double_mut_pos_list = np.stack((candidates_tria_1_np, candidates_tria_2_np), axis=1).tolist()

    return filtered_double_mut_pos_list


def epistatic_interaction_double_mutation(double_mut_positions: list, query_pos: int):
    """
    Analyzes a list of double mutation positions given a specific amino acid position to determine triangular
    epistatic interactions between amino acid positions

    :param double_mut_positions: n x 2 list of double mutation positions
    :param query_pos: amino acid position to analyze
    :return: epistatic_interaction_given_query: list of epistatic triangles given the queried amino acid position
    """

    # Convert to numpy arrayArray of combinations
    double_mut_positions = np.array(double_mut_positions)

    # Find query pairs
    idx_query_pos, _ = np.nonzero(double_mut_positions == query_pos)

    query_pairs = double_mut_positions[idx_query_pos, :]
    # print("Query pairs: ", query_pairs)

    # Find epistasis partner
    idx_1, idx_2 = np.nonzero(query_pairs != query_pos)

    epistasis_partner = query_pairs[idx_1, idx_2]
    # print("Epistasis partner: ", epistasis_partner)

    # List of epistasis partner interactions
    pairs_given_partner_list = np.zeros((1, 2))

    for partner_pos in range(len(epistasis_partner)):
        epistasis_partner_pos = epistasis_partner[partner_pos]
        idx_epistasis_partner_pos, _ = np.nonzero(double_mut_positions == epistasis_partner_pos)
        pairs_given_partner = double_mut_positions[idx_epistasis_partner_pos, :]
        pairs_given_partner_list = np.concatenate((pairs_given_partner_list, pairs_given_partner), axis=0)

    pairs_given_partner_list = pairs_given_partner_list[1:, :]
    # print("Pairs given partner: ", pairs_given_partner_list)

    # Create list of combinations of partners to compare with array
    poss_interaction_comb = np.zeros((1, 2))

    if len(epistasis_partner) == 2:
        poss_interaction_comb = np.concatenate((poss_interaction_comb, np.expand_dims(epistasis_partner, axis=0)),
                                               axis=0)
        # add other permutation
        epistasis_partner_perm = np.array([[epistasis_partner[1], epistasis_partner[0]]])
        poss_interaction_comb = np.concatenate((poss_interaction_comb, epistasis_partner_perm), axis=0)

        poss_interaction_comb = poss_interaction_comb[1:, :]

    else:

        for pos_1 in range(len(epistasis_partner)):
            for pos_2 in range(len(epistasis_partner)):
                if epistasis_partner[pos_1] != epistasis_partner[pos_2]:
                    comb = np.expand_dims(np.array([epistasis_partner[pos_1], epistasis_partner[pos_2]]), axis=0)

                    poss_interaction_comb = np.concatenate((poss_interaction_comb, comb), axis=0)

        poss_interaction_comb = poss_interaction_comb[1:, :]

    # print("Possible interaction combinations: ", poss_interaction_comb)

    # Compare

    epistatic_interaction_given_query = []

    for pair in range(len(pairs_given_partner_list)):
        for comb in range(len(poss_interaction_comb)):
            if pairs_given_partner_list[pair, 0] == poss_interaction_comb[comb, 0] and pairs_given_partner_list[
                pair, 1] == poss_interaction_comb[comb, 1]:
                epistatic_interaction_given_query.append(pairs_given_partner_list[pair, :].tolist())

    epistatic_interaction_given_query = np.array(epistatic_interaction_given_query)

    # Add query position to list of epistatic interaction if not empty
    if epistatic_interaction_given_query.size > 0:
        epistatic_interaction_given_query = np.unique(epistatic_interaction_given_query, axis=0)
        query_vector = query_pos * np.ones(len(epistatic_interaction_given_query)).reshape(-1, 1)
        epistatic_interaction_given_query = np.column_stack((query_vector, epistatic_interaction_given_query)).squeeze()
        epistatic_interaction_given_query = sorted(epistatic_interaction_given_query.tolist())

    # print("Epistatic interaction triangle: ", epistatic_interaction_given_query)
    # print(" ")

    return epistatic_interaction_given_query


def epistasis_graph(double_mut_positions: list) -> nx.Graph:
    """
    Creates an epistasis double_mut_epistasis_graph given a list of positions of double mutation.

    :param double_mut_positions: list of lists of positions of double mutations
    :return: double_mut_epistasis_graph: epistasis double_mut_epistasis_graph
    """
    graph = nx.Graph()

    edges_list = []
    for pair in double_mut_positions:
        edges_list.append(tuple(pair))

    graph.add_edges_from(edges_list)

    return graph


def construct_structural_epistasis_graph(double_mutant_edges: list, filter_threshold: int,
                                         distance_matrix: np.ndarray, zero_edge_nodes: bool = True) -> nx.Graph:
    """
    Given interaction edges and a distance matrix, construct a structural epistasis graph

    :param double_mutant_edges: list of interactions
    :param filter_threshold: integer, nodes above this threshold are incoporated in the graph
    :param distance_matrix: numpy array of min dimer distance matrix
    :param zero_edge_nodes: boolean variable, indicates if nodes without edges should be included
    :return: structural_epistasis_graph: networkX graph
    """
    # Filter elements
    double_mutant_edges = np.array(double_mutant_edges)
    unique_values, counts = np.unique(double_mutant_edges, return_counts=True)
    sorted_indices = np.argsort(counts)
    unique_values_sorted_by_counts = unique_values[sorted_indices][::-1]
    counts = np.sort(counts)[::-1]

    filtered_count_indices = np.argwhere(counts > filter_threshold)
    filtered_values = unique_values_sorted_by_counts[filtered_count_indices]

    # Only keep edges of those highly connected nodes
    filtered_pos_comb_mut_edges = []
    for edge_cand_idx in range(0, len(double_mutant_edges)):
        edge_cand = double_mutant_edges[edge_cand_idx]
        if np.any(filtered_values == edge_cand[0]) or np.any(filtered_values == edge_cand[1]):
            filtered_pos_comb_mut_edges.append(edge_cand.tolist())

    # Convert to numpy array
    filtered_pos_comb_mut_edges = np.array(filtered_pos_comb_mut_edges)

    # Exclude stop codon
    if 291 in filtered_pos_comb_mut_edges:
        filtered_pos_comb_mut_edges_xstop = []
        for i in range(0, len(filtered_pos_comb_mut_edges)):
            if filtered_pos_comb_mut_edges[i, 0] != 291 and filtered_pos_comb_mut_edges[i, 1] != 291:
                filtered_pos_comb_mut_edges_xstop.append(filtered_pos_comb_mut_edges[i].tolist())
        filtered_pos_comb_mut_edges = filtered_pos_comb_mut_edges_xstop

    # Extract unique nodes
    unique_filtered_nodes = np.unique(filtered_pos_comb_mut_edges)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(distance_matrix)
    Y_red = pca.transform(distance_matrix)

    structural_epistasis_graph = nx.Graph()
    for node in range(1, 291):
        if zero_edge_nodes:
            structural_epistasis_graph.add_node(node, pos=(Y_red[node - 1, 0], Y_red[node - 1, 1]))
        else:
            if np.any(unique_filtered_nodes == node):
              structural_epistasis_graph.add_node(node, pos=(Y_red[node - 1, 0], Y_red[node - 1, 1]))

    edges_list = []
    for pair in filtered_pos_comb_mut_edges:
        edges_list.append(tuple(pair))

    structural_epistasis_graph.add_edges_from(edges_list)

    return structural_epistasis_graph


def epistatic_triangles(higher_order_mut_positions: list) -> list:
    """
    Given a list of higher order mutation positions, all epistatic triangular structures are extracted as a list
    :param higher_order_mut_positions: list of higher order mutation positions
    :return: epistatic_triangle_list: returns a list of epistatic triangles
    """
    # Determine all epistatic triangles for all AA positions
    AA_pos_list = np.unique(higher_order_mut_positions)

    epistatic_triangle_list = []

    for AA_pos in range(len(AA_pos_list)):
        epistatic_triangle = epistatic_interaction_double_mutation(higher_order_mut_positions, AA_pos_list[AA_pos])
        if len(epistatic_triangle) > 0:
            # Append list of triangles with the new triangles one by one
            if any(isinstance(j, list) for j in epistatic_triangle):
                for triangle in range(0, len(epistatic_triangle)):
                    epistatic_triangle_list.append(sorted(epistatic_triangle[triangle]))
            # Append list of triangles with the one new triangle
            else:
                epistatic_triangle_list.append(sorted(epistatic_triangle))

    # List of epistatic triangles
    epistatic_triangle_list = np.unique(np.array(epistatic_triangle_list), axis=0).astype(int).tolist()

    return epistatic_triangle_list
