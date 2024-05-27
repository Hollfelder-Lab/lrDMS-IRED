import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import os
import argparse


parser = argparse.ArgumentParser(description="""Helper script for clustering UMIs with MMseqs2.""")

parser.add_argument('clusterfile', help="Provide MMseqs2 output file (as .tsv).")
parser.add_argument('--sizethreshs', help="Low and high thresholds for cluster sizes. Defaults to 1 1000.", nargs=2, default=[1, 1000], type=int)
parser.add_argument('--splitfile', help="If a read file (fasta/fastq) is provided here, individual fasta files per cluster will be generated.", type=str)
parser.add_argument('--similarityfile', help='If a read file (fasta/fastq) is provided here, <similaritysamplesize> clusters will be analysed for internal cluster similarity.', type=str)
parser.add_argument('--similaritysamplesize', help="Sample size for within cluster similarity analysis. Defaults to 500.", default=500, type=int)

args = parser.parse_args()
name = args.clusterfile
sizeLOW, sizeHIGH = args.sizethreshs
similaritysamplesize = args.similaritysamplesize
readsfile = args.splitfile
umifile = args.similarityfile

### LOAD IN CLUSTER DATA FROM MMSEQ2 OUTPUT
print(f"Loading {name}")
clusters = np.loadtxt(name, delimiter="\t", dtype=str)
name = name[:-4] #remove file ending to use as prefix later

###
# Count cluster sizes and create cluster list
oldclus = ""
clusterlst = []
for i in range(len(clusters)):
    if clusters[i][0] != oldclus:   #If new cluster
        oldclus = clusters[i][0]
        #Create a new cluster entry
        clusterlst.append([])
    clusterlst[-1].append(clusters[i][1])   #add cluster member to current cluster

print(f"\nNumber of clusters: {len(clusterlst)}")
N_largeclus = 0
for clus in clusterlst:
    if sizeLOW <= len(clus) <= sizeHIGH:
        N_largeclus += 1

clussizes = [len(clus) for clus in clusterlst]
print(f"Mean cluster size: {np.average(clussizes)}")
print(f"Median cluster size: {np.median(clussizes)}")
hist_clussize = np.repeat(clussizes, clussizes)
med = np.median(hist_clussize)
print(f"Median number of sequences per cluster: {med}")
print(f"Clusters with {sizeLOW} <= members <= {sizeHIGH}: {N_largeclus}")



###
# Plotting cluster size distribution
print("\nPlotting cluster size distributions...")
binwidth = 1
fig = plt.figure()
plt.hist(hist_clussize, bins=np.arange(min(hist_clussize), max(hist_clussize)+binwidth, binwidth))
textstr = f"Median: {med}"
plt.text(0.85, 0.85, textstr, transform=fig.transFigure, ha='right')
plt.xlabel("Cluster size")
plt.ylabel("Number of sequences")
plt.savefig(f"{name}_full_clustersizes_sequences.pdf", bbox_inches='tight')
plt.xlim([sizeLOW,sizeHIGH])
plt.savefig(f"{name}_zoom_clustersizes_sequences.pdf", bbox_inches='tight')




###
# Writing clusters to file
if readsfile is not None:
    print("\nWriting cluster files...")
    filename, extension = readsfile.rsplit(".", 1)  #splits the first orrucence from the end
    reads = SeqIO.index(readsfile, extension)
    print(f"Reads loaded: {len(reads)}")

    output_folder = f"{name}_clusterfiles"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 

    count = 0
    for clusids in clusterlst:
        if sizeLOW <= len(clusids) <= sizeHIGH:
            clus_members = [reads[seqid] for seqid in clusids]
            fname = output_folder + "/cluster_" + str(count) + ".fasta" #FASTA OR FASTQ?
            SeqIO.write(clus_members, fname, "fasta")
            count += 1
    print("Cluster files written")



# Calculating in cluster similarities
if umifile is not None:
    from skbio.alignment import StripedSmithWaterman
    filename, extension = umifile.rsplit(".", 1)  #splits the first orrucence from the end
    reads = SeqIO.index(umifile, extension)
    print("\nCalculating internal cluster similarities...")
    print(f"Reads loaded: {len(reads)}\n")

    similarities = []
    selected_sizes = []
    run = 0
    for clus in np.random.choice(clusterlst, len(clusterlst), replace=False):      ### Randomize cluster order for similarity analysis, because the cluster order might be biased
        if sizeLOW <= len(clus) <= sizeHIGH:    ### calc only for selected clusters
            count = 0
            total_score = 0
            for i in range(len(clus)):
                query = StripedSmithWaterman(str(reads[clus[i]].seq), score_only=True)
                for j in range(i+1, len(clus)):
                    aln = query(str(reads[clus[j]].seq))
                    score = aln.optimal_alignment_score
                    total_score += score
                    count += 1
            similarities.append(total_score / count)
            selected_sizes.append(len(clus))
            run += 1
            print(f"Cluster {run}: {len(clus)} members. {total_score / count:.2f} average internal similarity   ", end='\r')
            if run == similaritysamplesize:
                break
        
    print(f"\nAverage similarity within all clusters: {np.average(similarities):.2f}")

    plt.figure()
    plt.hist(similarities, bins=20, color="black")
    plt.ylabel("Count")
    plt.xlabel("internal cluster similarity")
    plt.savefig(f"{name}_hist_similarities.pdf", bbox_inches='tight')

    plt.figure()
    plt.scatter(similarities, selected_sizes)
    plt.xlabel("internal cluster similarity")
    plt.ylabel("cluster size")
    plt.savefig(f"{name}_scatter_sizeVSsimilarity.pdf", bbox_inches='tight')


