# Script for UMI extraction from basecalled Nanopore reads

from Bio import SeqIO
from Bio.Seq import Seq
from skbio.alignment import StripedSmithWaterman
import multiprocessing
import itertools
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="""Main script for UMI-linked consensus sequencing.""")

parser.add_argument(
    "-T", "--threads", type=int, default=0, help="Number of threads to execute in parallel. Defaults to CPU count."
)
parser.add_argument("-v", "--version", action="version", version="1.1.1")
parser.add_argument("-i", "--input", help="Provide basecalled reads in fastq format.", required=True)
parser.add_argument("-o", "--output", help="Specify the name of the UMI output fasta file.", required=True)
parser.add_argument("--probe", help="A short sequence (eg 50 bp) adjacent to the UMI in fasta format.", required=True)
parser.add_argument(
    "--umi_loc",
    help="Location of UMI in reference to the probe. Upstream (up) or downstream (down).",
    choices=("up", "down"),
    required=True,
)
parser.add_argument("--umi_len", help="Length of the UMI to be extracted.", type=int, required=True)
parser.add_argument(
    "--min_probe_score",
    help="Minimal alignment score of probe for processing. Defaults to length of probe sequence.",
    type=int,
    default=0,
)


# Parse arguments
args = parser.parse_args()
threads = args.threads
if threads == 0:
    threads = multiprocessing.cpu_count()

input_file = args.input
probe_file = args.probe
output_file = args.output
umi_loc = args.umi_loc
umi_len = args.umi_len


def prnt_aln(aln):
    print("Score: %d" % aln.optimal_alignment_score)
    print(aln.aligned_target_sequence)  # READ
    print(aln.aligned_query_sequence)  # Search
    s = aln.target_begin
    d = aln.target_end_optimal
    l = len(aln.target_sequence)
    print("Left Border: %d\t Right Border: %d \n" % (s, l - d))


def extract_left(aln):
    umi_end = aln.target_begin
    umi_begin = umi_end - umi_len
    return umi_begin, umi_end


def extract_right(aln):
    umi_begin = aln.target_end_optimal + 1
    umi_end = umi_begin + umi_len
    return umi_begin, umi_end


# Set up variables

proberec = SeqIO.read(probe_file, "fasta")
probe_minaln_score = args.min_probe_score
if probe_minaln_score == 0:
    probe_minaln_score = len(proberec.seq)

probe_fwd = str(proberec.seq)
probe_rev = str(proberec.reverse_complement().seq)
query_fwd = StripedSmithWaterman(probe_fwd)
query_rev = StripedSmithWaterman(probe_rev)

umi_list = []
count, bad_aln, short_aln, success = 0, 0, 0, 0
for record in SeqIO.parse(input_file, "fastq"):
    rec_seq = str(record.seq)
    alnF = query_fwd(rec_seq)
    alnR = query_rev(rec_seq)
    scoreF = alnF.optimal_alignment_score
    scoreR = alnR.optimal_alignment_score
    # Check basic alignment score
    if scoreF > probe_minaln_score or scoreR > probe_minaln_score:
        if scoreF > scoreR:  # Target in fwd orientation
            # Get umi location
            if umi_loc == "down":
                umi_begin, umi_end = extract_right(alnF)  # FWD of downstream is right
            elif umi_loc == "up":
                umi_begin, umi_end = extract_left(alnF)  # FWD of upstream is left
            # append to UMI to record list
            if umi_end < len(alnF.target_sequence) and umi_begin > 0:  # UMI could be out of bounds
                umi = alnF.target_sequence[umi_begin:umi_end]
                record.letter_annotations = {}  # remove qality
                record.seq = Seq(umi)
                umi_list.append(record)
                success += 1
            else:
                short_aln += 1
        else:  # Target in rev orientation
            # Get umi location
            if umi_loc == "down":
                umi_begin, umi_end = extract_left(alnR)  # REV of downstream is left
            elif umi_loc == "up":
                umi_begin, umi_end = extract_right(alnR)  # REV of upstream is right
            # append to UMI to record list
            if umi_begin > 0 and umi_end < len(alnR.target_sequence):  # UMI could be out of bounds
                umiR = alnR.target_sequence[umi_begin:umi_end]
                record.letter_annotations = {}  # remove qality
                record.seq = Seq(umiR).reverse_complement()
                umi_list.append(record)
                success += 1
            else:
                short_aln += 1
    else:
        bad_aln += 1
    count += 1

    if count % 1000 == 0:
        print("%d sequences analysed" % count, end="\r")

print("%d sequences analysed" % count)
print("\nUMIs extracted: %d" % success)
print("Discarded: %.2f%%:" % ((bad_aln + short_aln) / count * 100))
print("Bad alignment: %d" % bad_aln)
print("Incomplete UMI: %d" % short_aln)

SeqIO.write(umi_list, output_file, "fasta")
