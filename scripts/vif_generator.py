
from Bio import SeqIO 
import argparse
import multiprocessing
import re
from skbio.alignment import StripedSmithWaterman
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="""Variant identity file generator: Creating VIF file for DiMSum pipeline from Nanopore polishing output""")

parser.add_argument('-T', '--threads', type=int, default=0, help='Number of threads to execute in parallel. Defaults to CPU count.')
parser.add_argument('--probe_UMI_down', help='A short sequence (eg 14 bp) beginning 4 bp downstream of the UMI in fasta format.', required=True)
parser.add_argument('--probe_UMI_up', help='A short sequence (eg 14 bp) ending 4 bp upstream of the UMI in fasta format.', required=True)
parser.add_argument('--probe_frame_down', help='A short sequence (eg 10 bp) flanking the frame (downstream) in fasta format.', required=True)
parser.add_argument('--probe_frame_up', help='(eg 10 bp) flanking the frame (upstream) in fasta format', required=True)
parser.add_argument('--clusterfolder', help='path to clusterfolder', required=True)

#extension for parser and threads
args = parser.parse_args()
threads = args.threads
if threads == 0:
    threads = multiprocessing.cpu_count()
    
#set up input variables
probe_UMI_down_file = args.probe_UMI_down
probe_UMI_up_file = args.probe_UMI_up
clusterfolder = args.clusterfolder
probe_frame_down_file = args.probe_frame_down
probe_frame_up_file = args.probe_frame_up 


#set up probe seq and sequence seq variables 

probe_UMI_down = SeqIO.read(probe_UMI_down_file, "fasta")
probe_UMI_down_seq = str(probe_UMI_down.seq)

probe_UMI_up = SeqIO.read(probe_UMI_up_file, "fasta")
probe_UMI_up_seq = str(probe_UMI_up.seq)

probe_frame_down = SeqIO.read(probe_frame_down_file, "fasta")
probe_frame_down_seq = str(probe_frame_down.seq)

probe_frame_up = SeqIO.read(probe_frame_up_file, "fasta")
probe_frame_up_seq = str(probe_frame_up.seq)

#________________________________________________________________________________________________________________________________
#set up function that finds start and end of the UMI depending on probes upsteam and downstream of the UMI 

def search_UMI_down(probe_UMI_down, sequence):
    no_UMI_down_probe = False
    if re.search(probe_UMI_down, sequence) is None:
        no_UMI_down_probe = True
        return(no_UMI_down_probe)
    else:
        for match in re.finditer(probe_UMI_down, sequence):
            UMI_end = match.start() - 4
            return(UMI_end)
        

#_____________

#upstream: same as downstream in the other direction. 

def search_UMI_up(probe_UMI_up, sequence):
    no_UMI_up_probe = False
    if re.search(probe_UMI_up, sequence) is None:
        no_UMI_up_probe = True
        return(no_UMI_up_probe)
    else:
        for match in re.finditer(probe_UMI_up, sequence):
            UMI_start = match.end() + 4
            return(UMI_start)
        

#________________________________________________________________________________________________________________________________
#set up function that extracts the UMI 

def extract_UMI(sequence):
    start = search_UMI_up(probe_UMI_up_seq, sequence)
    end = search_UMI_down(probe_UMI_down_seq, sequence)
    if start is True and end is True:
        return("UMI_not_detectable")
    if start is True:
        start = end - 65
        UMI = sequence[start:end]
        return(UMI)
    if end is True:
        end = start + 65
        UMI = sequence[start:end]
        return(UMI)
    else: 
        UMI = sequence[start:end]
        return(UMI)

#_____________
#set up function that looks up the position of the frame start

def search_frame_up(probe_frame_up, sequence):
    no_frame_up_probe = False
    if re.search(probe_frame_up, sequence) is None:
        no_frame_up_probe = True
        return(no_frame_up_probe)
    else:
        for match in re.finditer(probe_frame_up, sequence):
            frame_start = match.end()
            #print(frame_start)
            return(frame_start)
    
        
#_____________
#set up function that looks up the position of the frame end

def search_frame_down(probe_frame_down, sequence):
    no_frame_down_probe = False
    if re.search(probe_frame_down, sequence) is None:
        no_frame_down_probe = True
        return(no_frame_down_probe)
    else:
        for match in re.finditer(probe_frame_down, sequence):
            frame_end = match.start()
            return(frame_end)

#________________________________________________________________________________________________________________________________
#set up function that extracts the frame sequence


def extract_frame(sequence):
    start = search_frame_up(probe_frame_up_seq, sequence)
    end = search_frame_down(probe_frame_down_seq, sequence)
    
    if start is True and end is True:
        return("frame_not_detectable")
    if start is True:
        start = end - 873
        frame_seq = sequence[start:end]
        return(frame_seq)
    if end is True:
        end = start + 873
        frame_seq = sequence[start:end]
        return(frame_seq)
    else:
        frame = sequence[start:end]
        return(frame)
        
#________________________________________________________________________________________________________________________________
#loop goes through target sequences, extracts their name and sequence, extracts UMI and the frame and generates .txt variant idenitfyer file

#create .csv file with barcode and mutations
with open('VIF.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["barcode", "variant"])


#generate list of clusters_consensus files in given clusterfolder  

clusters = os.listdir(clusterfolder)

#set up counting variables for counting how often no UMI was recoginzed
no_UMI_count = 0
no_frame_count = 0

#counts not readable seqIO reads
NoSeqIOread = 0

for c in clusters:
    
    #very few reads are not readable by SeqIO -> this exception handels this (is counted by NoSeqIOread, should be very seldom: once in X46 Dataset)
    try:
        target_file = SeqIO.read(f"{clusterfolder}/{c}", "fasta") #read target consensus sequence
        target = str(target_file.seq)
    except UnicodeDecodeError:
        print("error")
        NoSeqIOread += 1
    
    cname = c[:-11] #erases the .fasta and _cons of the clusters and gives "cluster_XXXX" as cname
    print(cname)
    
    #extract UMI from target
    UMI = extract_UMI(target)
    frame = extract_frame(target)
    
    if UMI == "UMI_not_detectable": #if flexible solve (with fixed difference between up and down was not succesful, cluster is counted as not)
        no_UMI_count += 1
        print("patternmatch (UMI) not succesful")
        continue #if UMI not detectable it is not written in the .txt file

    if frame == "frame_not_detectable": #if flexible solve (with fixed difference between up and down was not succesful, cluster is counted as not)
        no_frame_count += 1
        print("patternmatch (frame) not succesful")
        continue
    
    with open('VIF.txt', 'a', newline='') as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow([f"{UMI}", f"{frame}"])

print("\n")
print(f"total number of undetected UMIs: {no_UMI_count}")
print(f"total number of undetected frames: {no_frame_count}")
    
        
