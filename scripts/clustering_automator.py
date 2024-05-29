#Builds an sh script that can be used in the terminal

import os

clusterfolder = f"clusters_clusterfiles"
reference = "wt.fasta"
outputfolder = f"test"
scriptname = f"auto.sh"

if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)

clus = os.listdir(clusterfolder)
print(f"Analysing all {len(clus)} files in {clusterfolder}")

f = open(scriptname, "w")


#write cluster commands
for c in clus:
    cname = c[:-6]
    stepr1 = f"minimap2 -x ava-ont {reference} {clusterfolder}/{c} > aln_{cname}.paf" #perform minimap for racon
    stepr2 = f"racon {clusterfolder}/{c} aln_{cname}.paf {reference} > racon_{cname}.fasta" #perform racon 
    step1 = f"mini_align -i {clusterfolder}/{c} -r racon_{cname}.fasta -p {cname}.minialn -m"  # -f not necessary to rebuild reference index every time     #output: .minialn.bam .minialn.bai
    step2 = f"medaka consensus {cname}.minialn.bam {cname}.probs --model r103_min_high_g360"      #add model etc       #output .probs
    step3 = f"medaka variant racon_{cname}.fasta {cname}.probs {cname}.vcf"                    
    step4 = f"python3 vcf2fasta.py -v {cname}.vcf -r racon_{cname}.fasta -o {outputfolder}/{cname}_cons.fasta" #python command for vcf2fasta script
    step5 = f"rm {cname}.minialn.bam {cname}.minialn.bam.bai {cname}.probs racon_{cname}.fasta.fai racon_{cname}.fasta.mmi aln_{cname}.paf racon_{cname}.fasta {cname}.vcf" #remove unnecesary files 
    f.write(stepr1+"\n")
    f.write(stepr2+"\n")
    f.write(step1+"\n")
    f.write(step2+"\n")
    f.write(step3+"\n")
    f.write(step4+"\n")
    f.write(step5+"\n")
print(stepr1)
print(stepr2)  
print(step1)
print(step2)
print(step3)
print(step4)
print(step5)
print()


f.close()

