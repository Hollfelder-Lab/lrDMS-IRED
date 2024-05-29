from allel import read_vcf
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(
    description="Simple vcf2fasta tool for generating fasta files from medaka variants vcf files"
)
parser.add_argument("-v", "--vcf", help="VCF file.", required=True)
parser.add_argument("-r", "--reference", help="Reference sequence fasta.", required=True)
parser.add_argument("-o", "--output", help="Name output file if you want a fasta.", required=False)
parser.add_argument("--min_qual", help="Quality threshold for filtering.", required=False, type=float, default=0.0)


# Arguments
args = parser.parse_args()
min_qual = args.min_qual
vcf_file = args.vcf
reference_file = args.reference
output_file = args.output


def mutate_reference(muts, refstr):
    ref = list(
        refstr
    )  # python strings are immutable. For convenience work with the reference as list and at the end convert back to string.
    # Checks
    # 1: Positions within ref
    mut_positions = [x[1] for x in muts]
    if len(mut_positions) > 0:
        if max(mut_positions) > len(ref):
            raise Exception(f"Position not within reference: {mut_positions}")
    # 2: WT position correct
    seq_wt = [x[0][0] for x in muts]
    ac_wt = [ref[x[1] - 1] for x in muts]
    assert seq_wt == ac_wt, "Reference sequences do not match"

    # Generate variant gene sequence
    variant = ref.copy()
    for mut in reversed(muts):
        pos = mut[1] - 1
        variant[pos] = mut[2]

    return "".join(variant)


# Load vcf and reference sequenc
vcf = read_vcf(vcf_file, fields="*")
reference = SeqIO.read(reference_file, "fasta")
refstr = str(reference.seq).upper()

# Extract mutations
muts = []
if vcf is not None:
    for imut in range(len(vcf["variants/QUAL"])):
        ref = vcf["variants/REF"][imut]
        pos = vcf["variants/POS"][imut]
        alt = vcf["variants/ALT"][imut][0]
        if vcf["variants/QUAL"][imut] > min_qual:
            mut = [ref, pos, alt]
            muts.append(mut)
str_muts = ["".join(str(e) for e in v) for v in muts]
print(f"List of mutations extracted:\n{str_muts}")
# Generate mutated reference
variant_sequence = mutate_reference(muts, refstr)
print(f"Mutated reference sequence:\n{variant_sequence}")
# Save to fasta
if output_file is not None:
    record = SeqRecord(Seq(variant_sequence), id=f"consensus_of_{vcf_file}", description="")
    SeqIO.write(record, output_file, "fasta")
    print(f"Fasta written to {output_file}.")
