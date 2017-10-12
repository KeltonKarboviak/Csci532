# Lab 2 Parallel DNA Sequencing with MPI

### Download the seed files

Run the following commands to download and unzip the seed files which we need to run the dna sequencing program:

```bash
mkdir seeds
cd seeds
wget -c ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR192/ERR192339/ERR192339.fastq.gz
gzip -d ERR192339.fastq.gz
```

There are more seed files available at the same FTP site (e.g., ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR192/ERR192340/ERR192340.fastq.gz
)

**Note:** The seed files are very large (~3 GB) in file size, so it may take sometime to download and unzip them


### Download the chromosome files

Run the following commands to download and unzip the chromosome files which are required by this program:

```bash
mkdir chromosomes
cd chromosomes
wget -c http://hgdownload.cse.ucsc.edu/goldenPath/mm10/chromosomes/chr1.fa.gz
gzip -d chr1.fa.gz
```

There are more chromosome files avaiable at the same site (e.g., http://hgdownload.cse.ucsc.edu/goldenPath/mm10/chromosomes/chr2.fa.gz)


### Compilation

Run the following commands to compile and run the parallel dna sequencing program:

```bash
mkdir build
cd build
mpicxx ../dna_sequencer.cxx -o dna_sequencer -std=c++11
mpirun -np 4 dna_sequencer --seeds ../seeds/*.fastq --chrs ../chromosomes/*.fa --output out.txt
```
