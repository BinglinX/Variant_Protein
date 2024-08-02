#$ -l tmem=2G,h_rt=10:00:00,gpu=true
#$ -S /bin/bash
#$ -N Variant_Protein
#$ -j y
#$ -R y
#$ -cwd
#$ -pe gpu 1

hostname
date

source /share/apps/source_files/python/python-3.8.5.source
module load python/3.8.5

source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate go
which python3
python3 --version

python3 main.py -cn base
