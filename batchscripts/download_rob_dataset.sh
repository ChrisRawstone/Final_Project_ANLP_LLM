#!/bin/sh
#BSUB -J Train
#BSUB -o logs/Train%J.out
#BSUB -e logs/Train%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2G]"
#BSUB -W 04:00
#BSUB -N

#BSUB 
# end of BSUB options


# download the dataset https://itu.dk/people/robv/sprogmodel_data.tar.gz
wget https://itu.dk/people/robv/sprogmodel_data.tar.gz

