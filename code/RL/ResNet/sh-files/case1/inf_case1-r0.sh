#!/bin/sh
#SBATCH --job-name=inf_c1r0
#SBATCH -N 1
#SBATCH -n 12    ##24 cores(of 48) so you get 1/2 of machine RAM ( 192 GB total)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output IS_job_%j.out
#SBATCH --error IS_job_%j.err
#SBATCH -p dgx_aic,AI_Center

module load python3/anaconda/2021.07
source /work/bharath/env/infoSpread_torch/bin/activate

cd /work/bharath/InfoSpread-new/temp-new/case1/case1-r0/
export PYTHONPATH="/work/bharath/InfoSpread-new/temp-new/case1/case1-r0":$PYTHONPATH

python inference-new.py --nodes 10
python inference-new.py --nodes 25
python inference-new.py --nodes 50