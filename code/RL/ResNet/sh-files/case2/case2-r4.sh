#!/bin/sh
#SBATCH --job-name=c2-r4-10
#SBATCH -N 1
#SBATCH -n 24    ##24 cores(of 48) so you get 1/2 of machine RAM ( 192 GB total)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output IS_job_%j.out
#SBATCH --error IS_job_%j.err
#SBATCH -p dgx_aic,AI_Center

module load python3/anaconda/2021.07
source /work/bharath/env/infoSpread_torch/bin/activate

cd /work/bharath/InfoSpread-new/temp-new/case2/case2-r4/
export PYTHONPATH="/work/bharath/InfoSpread-new/temp-new/case2/case2-r4":$PYTHONPATH

# 10 nodes, 50 steps, source trust 1
## n=1
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 1 --actions 1
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 1 --actions 2
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 1 --actions 3

## n=2
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 2 --actions 1
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 2 --actions 2
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 2 --actions 3

## n=3
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 3 --actions 1
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 3 --actions 2
python train_batch-latest.py --max_nodes 10 --max_steps 50 --episodes 300 --batch_size 100 --states_per_episode 200 --source_trust 1 --infected_nodes 3 --actions 3

python inference.py --nodes 10