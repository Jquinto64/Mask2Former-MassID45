#!/bin/bash
#SBATCH -p rtx6000         # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:4       # request GPU(s)
#SBATCH -c 8               # number of CPU cores
#SBATCH --mem=20G          # memory per node
#SBATCH --array=0          # array value (for running multiple seeds, etc)
#SBATCH --qos=m
#SBATCH --time=12:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=mask2former_lifeplan_b_512_sahi_tiled_v9_R50_keep_cutoff_5_epochs_one_cycle_lr_5e-5_color_augs_15k_iters # customize this for your project
#SBATCH --exclude=gpu177,gpu132,gpu170

source ~/.bashrc
source activate md3
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# FOR SAHI DATASETS
TILE_SIZE=512
python train_net.py --num-gpus 4 \
--resume \
--exp_id ${TILE_SIZE} \
--config-file /h/jquinto/Mask2Former/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
--dataset_path /h/jquinto/Mask2Former/datasets/lifeplan_${TILE_SIZE}/ \
OUTPUT_DIR output_lifeplan_b_${TILE_SIZE}_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters \
DATASETS.TRAIN "(\"lifeplan_${TILE_SIZE}_train\",)" \
DATASETS.TEST "(\"lifeplan_${TILE_SIZE}_valid\",)"  \
MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \