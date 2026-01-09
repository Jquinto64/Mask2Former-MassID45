#!/bin/bash
#SBATCH --gres=gpu:l40s:2   # request GPU(s)
#SBATCH --cpus-per-task=8   # number of CPU cores
#SBATCH --mem=20G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --time=12:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=mask2former_lifeplan_b_512_sahi_tiled_v9_R50_keep_cutoff_5_epochs_one_cycle_lr_5e-5_color_augs_15k_iters # customize this for your project
#SBATCH --exclude=gpu177,gpu132,gpu170

ENV_NAME=mask2former
module load StdEnv/2020
module load python/3.10.2
module load cuda/11.8.0
module load opencv/4.8.0
source /home/jquinto/projects/aip-gwtaylor/jquinto/virtualenvs/$ENV_NAME/bin/activate

# Debugging outputs
pwd
python --version
pip freeze

# FOR SAHI DATASETS
TILE_SIZE=512
python train_net.py --num-gpus 2 \
--resume \
--exp_id ${TILE_SIZE} \
--config-file /h/jquinto/Mask2Former/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
--dataset_path /h/jquinto/Mask2Former/datasets/lifeplan_${TILE_SIZE}/ \
OUTPUT_DIR output_lifeplan_b_${TILE_SIZE}_sahi_tiled_v9_R50_1024_one_cycle_lr_5e-5_colour_augs_15k_iters \
DATASETS.TRAIN "(\"lifeplan_${TILE_SIZE}_train\",)" \
DATASETS.TEST "(\"lifeplan_${TILE_SIZE}_valid\",)"  \
MODEL.WEIGHTS /h/jquinto/Mask2Former/model_final_3c8ec9.pkl \