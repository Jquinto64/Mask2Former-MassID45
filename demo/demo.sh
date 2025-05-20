#!/bin/bash
#SBATCH -p t4v2             # partition: should be gpu on MaRS, and a40, t4v1, t4v2, or rtx6000 on Vaughan (v)
#SBATCH --gres=gpu:4      # request GPU(s)
#SBATCH -c 32              # number of CPU cores
#SBATCH --mem=16G           # memory per node
#SBATCH --array=0           # array value (for running multiple seeds, etc)
#SBATCH --qos=m2
#SBATCH --time=8:00:00
#SBATCH --output=slogs/%x_%A-%a_%n-%t.out
                            # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                            # Note: You must manually create output directory "slogs" 
#SBATCH --open-mode=append  # Use append mode otherwise preemption resets the checkpoint file
#SBATCH --job-name=mask2former_lifeplan_b_256_tiled_v5_t4v2_demo # customize this for your project
source ~/.bashrc
source activate md3
module load cuda-11.3

SEED="$SLURM_ARRAY_TASK_ID"

# Debugging outputs
pwd
which conda
python --version
pip freeze

# The "${@}" allows us to pass through arguments at the end of the sbatch command to our script, main.py.
python demo.py --config-file /h/jquinto/Mask2Former/configs/lifeplan/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --input /h/jquinto/GRLLN6_256/*.png --output /h/jquinto/Mask2Former/demo/GRLLN6_256_v5 --opts MODEL.WEIGHTS /h/jquinto/Mask2Former/output/model_0001999.pth
