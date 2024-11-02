#!/bin/bash
####
#SBATCH --job-name=prepare_3d2d
### partition: {week, day, month}
#SBATCH --partition=week
#SBATCH --cpus-per-task=32
# #SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time 7-0
# #SBATCH --nodelist=node-gpu-03
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out

# GPU-3 
source /home/rawangq1/git_wq/SSL_veronika/.venv/bin/activate

python /home/rawangq1/git_wq/SSL_veronika/utils/mp_3Dslice2D.py /mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/sim/ 5
## srun exec/run_gpu3.sh 3
## sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,Elapsed,AveCPU,AllocCPUS,AllocGRES,ReqMem

echo "Finished"
