#!/bin/bash
####
#SBATCH --job-name=prepare_sim
### partition: {week, day, month}
#SBATCH --partition=day
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time 1-0
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out

# GPU-3 
source /home/rawangq1/.venv/bin/activate

python /home/rawangq1/git_wq/SSL_veronika/utils/mp_make_motion_dataset.py /mnt/qdata/share/raecker1/ukbdata_70k/abdominal_MRI/processed/ 
### python /home/rawangq1/git_wq/SSL_veronika/utils/mp_removeaxis.py /mnt/qdata/share/rawangq1/ukbdata_70k/abdominal_MRI/sim
## srun exec/run_gpu3.sh 3
## sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,Elapsed,AveCPU,AllocCPUS,AllocGRES,ReqMem

echo "Finished"
