def create_slurm_files(T_value, folder_value, num_files):
    import os

    # Directory where the job files will be saved
    directory = "slurm_jobs"
    os.makedirs(directory, exist_ok=True)

    # Template for the SLURM job script
    job_template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=0:45:00
#SBATCH --mem=48G
#SBATCH --output=slurm/ponita_isr_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

# Activate your environment
source activate ponita
# Check whether the GPU is available
srun python $HOME/PONITA_SLR/main_isr_2002.py --model_name Ponita_{T}_f{i}_v8 --root_metadata NGT/kfold/{folder}/{T}/metadata_fold_{i}.json --save_folder logs/v8/{T} --wandb_log_folder NGT200Main_{folder}_ablations
"""

    # Create each job file
    for i in range(1, num_files + 1):
        filename = f"{T_value}_isr_200_f{i}.job"
        with open(filename, 'w') as f:
            # Write the job script to the file
            f.write(job_template.format(T=T_value, folder=folder_value, i=i))
        print(f"Created: {filename}")

# Customize these variables as needed
T_values = ["T1", "T2", "T3"]
folder_value = "1_2_3"
num_files = 6  # Creates metadata_1.job to metadata_6.job

# Call the function to create the files
for T_value in T_values:
    create_slurm_files(T_value, folder_value, num_files)
