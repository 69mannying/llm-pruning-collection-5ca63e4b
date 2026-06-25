for slurm_script in $(ls scripts/prune_llama3_d*.sh); do
    echo "sbatch $slurm_script"
    sbatch $slurm_script
done