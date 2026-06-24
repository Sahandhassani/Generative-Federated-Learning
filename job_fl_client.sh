#!/bin/bash
#SBATCH --job-name=fl_client
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# This script should live in: CVMRI/FL/job_fl_client.sh
cd /scratch/shassanizorgabad22/CVMRI/FL

mkdir -p logs outputs/history outputs/checkpoints outputs/generated

module purge
module load cuda/12.8.0
module load anaconda3/2025.06

source activate cv_env
eval "$(conda shell.bash hook)"
conda activate cv_env

nvidia-smi

TASK_NAME=${TASK_NAME:-T1_to_FLAIR}
AGG_MODE=${AGG_MODE:-fedlaw}

export FL_OUTPUT_ROOT=${FL_OUTPUT_ROOT:-$(pwd)/outputs/${TASK_NAME}/${AGG_MODE}}
export FL_LOG_ROOT=${FL_LOG_ROOT:-$(pwd)/logs/${TASK_NAME}/${AGG_MODE}}
mkdir -p "${FL_OUTPUT_ROOT}" "${FL_LOG_ROOT}"

SERVER_HOST=$(cat logs/fl_server_host.txt)
SERVER_PORT=$(cat logs/fl_server_port.txt)
CLIENT_ID=${SLURM_ARRAY_TASK_ID}

SEED=${SEED:-42}
export FL_SEED=${SEED}
export STRICT_DETERMINISM=${STRICT_DETERMINISM:-1}
export FL_OUTPUT_ROOT=${FL_OUTPUT_ROOT:-$(pwd)/outputs/${TASK_NAME}/${AGG_MODE}/seed_${SEED}}
export FL_LOG_ROOT=${FL_LOG_ROOT:-$(pwd)/logs/${TASK_NAME}/${AGG_MODE}/seed_${SEED}}

echo "Client node: $(hostname)"
echo "Client ID: ${CLIENT_ID}"
echo "Connecting to ${SERVER_HOST}:${SERVER_PORT}"
echo "TASK_NAME=${TASK_NAME}"
echo "AGG_MODE=${AGG_MODE}"
echo "FL_OUTPUT_ROOT=${FL_OUTPUT_ROOT}"

srun python -u client.py \
    --server_address "${SERVER_HOST}:${SERVER_PORT}" \
    --client_id "${CLIENT_ID}" \
    --direction "${TASK_NAME}"