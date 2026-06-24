#!/bin/bash
#SBATCH --job-name=fl_server
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# This script should live in: CVMRI/FL/job_fl_server.sh
cd /scratch/shassanizorgabad22/CVMRI/FL

mkdir -p logs outputs/history outputs/checkpoints outputs/generated outputs/server_states

module purge
module load cuda/12.8.0
module load anaconda3/2025.06

source activate cv_env
eval "$(conda shell.bash hook)"
conda activate cv_env

nvidia-smi

SERVER_HOST=$(hostname)
SERVER_PORT=${SERVER_PORT:-8080}

# Recommended comparison list:
#   fedavg
#   loss_weighted
#   gamma_loss_weighted
#   ssim_weighted
#   nn_weighted
#   fedadam
#   fedadam_ssim

#   fedlaw
#   qagafed
#   hybrid_fedgraph
#   fedga
#   fgl_ac
#   gpfl

# Experiment controls
RUN_MODE=${RUN_MODE:-fl} #centralized or fl
TASK_NAME=${TASK_NAME:-T1_to_FLAIR}
AGG_MODE=${AGG_MODE:-fedlaw}  #fgl_ac
NUM_ROUNDS=${NUM_ROUNDS:-400}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
NUM_CLIENTS=${NUM_CLIENTS:-4}
GAMMA=${GAMMA:-1.0}
SEED=${SEED:-42}
export FL_SEED=${SEED}
export STRICT_DETERMINISM=${STRICT_DETERMINISM:-1}

# Optional path overrides
export FL_OUTPUT_ROOT=${FL_OUTPUT_ROOT:-$(pwd)/outputs/${TASK_NAME}/${RUN_MODE}/${AGG_MODE}/seed_${SEED}}
export FL_LOG_ROOT=${FL_LOG_ROOT:-$(pwd)/logs/${TASK_NAME}/${RUN_MODE}/${AGG_MODE}/seed_${SEED}}
mkdir -p "${FL_OUTPUT_ROOT}" "${FL_LOG_ROOT}" "${FL_OUTPUT_ROOT}/history"

echo "${SERVER_HOST}" > logs/fl_server_host.txt
echo "${SERVER_PORT}" > logs/fl_server_port.txt

echo "Flower server running on ${SERVER_HOST}:${SERVER_PORT}"
echo "SERVER NODE: $(hostname)"
echo "SERVER IP: $(hostname -I)"
echo "TASK_NAME=${TASK_NAME}"
echo "AGG_MODE=${AGG_MODE}"
echo "FL_OUTPUT_ROOT=${FL_OUTPUT_ROOT}"

srun python -u server.py \
    --run_mode "${RUN_MODE}" \
    --server_address 0.0.0.0:${SERVER_PORT} \
    --seed "${SEED}" \
    --num_rounds "${NUM_ROUNDS}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --num_clients "${NUM_CLIENTS}" \
    --task_name "${TASK_NAME}" \
    --aggregation_mode "${AGG_MODE}" \
    --gamma "${GAMMA}" \
    --loss_metric_key val_loss \
    --score_metric_key val_ssim \
    --server_lr "${SERVER_LR:-0.1}" \
    --nn_hidden_dim "${NN_HIDDEN_DIM:-16}" \
    --nn_lr "${NN_LR:-0.001}" \
    --nn_warmup_rounds "${NN_WARMUP_ROUNDS:-5}" \
    --nn_temperature "${NN_TEMPERATURE:-0.10}" \
    --history_path "${FL_OUTPUT_ROOT}/history/${AGG_MODE}_history.jsonl" \
    #--rl_lr "${RL_LR:-0.10}" \
    #--rl_entropy "${RL_ENTROPY:-0.02}" \
    #--rl_exploration "${RL_EXPLORATION:-0.05}" \
    #--rl_warmup_rounds "${RL_WARMUP_ROUNDS:-5}" \