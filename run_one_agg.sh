#!/bin/bash
# Run one aggregation experiment at a time.
# Usage:
#   bash run_one_agg.sh fedavg
#   bash run_one_agg.sh nn_weighted
#
# Recommended comparison list:

#choices=
#    "fedavg",
#    "loss_weighted",
#    "gamma_loss_weighted",
#    "ssim_weighted",
#    "nn_weighted",
#    "fedlaw",
#    "qagafed",
#    "fgl_ac",
#    "fedga",
#    "gpfl",
#    "hybrid_fedgraph",
#    "rl_bandit",
#    "fednova",
#    "fedadam",
#    "fedadam_ssim",


set -euo pipefail
ls -ld 
AGG=${1:-gpfl}

export TASK_NAME=${TASK_NAME:-T2_to_T1}
export AGG_MODE=${AGG}
export NUM_ROUNDS=${NUM_ROUNDS:-400}
export LOCAL_EPOCHS=${LOCAL_EPOCHS:-1}
export NUM_CLIENTS=${NUM_CLIENTS:-4}
export SERVER_PORT=${SERVER_PORT:-8080}

echo "Submitting server for AGG_MODE=${AGG_MODE}"
SERVER_JOB=$(sbatch --export=ALL job_fl_server.sh | awk '{print $4}')
echo "Server job id: ${SERVER_JOB}"

echo "Submitting clients after server starts writing host/port files."
sleep 20
CLIENT_JOB=$(sbatch --export=ALL --dependency=after:${SERVER_JOB} job_fl_client.sh | awk '{print $4}' || true)

echo "If dependency prevents clients from starting until server exits on your cluster, submit clients manually after 20-30 seconds:"
echo "  AGG_MODE=${AGG_MODE} TASK_NAME=${TASK_NAME} sbatch job_fl_client.sh"
