# CVMRI/FL federated pix2pix patch

Place these files in:

```bash
/scratch/shassanizorgabad22/CVMRI/FL/
```

Your directory should become:

```bash
CVMRI/
  models/
  preprocessing/
  trainer/
  utils/
  FL/
    fl_common.py
    client.py
    server.py
    job_fl_server.sh
    job_fl_client.sh
    run_one_agg.sh
```

## Run one experiment

```bash
cd /scratch/shassanizorgabad22/CVMRI/FL

AGG_MODE=fedavg TASK_NAME=T2_to_T1 sbatch job_fl_server.sh
# wait until logs/fl_server_host.txt and logs/fl_server_port.txt exist
AGG_MODE=fedavg TASK_NAME=T2_to_T1 sbatch job_fl_client.sh
```

## Aggregation modes

```text
fedavg
loss_weighted
gamma_loss_weighted
ssim_weighted
nn_weighted
fednova
fedadam
fedadam_ssim
```

Recommended first tests:

```bash
AGG_MODE=fedavg NUM_ROUNDS=5 sbatch job_fl_server.sh
AGG_MODE=fedavg NUM_ROUNDS=5 sbatch job_fl_client.sh

AGG_MODE=ssim_weighted NUM_ROUNDS=5 sbatch job_fl_server.sh
AGG_MODE=ssim_weighted NUM_ROUNDS=5 sbatch job_fl_client.sh

AGG_MODE=nn_weighted NUM_ROUNDS=5 sbatch job_fl_server.sh
AGG_MODE=nn_weighted NUM_ROUNDS=5 sbatch job_fl_client.sh
```

Then run full 400-round versions.

## Important note about NN weighting

`nn_weighted` is not a magic supervised SSIM optimizer at the beginning, because true global SSIM after aggregation is only known after aggregation. So the included NN learns a stable mapping from client statistics to SSIM-oriented target weights. It uses `val_ssim`, `val_psnr`, `val_loss`, sample share, round progress, and client id as features.

For your paper/experiments, compare:
1. FedAvg baseline
2. Loss-weighted
3. SSIM-weighted
4. NN-weighted
5. FedNova
6. FedAdam
7. FedAdam + SSIM weighting
