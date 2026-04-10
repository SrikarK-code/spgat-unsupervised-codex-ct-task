#!/bin/bash

# Define the 5 ablation configurations: "GraphType ModelType"
declare -a ABLATIONS=(
    "knn gcn"
    "mknn gcn"
    "snn gcn"
    "knn gat"
    "mknn gat"
)

# Use your preferred seed
SEED=43

echo "Submitting Mu Dictionary Ablations to background..."

for idx in "${!ABLATIONS[@]}"; do
    # Split the string into variables
    read GRAPH MODEL <<< "${ABLATIONS[$idx]}"
    
    OUTDIR="ablation_${GRAPH}_${MODEL}"
    echo "Starting Ablation: Graph=${GRAPH}, Model=${MODEL} -> Output: ${OUTDIR}"
    
    # Run in background with nohup
    nohup python -u run_mu_ablations.py \
        --seed $SEED \
        --graph $GRAPH \
        --model $MODEL \
        --outdir $OUTDIR > "${OUTDIR}_log.txt" 2>&1 &
done

echo "All 5 jobs submitted. Run 'top' or 'htop' to monitor."




# #!/bin/bash

# # Array of seeds (42, 43, 44, and two randoms represented by -1)
# SEEDS=(42 43 44 -1 -1)

# # Run 5 Leiden Jobs
# for i in "${!SEEDS[@]}"; do
#     SEED=${SEEDS[$i]}
#     OUTDIR="exp_leiden_run_$i"
#     echo "Starting Leiden Run $i (Seed: $SEED)..."
#     nohup python -u run_exp.py --seed $SEED --cluster leiden --outdir $OUTDIR > "${OUTDIR}_log.txt" 2>&1 &
# done

# # Run 5 Argmax Jobs
# for i in "${!SEEDS[@]}"; do
#     SEED=${SEEDS[$i]}
#     OUTDIR="exp_argmax_run_$i"
#     echo "Starting Argmax Run $i (Seed: $SEED)..."
#     nohup python -u run_exp.py --seed $SEED --cluster argmax --outdir $OUTDIR > "${OUTDIR}_log.txt" 2>&1 &
# done

# echo "All 10 jobs submitted to background. Run 'top' or 'htop' to monitor."
