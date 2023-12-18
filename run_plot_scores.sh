NETWORK=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
SOLVER="heun"
SKIP_METHOD="gaussian"
SIGMA_MAX=80.0
DS_PARAMS_DIR="gaussian_params"

SEEDS="0-99"

BASE_N_STEPS=18
COMMON_SAVE_DIR=scores
OUT_DIR=tmp

mkdir $COMMON_SAVE_DIR
mkdir $COMMON_SAVE_DIR/gaussian $COMMON_SAVE_DIR/isotropic $COMMON_SAVE_DIR/neural 

torchrun --standalone --nproc_per_node=1 generate.py --network=${NETWORK} --solver=${SOLVER} \
        --sigma_max=${SIGMA_MAX} --sigma_skip=${SIGMA_MAX} --skip_method=${SKIP_METHOD} \
       	--ds_params_dir=${DS_PARAMS_DIR} --seeds=${SEEDS} --scores_dir=${COMMON_SAVE_DIR} --outdir=${OUT_DIR}

python plot_scores.py ${COMMON_SAVE_DIR}

