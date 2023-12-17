NETWORK=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
SOLVER="heun"
SKIP_METHOD="gaussian"
SIGMA_MAX=80.0
DS_PARAMS_DIR="gaussian_params"

BATCH=512
SEEDS="0-9999"

CUDA_VISIBLE_DEVICES=0

BASE_NFE=35
COMMON_SAVE_DIR=images_with_skips

mkdir $COMMON_SAVE_DIR


N_SKIPS=(0 1 2 3 4 5 6 7 10 12)
SIGMA_SKIPS=(80.0 57.6 40.8 28.4 19.4 12.9 8.4 5.3 3.3 1.1 0.3)
for idx in ${!N_SKIPS[*]}
do
    N_SKIP=${N_SKIPS[$idx]}
    NFE=$((BASE_NFE - 2 * N_SKIP))
    SIGMA_SKIP=${SIGMA_SKIPS[$idx]}
    
    SAVE_DIR=${COMMON_SAVE_DIR}/${N_SKIP}
    mkdir SAVE_DIR

    python generate.py \
        --network=${NETWORK} --solver=${SOLVER} --skip_method=${SKIP_METHOD} \
        --sigma_max=${SIGMA_MAX} --sigma_skip=${SIGMA_SKIP} --ds_params_dir=${DS_PARAMS_DIR}\
        --batch=${BATCH} --seeds=${SEEDS} --outdir=${SAVE_DIR} 
done