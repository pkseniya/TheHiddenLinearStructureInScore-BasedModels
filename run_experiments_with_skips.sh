NETWORK=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
SOLVER="heun"
SKIP_METHOD="gaussian"
SIGMA_MAX=80.0
DS_PARAMS_DIR="gaussian_params"

BATCH=1024

SEEDS="0-49999"

BASE_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
N_PROC=$((1 + ${#BASE_CUDA_VISIBLE_DEVICES} / 2))

BASE_N_STEPS=18
COMMON_SAVE_DIR=images_with_skips

mkdir $COMMON_SAVE_DIR


N_SKIPS=(0 1 2 3 4 5 6 7 8 10 12)
SIGMA_SKIPS=(80.0 57.6 40.8 28.4 19.4 12.9 8.4 5.3 3.3 1.1 0.3)
for idx in ${!N_SKIPS[*]}
do
    N_SKIP=${N_SKIPS[$idx]}
    N_STEPS=$((BASE_N_STEPS - N_SKIP))
    SIGMA_SKIP=${SIGMA_SKIPS[$idx]}
    
    SAVE_DIR=${COMMON_SAVE_DIR}/${N_SKIP}

    CUDA_VISIBLE_DEVICES=${BASE_CUDA_VISIBLE_DEVICES}
    torchrun --standalone --nproc_per_node=${N_PROC} generate.py \
        --network=${NETWORK} --solver=${SOLVER} --steps=${N_STEPS} --skip_method=${SKIP_METHOD} \
        --sigma_max=${SIGMA_MAX} --sigma_skip=${SIGMA_SKIP} --ds_params_dir=${DS_PARAMS_DIR}\
        --batch=${BATCH} --seeds=${SEEDS} --outdir=${SAVE_DIR}

    CUDA_VISIBLE_DEVICES=${BASE_CUDA_VISIBLE_DEVICES:0:1}
    NUM_IMAGES=$(ls -l ${SAVE_DIR}/*.png | wc -l)
    python -u fid.py calc --images=${SAVE_DIR} --num=${NUM_IMAGES} \
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz > ${COMMON_SAVE_DIR}/${N_SKIP}.txt
done