export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3

cd src


# torchrun --nproc_per_node=4 main_crop.py

# torchrun --nproc_per_node=4 main_crop_scs.py

# torchrun --nproc_per_node=4 main_coords.py
