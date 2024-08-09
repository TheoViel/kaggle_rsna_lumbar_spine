export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3

cd src

# torchrun --nproc_per_node=4 main_scs.py
# torchrun --nproc_per_node=4 main_crop_scs.py

torchrun --nproc_per_node=4 main_crop_nfn.py
# torchrun --nproc_per_node=4 main_nfn.py

# torchrun --nproc_per_node=4 main_ss.py

echo

torchrun --nproc_per_node=4 main_crop_scs.py