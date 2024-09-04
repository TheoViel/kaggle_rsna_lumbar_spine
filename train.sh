export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3

cd src

# torchrun --nproc_per_node=4 main_scs.py
# torchrun --nproc_per_node=4 main_nfn.py
# torchrun --nproc_per_node=4 main_ss.py

# torchrun --nproc_per_node=4 main_crop.py
# torchrun --nproc_per_node=4 main_crop_ax.py

# torchrun --nproc_per_node=4 main_crop_scs.py
# echo
torchrun --nproc_per_node=4 main_crop_nfn.py
# torchrun --nproc_per_node=4 main_crop_ss.py

# echo

# torchrun --nproc_per_node=4 main_coords.py
# torchrun --nproc_per_node=4 main_coords_ax.py


# torchrun --nproc_per_node=4 main_crop_nfn_bi.py
