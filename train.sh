export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3

cd src

# torchrun --nproc_per_node=4 main_scs.py
# torchrun --nproc_per_node=4 main_nfn.py
# torchrun --nproc_per_node=4 main_ss.py

# torchrun --nproc_per_node=4 main_crop.py --lr 2e-3
# torchrun --nproc_per_node=4 main_crop.py --lr 3e-3
# torchrun --nproc_per_node=4 main_crop.py --model coatnet_2_rw_224

torchrun --nproc_per_node=4 main_crop_scs.py

# torchrun --nproc_per_node=4 main_crop.py
# torchrun --nproc_per_node=4 main_spinenet.py

# torchrun --nproc_per_node=4 main_crop_ax.py

# torchrun --nproc_per_node=4 main_crop_scs.py
# echo
# torchrun --nproc_per_node=4 main_crop_nfn.py
# torchrun --nproc_per_node=4 main_crop_nfn.py --model coatnet_2_rw_224
# torchrun --nproc_per_node=4 main_crop_nfn.py --model coatnet_0_rw_224
# torchrun --nproc_per_node=4 main_crop_ss.py

# echo

# torchrun --nproc_per_node=4 main_coords.py
# torchrun --nproc_per_node=4 main_coords_dec.py
# torchrun --nproc_per_node=4 main_coords_ax.py


# torchrun --nproc_per_node=4 main_crop_bi.py
# torchrun --nproc_per_node=4 main_ss_dec.py
# torchrun --nproc_per_node=4 main_ss.py
