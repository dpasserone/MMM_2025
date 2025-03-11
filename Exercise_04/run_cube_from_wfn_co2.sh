#!/bin/bash -l

DIR="./"

mkdir cubes

/home/jovyan/soft/cp2k-spm-tools/cube_from_wfn.py \
  --cp2k_input_file $DIR/aiida.inp \
  --basis_set_file BASIS_MOLOPT \
  --xyz_file $DIR/aiida.coords.xyz \
  --wfn_file $DIR/aiida-RESTART.wfn \
  --output_dir ./cubes/ \
  --n_homo 7 \
  --n_lumo 5 \
  --dx 0.2 \
  --eval_cutoff 14.0 \

