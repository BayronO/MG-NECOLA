#!/bin/bash
source ~/.bashrc

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export MASTER_ADDR="localhost"
export MASTER_PORT=12355
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

#models=("fR_p" "fR_pp" "fR_ppp" "fR_pppp")
#realization=99
output_dir="/data/mrhurtad/bayron-proj/tests_m2m/bpT/MG-SRNet"

echo "Master node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES before training: $CUDA_VISIBLE_DEVICES"

#for model in "${models[@]}"; do
for realization in $(seq 0 98); do
    echo "=== Running test for realization $realization ==="

    test_in="/data/jborjuel/NN/fRemu/fRemu_${realization}.npy"
    test_tgt="/data/jborjuel/NN/fRemu/fRemu_0.npy"

    # Run the test
    python m2m.py test \
        --test-in-patterns "$test_in" \
        --test-tgt-patterns "$test_tgt" \
        --in-norms cosmology.dis \
        --tgt-norms cosmology.dis \
        --model vnet.VNet --callback-at . \
        --batches 1 \
        --batch-size 2 \
        --crop 128 --crop-step 128 \
        --in-pad 20 --tgt-pad 0 \
        --load-state "checkpoint.pt" \
        #> "${output_dir}/test_${model}.log" 2>&1

    # Ubicación esperada del archivo generado por el test
    original_file="${output_dir}/fRemu_0_out_state92.npy"

    # Nombre final deseado
    new_file="${output_dir}/fRemu_${realization}.npy"

    if [ -f "$original_file" ]; then
        mv "$original_file" "$new_file"
        echo "✅ Archivo renombrado a: $new_file"
    else
        echo "⚠️ No se encontró el archivo esperado para realization $realization"
    fi

    echo "=== Test para realization $realization completado ==="
done

echo "🎯 Todos los tests completados!"
