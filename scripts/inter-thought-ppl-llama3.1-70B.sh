export TOGETHERAI_API_KEY=XXX

MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
DATASETS=("aqua" "mmlu" "commonsenseqa" "strategyqa")
METHODS=("cot" "l2m")

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for i in {0..49}; do
            python step-2-interthoughts-distance.py \
                --model_name "$MODEL" \
                --thoughts_file "exp-data-scale/$dataset/thoughts/$MODEL--$method--$dataset--$i.json"
        done
    done
done