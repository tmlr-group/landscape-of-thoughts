export TOGETHERAI_API_KEY=XXX

python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method cot
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name commonsenseqa --dataset_path data/commonsenseqa.jsonl --method cot
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name mmlu --dataset_path data/mmlu_college_physics.json --method cot
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name strategyqa --dataset_path data/strategyqa.json --method cot

python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method l2m
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name commonsenseqa --dataset_path data/commonsenseqa.jsonl --method l2m
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name mmlu --dataset_path data/mmlu_college_physics.json --method l2m
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name strategyqa --dataset_path data/strategyqa.json --method l2m

python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method tot
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name commonsenseqa --dataset_path data/commonsenseqa.jsonl --method tot
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name mmlu --dataset_path data/mmlu_college_physics.json --method tot
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name strategyqa --dataset_path data/strategyqa.json --method tot

python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method mcts
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name commonsenseqa --dataset_path data/commonsenseqa.jsonl --method mcts
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name mmlu --dataset_path data/mmlu_college_physics.json --method mcts
python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name strategyqa --dataset_path data/strategyqa.json --method mcts

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DATASETS=("aqua" "mmlu" "strategyqa" "commonsenseqa")
METHODS=("cot" "l2m" "tot" "mcts")

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        for i in {0..49}; do
            python step2-compute-distance-matrix.py \
                --model_name "$MODEL" \
                --thoughts_file "exp-data-scale/$dataset/thoughts/$MODEL--$method--$dataset--$i.json"
        done
    done
done