<!-- <div align="center"><img src="imgs/banner.png" width="700"/></div> -->

<h1 align="center"> Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models </h1>

<!-- <p align="center">
    <a href="https://deepinception.github.io/"><img src="https://img.shields.io/badge/Project Website-deepinception" alt="Website"></a>
    <a href="https://arxiv.org/abs/2311.03191"><img src="https://img.shields.io/badge/cs.ML-arXiv%3A2311.03191-b31b1b" alt="Paper"></a>
    <img src="https://badges.toozhao.com/badges/01HEPEKFHAV8CP6JTE7JWYGVV3/blue.svg" alt="Count">
    <img src="https://img.shields.io/github/stars/tmlr-group/DeepInception?color=yellow&label=Star" alt="Stars" >
</p> -->

<!-- <div class="is-size-5 publication-authors" align="center">
            <span class="author-block">
              <a href="https://github.com/XuanLi728">Xuan Li</a><sup>1*</sup>,
            </span>
            <span class="author-block">
              <a href="https://github.com/AndrewZhou924">Zhanke Zhou</a><sup>1*</sup>,
            </span>
            <span class="author-block">
              <a href="https://zfancy.github.io/">Jianing Zhu</a><sup>1*</sup>,
            </span>
            <span class="author-block">
              <a href="https://bhanml.github.io/">Bo Han</a><sup>1</sup>,
            </span>
</div> -->

We present `landscape-of-thoughts`, a visualization tool that maps LLMs' reasoning paths into 2D space using t-SNE, enabling users to analyze and improve reasoning in multiple-choice tasks by representing reasoning states as feature vectors relative to possible answers.

![demo](imgs/demo.png)

## Setting up environment

We use `python==3.10` and `torch==2.5.1` with `CUDA==12.6`. To set up the environment, run:

```bash
conda create -n landscape python=3.10
pip3 install -r requirements.txt
```

## Preparing the opensource LLM

- **Using TogetherAI API**

  We use TogetherAI API to interact with the LLM. The documents are in [Doc](https://docs.together.ai/docs/introduction).

  Set your API key by running:

  ```bash
  export TOGETHERAI_API_KEY=YOUR API KEY
  ```

  Test your connection by running:

  ```python
    from together import Together

    client = Together()

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
    )
    print(response.choices[0].message.content)
  ```

- **Using self-hosted LLM**

  Take `Llama-3.2-1B-Instruct` as an example, saved at `ROOT_PATH/hf_models/Llama-3.2-1B-Instruct`, we start from downloading the model from Huggingface.

  ```bash
  huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ROOT_PATH/hf_models
  ```

  **NOTE: Remember to rename the model folder (remove "meta_llama\_") to avoid conflict with TogetherAI API.**

  Then, we use [fastchat](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md) or [vllm](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html) to host the model and OpenAI-Compatible RESTful server to interact with them.

  **For fastchat:**

  ```bash
  python3 -m fastchat.serve.controller
  python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 # the default API CALL URL

  # we can start two model in different GPU with different port;
  # the port is only for controller to find the model

  CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.model_worker \
  --model-path ROOT_PATH/hf_models/Llama-3.2-1B-Instruct \
  --port 32001 \
  --worker-address http://localhost:32001

  CUDA_VISIBLE_DEVICES=1 python -m fastchat.serve.model_worker \
  --model-path ROOT_PATH/hf_models/Llama-3.2-3B-Instruct \
  --port 32003 \
  --worker-address http://localhost:32003
  ```

  **For vllm:**

  ```bash
  vllm serve meta-llama/Llama-3.2-1B-Instruct --api-key token-abc123 --download_dir ROOT_PATH/hf_models
  ```

  Now we can call `Llama-3.2-1B-Instruct` via OpenAI SDK by running:

  ```python
  from openai import OpenAI
  client = OpenAI(
      base_url="http://localhost:8000/v1",
      api_key="token-abc123",
  )
  model = "meta-llama/Llama-3.2-1B-Instruct" # if you use vllm
  # model = "Llama-3.2-1B-Instruct" # if you use fastchat
  messages = [
      {"role": "system", "content": "I am a large language model."},
  ]
  question = "Hello!"
  messages.append({"role": "user", "content": question})
  response = client.chat.completions.create(model=model,messages=messages,)
  print(response.choices[0].message.content)
  ```

## Downloading reference responses and processed data

❗❗❗ NOTE: The following calculation requires **massive API queries**, we provide the results we calclulated for reference, see [GazeEzio/Landscape-of-Thought](https://huggingface.co/datasets/GazeEzio/Landscape-of-Thought) in huggingface, run:

```bash
sudo apt-get install git-lfs
git lfs clone git@hf.co:datasets/GazeEzio/Landscape-Data
```

We also provide the bash commands of the following steps in [here](scripts/landscape_llama3.1-8B-Instruct.sh).

The file tree of the provided data:

```
Landscape-Data/
├── aqua
│   ├── Tree
│   ├── distance_matrix
│   ├── inter_distance_matrix
│   └── thoughts
├── commonsenseqa
├── mmlu
└── strategyqa
```

## Obtaining responses from LLM

The employed dataset can be found in [data](./data). Taking `Llama-3.1-8B-Instruct` as an example, to obtain the ploting data in `AQuA` dataset with `CoT` using TogetherAI, run:

```bash
export TOGETHERAI_API_KEY=XXX

python step1-sample-reasoning-trace.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method cot
```

The results will be saved as `./exp-data-scale/aqua/thoughts/Meta-Llama-3.1-8B-Instruct-Turbo--cot--aqua--*.json`.

Expected file tree:

```
exp-data-scale
├── aqua <== dataset_name
│   ├── distance_matrix <== the data for ploting
│   └── thoughts <== the LLM raw responses
├── commonsenseqa
│   ├── distance_matrix
│   └── thoughts
├── mmlu
│   ├── distance_matrix
│   └── thoughts
└── strategyqa
    ├── distance_matrix
    └── thoughts
```

## Calculating the distance matrix

To plot the previous responses, we need to map the natural language to 2D landscape by calculating the respective distance matrix. To do this, run:

```bash
python step-2-compute-distance-matrix.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --dataset_path data/aqua.jsonl --method cot
```

The results will be saved as `./exp-data-scale/aqua/distance_matrix/Meta-Llama-3.1-8B-Instruct-Turbo--cot--aqua--*.pkl`.

## Ploting

To plot the landscape using distance matrix, run:

```bash
python PLOT-landscape.py --model_name Meta-Llama-3.1-8B-Instruct-Turbo --dataset_name aqua --method cot
```

You can find the landscape plot in `./figures/landscape_appendix/FIG1_{model_name}-{dataset_name}-{method}.png`.

## Citation

If you fine our paper or code are useful for your project, please consider citing our paper:

```
@inproceeding{XXX,
  title={Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models},
  author={Zhou, Zhanke XXXXXX},
  booktitle={arXiv},
  year={2025}
}
```
