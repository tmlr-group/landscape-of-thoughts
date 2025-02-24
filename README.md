# Trace of thoughts

## Using TogetherAI API

```
# FX's API-KEY
export TOGETHERAI_API_KEY=dc7b6e35a7a0e0a582905d0c909ed0fb945208a40e25ca8cfee12a1855637b9c

# ZK's API-KEY
export TOGETHERAI_API_KEY=611c2c35f003a2ce973dcadac000bc2e1bd69ed50784f83f47e67e54ae83b641

# Xuan's API-KEY
export TOGETHERAI_API_KEY=651471eb039c544aeb2c1886dc0209d198fab2d8d5c7876d127a73da2faaa3239

python main.py --model_name=together_ai/meta-llama/Llama-3-8b-chat-hf --algo_name=cot --shots=3 --dataset=gsm8k
```

## Start fast chat server

Task Llama-3.2 1B as example, saved at `ROOT_PATH/hf_models/Llama-3.2-1B-Instruct`, we start from downloading the model from Huggingface.

[tool](https://github.com/bodaay/HuggingFaceModelDownloader)

```bash
bash <(curl -sSL https://g.bodaay.io/hfd) -m meta-llama/Llama-3.2-1B-Instruct -c 8 -s ./hf_models
```

**NOTE: Remember to rename the model folder (remove "meta_llama\_") to avoid conflict with TogetherAI API.**

Then, we use fastchat to host the model and OpenAI-Compatible RESTful server to interact with them.

[fastchat doc](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)

[how to setup multiple model](https://wangjunjian.com/fastchat/vllm/2024/01/16/Using-FastChat-to-Deploy-LLM-on-CUDA.html)

```bash
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 # the default API CALL URL

# we can start two model in different GPU with different port;
# the port is only for controller to find the model

CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.model_worker \
 --model-path /home/csxuanli/code/trace-of-thoughts/hf_models/Llama-3.2-1B-Instruct \
 --port 32001 \
 --worker-address http://localhost:32001

CUDA_VISIBLE_DEVICES=1 python -m fastchat.serve.model_worker \
 --model-path /home/csxuanli/code/trace-of-thoughts/hf_models/Llama-3.2-3B-Instruct \
 --port 32003 \
 --worker-address http://localhost:32003
```

Now we can call `Llama-3.2-1B-Instruct` via OpenAI SDK.

```python
import openai

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "Llama-3.2-1B-Instruct"
prompt = "Once upon a time"

# create a completion
completion = openai.completions.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
```
