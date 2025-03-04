# 🤖 Model Interaction API Guide

<div align="center">
  <img src="https://img.shields.io/badge/API_Format-OpenAI_Compatible-blue?logo=openai">
  <img src="https://img.shields.io/badge/Deploy_Options-Cloud%20%7C%20Local-green">
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python">
</div>

## 🔑 API Access Setup

### Cloud API Configuration

```bash
# Set TogetherAI credentials
export TOGETHERAI_API_KEY="your_api_key_here"
```

### Local Model Deployment

|                 | Document                                                             |
| --------------- | -------------------------------------------------------------------- |
| huggingface-cli | https://huggingface.co/docs/huggingface_hub/en/guides/cli            |
| vllm            | https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html |
| FastChat        | https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md      |

```bash
# 1. Download base model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir YOUR_MODEL_PATH

# 2. Choose deployment method
# Option A (recommend): vLLM
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --api-key "token-api-123" \
  --download_dir YOUR_MODEL_PATH \
  --port 8000

# Option B: FastChat
python -m fastchat.serve.controller # <== run in ssh session 1
python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000 # <== run in ssh session 2
CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.model_worker \
  --model-path YOUR_MODEL_PATH # <== run in ssh session 3
```

## 🚀 API Usage Examples

### Cloud API Client

```python
from together import Together

client = Together(api_key=os.environ["TOGETHERAI_API_KEY"])

response = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages=[{"role": "user", "content": "Explain quantum computing"}],
  temperature=0.7,
  max_tokens=500
)

print(f"Response: {response.choices[0].message.content}")
```

### Local API Client

```python
from openai import OpenAI

# Connect to local server
client = OpenAI(
  base_url="http://localhost:8000/v1",
  api_key="token-api-123"  # Match vLLM --api-key
)

# Unified API call
def query_model(prompt, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
  return client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5
  )
```

## 📊 API Endpoint Matrix

| Provider   | Base URL                    | Authentication | Supported Models                           |
| ---------- | --------------------------- | -------------- | ------------------------------------------ |
| TogetherAI | https://api.together.xyz/v1 | API Key        | [document](https://www.together.ai/models) |
| vLLM       | http://localhost:8000/v1    | Self-host      | Any HF-compatible model                    |
| FastChat   | http://localhost:8000/v1    | Self-host      | Any HF-compatible model                    |

## 🛠️ API Troubleshooting

### Request Body

```json
{
  "model": "llama3-1b-instruct",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant" },
    { "role": "user", "content": "What's the capital of France?" }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### Response Structure

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 9,
    "total_tokens": 23
  }
}
```

### FAQs

1. **Connection Refused**  
   → Verify model worker is running: `netstat -tuln | grep 8000`

2. **CUDA Out of Memory**  
   → Reduce batch size: `--max-model-len 512` in vLLM

3. **Model Not Found**  
   → Check model path casing: `Llama-3.2-1B-Instruct` vs `llama-3-2b`

4. **Slow Responses**  
   → Enable continuous batching: `--enforce-eager` in vLLM
