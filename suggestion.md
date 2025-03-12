我们目标是：用户已经有了一个 LLM 的环境，可能是直接调 API，可能是本地 Llama inference，可能是 vLLM 等等。用户在这个环境下安装我们的库不会造成依赖冲突。用户直接 import 我们库，用最少行数的代码就能把他已有的 model，algorithm 和 dataset 扔到我们的 visualization 里来

✅ requirements 取最小集合。可以从一个空 conda/venv 环境开始装一遍，看需要手动输入安装的库有哪些。被动连带的库不用写进 requirements，因为有些库更新后这些连带的库会发生变化。版本用>=而不是=，不然容易和用户当前环境冲突。>=后面最好不要用最新版本的库，往前一年的版本会对用户更友好一些。我们的代码应该既不依赖 pytorch 也不依赖 gpu？这些都可以去掉。我们应该只需要 sklearn（tSNE）？

目前抽象在 model 端，其实 model 和 algorithms 只是 examples，教一下用户如何简单抽象出我们的接口样子。重点应该是 visualization 端的参数的直观性、兼容性和可用性。

import lot

dataset = lot.datasets.AQuA("data/aqua.jsonl")

# our examples

model = lot.models.Llama3(...)
model = lot.models.Mistral(...)

# prompts are part of the algorithm objects

algorithm = lot.algorithms.ToT(...)
algorithm = lot.algorithms.CoT(prompt_file="aqua_cot.txt")

# user's own model

# we put minimal requirements on how user must wrap their models

# as long as it supports **call**(prompt: str) and get_likelihood(prompt: str, continuation: str) functions

# I guess user only needs <=20 lines of code to wrap any existing model

model = CustomModel() # this is a class defined by the user
dataset = CustomDataset() # this is a class defined by the user

# Optional: we may also provide some helper class for users to wrap their dataset

# make sure this helper class covers enough number of common datasets

# otherwise we don't provide such a helper class

# e.g. here is a json dataset where users can specify the query and answer fields

dataset = lot.datasets.JsonDataset(query_field="query", answer_field="answer")

# num_sample refers to #path per test query

# return the feature of states

features, metrics = lot.sample(dataset, model, algorithm, num_sample=10)

# Do we want features to be a fixed 3D tensor or list of list of list?

# The reason is that different samples may result in different reasoning path length

# metrics look like this

# {"accuracy": list or ndarray, "consistency": list or ndarray, "uncertainty": list or ndarray, perplexity: list or ndarray}

# automatically append the features of choices to the features of states

# metrics=None is optional. If provided, visualize them on the side of the plot.

# save into a file sequence: visualization_0.png, ... visualization_999.png

lot.visualize(features, metrics=metrics, file*pattern="visualization*%d.png")

# lightweight verifier

targets = ...
verifier = sklearn.svm.SVC(...)
verifier.fit(features, targets)
