
# llm-tutorial

This tutorial covers loading pre-trained Large Language Models (LLMs) on the HPC system. It also includes some notes from my experiences with LLMs.

## Setup

### Connect to a Slurm Runtime
```bash
srun -w cn-m-2 -p soundbendor -A soundbendor --gres=gpu:2 --pty bash
```

### Load CUDA 11.8
```bash
module load python/3.10 cuda/11.8
```

> [!NOTE] The `bitsandbytes` module needs this specific version of cuda. Check [here](https://pypi.org/project/bitsandbytes/) for updates (waiting for 12.1 support).

### Activate the Python Environment
<!-- Created with: module load python/3.10 cuda/11.8; python3 -m venv env -->
```bash
source env/bin/activate
```

<!-- 
Pakages installed with:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install huggingface transformers accelerate bitsandbytes

if [Errno 28] No space left on device:
pip cache purge
rm -rf ~/.cache/huggingface/*
 -->

### Run the Demo
```bash
python generate_dataset.py
```

This code loads `GPT-2` and generates a dataset of prompts and responses. 

## Notes

In [generate_dataset.py](generate_dataset.py), I use GPT-2 as an example model. You can switch to a different model by changing the `model_name` variable. For example:

```python
model_name = 'meta-llama/Llama-2-7b-chat-hf'
```

> **Note:** To use Llama-2, you need an access token from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

Once you've recieved a token, to access it go to your hugging face account then: `Settings` > `Access Tokens`.

To streamline the process, set up the Git credential helper:

```bash
git config --global credential.helper store
```

Then, log in to Hugging Face:

```bash
huggingface-cli login
```

Paste your access token and confirm to add it to your git credentials.

With this setup, you're ready to run `Llama-2`.

Running `python generate_dataset.py` might take a while, especially for large models like `Llama-2-7b`.

### GPU Memory Tips
You might get an error that looks like this:

```No space left on device: '/nfs/stak/...'```

This is the part where we need to start talking about GPU memory. It can be a pain but here are some tips to help.

Determining the exact GPU memory requirement for a model from Hugging Face can be challenging. The memory usage depends on the model's size, batch size, and system overhead. For flagship models, a quick search like `'Model X' gpu memory usage` can provide a good estimate. For instance, `Llama-2-7b` roughly requires about **56 GB** of GPU memory. You can also aproximate the memory usage in this form:

```
# The default dtype of parameters in pytorch is float32
# 4 bytes per parameter for float32
total_size_bytes = total_parameters * bytes_per_parameter
```

> **Note:** We can change the datatype of the parameters by using [bitsandbytes](https://pypi.org/project/bitsandbytes/).

To check the available GPU memory in your Slurm runtime, run `nvidia-smi` and refer to the Memory Usage column.

> **Note:** Total memory is displayed in MiB. To approximate GB, you can generally disregard the last three digits.

#### GPU Resources on `soundbendor` Nodes

| Node   | GPUs               | Memory per GPU    | Total GPU Memory   |
|--------|--------------------|-------------------|--------------------|
| cn-m-1 | 6x Tesla T4        | 15360 MiB (~15 GB)| 92160 MiB (~92 GB) |
| cn-m-2 | 2x Quadro RTX 6000 | 24576 MiB (~24 GB)| 49152 MiB (~49 GB) |

#### Running Large Models on Multiple GPUs

To run large models like `Llama-2` we will have to do a little more work to fit our model on the server. To do this, we will utilize two libraries [accelerate](https://pypi.org/project/bitsandbyte) and [bitsandbytes](https://pypi.org/project/bitsandbytes/). See following code from [chat.py](chat.py):

...

`Accelerate`: Automatically places your model on the available GPU and enables mixed-precision training and inference, which can reduce memory usage and increase performance.

`bitsandbytes`: Let's us load the models parameters in smaller datatypes by quantizing the parameters.

### In Summary

We can employ several strategies to run large models with limited resources:
- Using smaller version of the model.
- Quatization with `bitsandbytes`
- Leverage vector-processing with `accelerate` see [this](https://youtu.be/MWCSGj9jEAo) for an explanation of how it works.
- It's also worth looking into `petals` which enables collaborative inference over a network. (it's pretty slow but a good last resort)