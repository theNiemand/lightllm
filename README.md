<div align="center">
  <picture>
    <img alt="LightLLM" src="assets/lightllm.drawio.png" width=90%>
  </picture>
</div>

---
<div align="center">

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://github.com/ModelTC/lightllm/blob/main/docs/TokenAttention.md)
[![Docker](https://github.com/ModelTC/lightllm/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/ModelTC/lightllm/actions/workflows/docker-publish.yml)
[![stars](https://img.shields.io/github/stars/ModelTC/lightllm?style=social)](https://github.com/ModelTC/lightllm)
![visitors](https://komarev.com/ghpvc/?username=lightllm&label=visitors)
[![Discord Banner](https://img.shields.io/discord/1139835312592392214?logo=discord&logoColor=white)](https://discord.gg/WzzfwVSguU)
[![license](https://img.shields.io/github/license/ModelTC/lightllm)](https://github.com/ModelTC/lightllm/blob/main/LICENSE)
</div>

LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance. LightLLM harnesses the strengths of numerous well-regarded open-source implementations, including but not limited to FasterTransformer, TGI, vLLM, and FlashAttention.

**English doc** is [here](https://lightllm-en.readthedocs.io/en/latest/).

**Chinese doc** is [here](https://lightllm-cn.readthedocs.io/en/latest/).

## HightLight Features

- Tri-process asynchronous collaboration: tokenization, model inference, and detokenization are performed asynchronously, leading to a considerable improvement in GPU utilization.
- [Token Attention](./docs/TokenAttention.md): implements token-wise's KV cache memory management mechanism, allowing for zero memory waste during inference.
- High-performance Router: collaborates with Token Attention to meticulously manage the GPU memory of each token, thereby optimizing system throughput.
- Int8KV Cache: This feature will increase the capacity of tokens to almost twice as much. only llama support.
- Multiple model support: Regular LLMs, Mixture-of-Expert LLMs, Multimodal LLMs and Reward LLMs. Full list can be found [here](https://lightllm-en.readthedocs.io/en/latest/models/supported_models.html).


## Get started

- [Installation](https://lightllm-en.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://lightllm-en.readthedocs.io/en/latest/getting_started/quickstart.html)

### Examples

- RUN [LLMs](https://lightllm-en.readthedocs.io/en/latest/models/test.html#llama2-70b-chat)
- RUN [VLMs](https://lightllm-en.readthedocs.io/en/latest/models/test.html#qwen-vl-chat)
- RUN [Reward LLMs](https://lightllm-en.readthedocs.io/en/latest/models/test.html#internlm2-1-8b-reward)

## [Benchmark](https://lightllm-en.readthedocs.io/en/latest/server/benchmark.html)

### FAQ

- The LLaMA tokenizer fails to load.
    - consider resolving this by running the command `pip install protobuf==3.20.0`.
- `error   : PTX .version 7.4 does not support .target sm_89`
    - launch with `bash tools/resolve_ptx_version python -m lightllm.server.api_server ... `

## Projects using lightllm

If you have a project that should be incorporated, please contact via email or create a pull request.

1. <details><summary> <b><a href=https://github.com/LazyAGI/LazyLLM>LazyLLM</a></b>: Easyest and lazyest way for building multi-agent LLMs applications.</summary>

    Once you have installed `lightllm` and `lazyllm`, and then you can use the following code to build your own chatbot:

    ~~~python
    from lazyllm import TrainableModule, deploy, WebModule
    # Model will be download automatically if you have an internet connection
    m = TrainableModule('internlm2-chat-7b').deploy_method(deploy.lightllm)
    WebModule(m).start().wait()
    ~~~

    Documents: https://lazyllm.readthedocs.io/

    </details>

## Community

For further information and discussion, [join our discord server](https://discord.gg/WzzfwVSguU).

## License

This repository is released under the [Apache-2.0](LICENSE) license.

## Acknowledgement

We learned a lot from the following projects when developing LightLLM.
- [Faster Transformer](https://github.com/NVIDIA/FasterTransformer)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [vLLM](https://github.com/vllm-project/vllm)
- [Flash Attention 1&2](https://github.com/Dao-AILab/flash-attention)
- [OpenAI Triton](https://github.com/openai/triton)
