# SimpleAI : examples

Here you can several examples of how to use [SimpleAI](https://github.com/lhenault/simpleAI) to expose models and integrate them. There is also a [list of external projects using *SimpleAI*](#external-repositories) below.

## Overview

| Model | Description | Number of parameters | Chat | Chat (streaming) | Instruct | Completion | Completion (streaming) | Embedding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [MPT-7B-Chat](/examples/MPT-7B-Chat) | MosaicML's MPT model fine-tuned for chat | 7B | ✔️ | ✔️ | ❌ | ❌ | ❌ | ❌ |
| [MPT-7B-Storywriter-65k+](/examples/MPT-7B-Storywriter-65kplus) | MosaicML's MPT model fine-tuned with a context length of 65k tokens on a filtered fiction subset of the books3 dataset | 7B | ❌ | ❌ | ❌ | ✔️ | ✔️ | ❌ |
| [StableLM - Open-Assistant](/examples/stablelm-open-assistant) | StabilityAI's StableLM model fine-tuned for chat by Open-Assistant | 7B | ✔️ | ✔️ | ❌ | ❌ | ❌ | ❌ |
| [Alpaca](/examples/alpaca-lora-7B) | Instruct model using LoRA to reproduce the Stanford Alpaca model | 7B | ❌ | ❌ | ✔️ | ❌ | ❌ | ❌ |
| [Sentence-Transformers](/examples/sentence-transformers) | A basic example on how to use `sentence-transformers` to create embeddings from text | N/A | ❌ | ❌ | ❌ | ❌ | ❌ | ✔️ |

## External repositories

### **simple-rwkv**

Find it [here](https://github.com/Nintorac/simple_rwkv).

> This project uses the RWKV-LM model and turns it into an gRPC service that can be used through SimpleAI.
>  
> RWKV is an RNN with Transformer-level language model performance that can be trained like a GPT transformer and is 100% attention-free. It combines the best of RNN and transformer, providing great performance, fast inference, saves VRAM, fast training, "infinite" ctx_len, and free sentence embedding.

* Author : [@Nintorac](https://github.com/Nintorac/)
* Implements for the same model and in a single gRPC server:
  * `chat` (including streaming),
  * `completions`,
  * and `embeddings`.

## Contribute

Feel free to contribute with your own examples, either by submitting a PR or getting in touch to get your project referenced here.
