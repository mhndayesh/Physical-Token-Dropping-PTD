import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2DecoderLayer, Qwen2ForCausalLM
import inspect

def main():
    print(inspect.signature(Qwen2Model.forward))
    print(inspect.signature(Qwen2DecoderLayer.forward))

if __name__ == "__main__":
    main()
