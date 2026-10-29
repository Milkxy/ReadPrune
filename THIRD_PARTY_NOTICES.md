# Third-party dependencies and provenance

This repository does not redistribute pretrained weights, evaluation datasets,
or upstream VLM source trees. Users obtain those resources separately under
their respective licenses and terms.

Runtime dependencies include NumPy, SciPy, PyTorch, Hugging Face Transformers,
Safetensors, and Pillow. Their licenses remain with their respective projects.

Related upstream implementations:

- CLIP: https://github.com/openai/CLIP
- Transformers: https://github.com/huggingface/transformers
- LLaVA: https://github.com/haotian-liu/LLaVA
- LLaVA-NeXT: https://github.com/LLaVA-VL/LLaVA-NeXT
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL

The first-party reference code was extracted from the frozen ReadPrune/FRA-Q2
implementation used in the image experiments. The published package removes
the unused exchange-repair and aggregation branches. Detailed extraction and
validation notes are in `docs/reproducibility.md`.
