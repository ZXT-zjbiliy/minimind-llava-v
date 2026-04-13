# LLaVA Reference Files

These files are copied from the LLaVA-OneVision-1.5 repo as local reference.

They are useful for:

- understanding the original vision encoder path
- checking projector design
- checking tensor shapes
- checking how image embeddings are inserted into the language stream

Important:

- these files still contain original Megatron-style imports
- they are not yet drop-in runnable inside this new scaffold
- use them as reference when porting the vision tower into your MiniMind-side code

Most important reference files:

- `rice_vision_model.py`
- `adapter.py`
- `llavaov_1_5_model.py`
- `config/vision-model.json`
- `config/qwen3.json`
