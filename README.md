# ConDeltas (Conditioning Deltas) for ComfyUI

**See bottom section for ELI5.**

Conditioning deltas are conditioning vectors that are obtained by subtracting one prompt conditioning from another. The result of this is a latent vector between the two prompts that can be added to another prompt at an arbitrary strength, with an end result that's similar to that of a LoRA or an embedding.

In layman's terms, a ConDelta id made by subtracting one prompt vector from another after they've been processed by CLIP (or T5, or whatever).  For instance, if we subtract these prompts:

"An art deco city" - "A city"

...we end up with a condelta that encodes the concept of an art deco style. 

That delta can then be saved and applied to other prompts at generation time. Since it's a vector, it can be scaled positively or negatively, so it can be used as an alternative to a negative prompt. It's also quite small (4mb for Flux and ~0.6mb for SDXL, probably even smaller for SD15), so the memory footprint is negligible, and since it's just modifying an existing vector, it has zero impact on speed. It's likely to work on most other models that are implemented in comfy's architecture as well.

Here's an example image showing this working in Flux, with a jungle ConDelta from right to left and a moon base ConDelta from top to bottom (note that you can theoretically add as many of these as you want).
![ComfyUI_08775_](https://github.com/user-attachments/assets/d54eda8c-a0d6-4c30-aae1-4608ac159e1c)

See the sample_workflows subdirectory for usage examples.

## ELI5

1. Create a folder called ConDelta (case sensitive) inside ComfyUI's models folder.
2. Copy the files from custom_nodes/ComfyUI-ConDelta/sample_models to the folder you just created.
3. Insert the Apply ConDelta node into your Comfy workflow between Clip Text Encode and your KSampler like this: ![Screenshot 2024-12-06 at 18-02-26 Unsaved Workflow (4) - ComfyUI](https://github.com/user-attachments/assets/1bb61c02-1480-4b41-a0c6-5e7fefcd6d87)
4. Make sure to select a ConDelta that's compatible with the model you're using.
