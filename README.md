# ComfyUI-Jax-Nodes

Custom ComfyUI nodes for some niche Krita-focused workflows and easier prompt / pipeline handling.

## Nodes Overview

### Utility / Workflow

- `Conditional Select` (`JAX_ConditionalSelect`)
  - Simple if/else node: chooses between `true_value` and `false_value` based on a boolean `condition` and outputs the selected value.

- `Sampler Pipe In` (`JAX_KritaPipeIn`)
  - Packs a set of sampler-related inputs (`model`, `positive`, `negative`, `vae`, `image`, `clip`, `latent`) into a single `pipe` dictionary that can be passed around a workflow.

- `Sampler Pipe Out` (`JAX_SamplerPipeOut`)
  - Unpacks the `pipe` dictionary back into individual outputs (`model`, `positive`, `negative`, `vae`, `image`, `clip`, `latent`), useful for branching or recombining pipelines.

- `Image Size Multiplier` (`JAX_ImageSizeMultiplier`)
  - Multiplies an input `width` and `height` by a `multiplier` and outputs the new dimensions plus a small Markdown string summarizing the final size.

### Prompt / Conditioning

- `Easy Prompt (W/ Append)` (`JAX_EasyPrompt`)
  - CLIP text encoder that lets you write prompts with inline LoRA tags like `<lora:name:weight>` and append extra positive/negative text. It loads and applies LoRAs to the `model`/`clip`, then returns the updated model and positive/negative conditioning.

- `Easy Prompt` (`JAX_EasyPromptSimple`)
  - Simpler version of the above: supports `<lora:...>` tags but without extra append fields, just `positive` and `negative` inputs.

### Krita Integration

Most of the Krita nodes below are adapted from Acly’s excellent `comfyui-tooling-nodes` project:
https://github.com/Acly/comfyui-tooling-nodes  
If you want the latest upstream behavior or additional tooling features, check that repo for updates and changes.

- `Krita Output` (`JAX_KritaOutput`)
  - Sends images back to Krita via WebSocket; optional canvas resize flag for the Krita plugin UI.

- `Send Text` (`JAX_KritaSendText`)
  - Sends text/markdown/html back to Krita as a UI payload.

- `Krita Canvas` (`JAX_KritaCanvas`)
  - Placeholder canvas image plus width/height/seed outputs for Krita-started workflows.

- `Krita Selection` (`JAX_KritaSelection`)
  - Placeholder selection mask and active flag.

- `Krita Image Layer` (`JAX_KritaImageLayer`)
  - Placeholder layer image and mask for Krita integration.

- `Krita Mask Layer` (`JAX_KritaMaskLayer`)
  - Placeholder mask layer output.

- `Parameter` (`JAX_Parameter`)
  - Workflow parameter node with typed defaults/min/max, matching Krita parameter UI.

- `Krita Style` (`JAX_KritaStyle`)
  - Krita-managed style/sampler preset outputs. Intended for workflows started from Krita.

- `Krita Strength` (`JAX_KritaStrength`)
  - Given `sigmas`, a `strength`, and a `denoise` value, computes an integer step index that approximates Krita-style strength control for diffusion, clamping values to safe ranges.
