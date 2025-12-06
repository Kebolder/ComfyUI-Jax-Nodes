from comfy_api.latest import ComfyExtension, io
import re

from comfy import sd as comfy_sd
import comfy.utils as comfy_utils
import folder_paths


_LORA_PATTERN = re.compile(r"<lora:([^:>]+?)(?::([^>]+))?>", re.IGNORECASE)


class ConditionalSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_ConditionalSelect",
            display_name="Conditional Select",
            category="logic",
            inputs=[
                io.Boolean.Input("condition"),
                io.AnyType.Input("true_value"),
                io.AnyType.Input("false_value"),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def execute(cls, condition: bool, true_value, false_value):
        return io.NodeOutput(true_value if condition else false_value)


class SamplerPipeIn(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_KritaPipeIn",
            display_name="Sampler Pipe In",
            category="Utility",
            inputs=[
                io.Model.Input("model", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Latent.Input("latent", optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="pipe")],
        )

    @classmethod
    def execute(
        cls,
        model=None,
        positive=None,
        negative=None,
        vae=None,
        image=None,
        clip=None,
        latent=None,
    ):
        bundle = {}
        if model is not None:
            bundle["model"] = model
        if positive is not None:
            bundle["positive"] = positive
        if negative is not None:
            bundle["negative"] = negative
        if vae is not None:
            bundle["vae"] = vae
        if image is not None:
            bundle["image"] = image
        if clip is not None:
            bundle["clip"] = clip
        if latent is not None:
            bundle["latent"] = latent
        return io.NodeOutput(bundle)


class KritaStrength(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_KritaStrength",
            display_name="Krita Strength",
            category="Utility",
            inputs=[
                io.Sigmas.Input("sigmas"),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0),
            ],
            outputs=[io.Int.Output(display_name="Step")],
        )

    @classmethod
    def execute(cls, sigmas, strength: float, denoise: float):
        strength = max(0.0, min(1.0, float(strength)))
        denoise = max(0.0, min(1.0, float(denoise)))

        total = len(sigmas)
        if total <= 1:
            print("[KritaStrength] Not enough sigmas")
            return io.NodeOutput(0)

        logical_steps = total - 1
        effective = max(1, round(logical_steps * denoise))

        step = round(effective * (1.0 - strength))
        step = max(0, min(step, logical_steps))

        print(
            f"[KritaStrength] total={total}, logical_steps={logical_steps}, "
            f"denoise={denoise}, strength={strength}, effective={effective}, step={step}"
        )

        return io.NodeOutput(step)


class KritaResizeCanvas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_KritaResizeCanvas",
            display_name="Krita Resize Canvas",
            category="Utility",
            inputs=[
                io.Int.Input("width", default=1024),
                io.Int.Input("height", default=1024),
                io.Boolean.Input("enabled", default=True),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, width: int, height: int, enabled: bool):
        if not enabled:
            return io.NodeOutput()

        payload = {
            "action": "resize_canvas",
            "width": int(width),
            "height": int(height),
        }

        return io.NodeOutput(
            ui={
                "text": [
                    {
                        "name": "Krita Resize Canvas",
                        "text": json.dumps(payload),
                        "content-type": "application/x-krita-command",
                    }
                ]
            }
        )


class SamplerPipeOut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_SamplerPipeOut",
            display_name="Sampler Pipe Out",
            category="Utility",
            inputs=[io.AnyType.Input("pipe", optional=True)],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Vae.Output(display_name="vae"),
                io.Image.Output(display_name="image"),
                io.Clip.Output(display_name="clip"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, pipe=None):
        pipe = pipe or {}
        model = pipe.get("model")
        positive = pipe.get("positive")
        negative = pipe.get("negative")
        vae = pipe.get("vae")
        image = pipe.get("image")
        clip = pipe.get("clip")
        latent = pipe.get("latent")
        return io.NodeOutput(model, positive, negative, vae, image, clip, latent)


class ModeSelect(io.ComfyNode):
    pass


class ImageSizeMultiplier(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_ImageSizeMultiplier",
            display_name="Image Size Multiplier",
            category="Utility",
            inputs=[
                io.Int.Input("width"),
                io.Int.Input("height"),
                io.Float.Input("multiplier", default=1.0, min=0.0, max=10.0),
            ],
            outputs=[
                io.Int.Output(display_name="Width"),
                io.Int.Output(display_name="Height"),
                io.String.Output(display_name="Final Size"),
            ],
        )

    @classmethod
    def execute(cls, width: int, height: int, multiplier: float):
        new_width = int(width * multiplier)
        new_height = int(height * multiplier)
        size_md = f"### Image Size:\nWidth: {new_width}\nHeight: {new_height}"
        return io.NodeOutput(new_width, new_height, size_md)


class LoraPromptEncoder(io.ComfyNode):
    @staticmethod
    def _merge(base: str, extra: str):
        base = (base or "").strip()
        extra = (extra or "").strip()
        if not base:
            return extra
        if not extra:
            return base
        return f"{base}, {extra}"

    @staticmethod
    def _apply_loras(model, clip, text: str):
        matches = _LORA_PATTERN.findall(text or "")
        cleaned_text = _LORA_PATTERN.sub("", text or "").strip()

        if matches and (model is not None or clip is not None):
            for name, strength_str in matches:
                lora_name = name.strip()
                strength_model = 1.0
                strength_clip = 1.0

                if strength_str is not None and strength_str.strip() != "":
                    raw = strength_str.strip()
                    # Support formats like "0.8", "0.8,0.5" (model,clip)
                    parts = [p.strip() for p in raw.split(",") if p.strip()]

                    def _to_float(val: str, default: float = 1.0) -> float:
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            return default

                    if len(parts) == 1:
                        strength_model = strength_clip = _to_float(parts[0], 1.0)
                    elif len(parts) >= 2:
                        strength_model = _to_float(parts[0], 1.0)
                        strength_clip = _to_float(parts[1], strength_model)
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                except Exception:
                    try:
                        lora_path = folder_paths.get_full_path_or_raise("loras", f"{lora_name}.safetensors")
                    except Exception:
                        # If still not found, skip this LoRA
                        continue

                lora = comfy_utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy_sd.load_lora_for_models(
                    model,
                    clip,
                    lora,
                    strength_model,
                    strength_clip,
                )

        return model, clip, cleaned_text

    @staticmethod
    def _encode_clip(clip, text: str):
        # Return an empty conditioning list if we have nothing to encode.
        if clip is None:
            return []
        text = (text or "").strip()
        if not text:
            return []
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_EasyPrompt",
            display_name="Easy Prompt (W/ Append)",
            description=(
                "All-in-one CLIP text encoder that supports loading LoRAs via "
                "<lora:name:weight> tags and appending extra tags (e.g. Krita-style prompts)."
            ),
            category="conditioning",
            inputs=[
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip"),
                io.String.Input("positive", multiline=True),
                io.String.Input("negative", multiline=True),
                io.String.Input("append_positive", multiline=True),
                io.String.Input("append_negative", multiline=True),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.Conditioning.Output(display_name="Positive"),
                io.Conditioning.Output(display_name="Negative"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model=None,
        clip=None,
        positive: str = "",
        negative: str = "",
        append_positive: str = "",
        append_negative: str = "",
    ):
        pos_merged = cls._merge(positive, append_positive)
        neg_merged = cls._merge(negative, append_negative)

        model, clip, pos_clean = cls._apply_loras(model, clip, pos_merged)
        model, clip, neg_clean = cls._apply_loras(model, clip, neg_merged)

        pos_cond = cls._encode_clip(clip, pos_clean)
        neg_cond = cls._encode_clip(clip, neg_clean)

        return io.NodeOutput(model, pos_cond, neg_cond)


class SimpleLoraPromptEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_EasyPromptSimple",
            display_name="Easy Prompt",
            description=(
                "CLIP text encoder that supports loading LoRAs via "
                "<lora:name:weight> tags."
            ),
            category="conditioning",
            inputs=[
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip"),
                io.String.Input("positive", multiline=True),
                io.String.Input("negative", multiline=True),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.Conditioning.Output(display_name="Positive"),
                io.Conditioning.Output(display_name="Negative"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model=None,
        clip=None,
        positive: str = "",
        negative: str = "",
    ):
        model, clip, pos_clean = LoraPromptEncoder._apply_loras(model, clip, positive)
        model, clip, neg_clean = LoraPromptEncoder._apply_loras(model, clip, negative)

        pos_cond = LoraPromptEncoder._encode_clip(clip, pos_clean)
        neg_cond = LoraPromptEncoder._encode_clip(clip, neg_clean)

        return io.NodeOutput(model, pos_cond, neg_cond)


class KritaModeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ConditionalSelect,
            SamplerPipeIn,
            SamplerPipeOut,
            ImageSizeMultiplier,
            SimpleLoraPromptEncoder,
            LoraPromptEncoder,
            KritaStrength,
            KritaResizeCanvas,
        ]


async def comfy_entrypoint():
    return KritaModeExtension()