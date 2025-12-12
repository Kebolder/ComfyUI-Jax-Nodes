from comfy_api.latest import io
import re

from comfy import sd as comfy_sd
import comfy.utils as comfy_utils
import folder_paths


_LORA_PATTERN = re.compile(r"<lora:([^:<>]+)(?::(-?[^:<>]*))?>", re.IGNORECASE)


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
        text = text or ""

        def _replace(match):
            nonlocal model, clip

            if model is None and clip is None:
                return match.group(0)

            lora_name = match.group(1).strip()
            strength_model = 1.0
            strength_clip = 1.0

            strength_str = match.group(2)
            if strength_str is not None and strength_str.strip() != "":
                raw = strength_str.strip()
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

            lora_path = None
            base_name = lora_name.replace("\\", "/").split("/")[-1]
            candidates = [
                lora_name,
                f"{lora_name}.safetensors",
                base_name,
                f"{base_name}.safetensors",
            ]
            for candidate in candidates:
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", candidate)
                    break
                except Exception:
                    continue

            if lora_path is None:
                print(f"[EasyPrompt] LoRA not found for tag {match.group(0)!r}")
                return match.group(0)

            try:
                lora = comfy_utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy_sd.load_lora_for_models(
                    model,
                    clip,
                    lora,
                    strength_model,
                    strength_clip,
                )
            except Exception as e:
                print(f"[EasyPrompt] Failed to load LoRA {lora_name!r}: {e}")
                return match.group(0)

            return ""

        cleaned_text = _LORA_PATTERN.sub(_replace, text).strip()
        return model, clip, cleaned_text

    @staticmethod
    def _encode_clip(clip, text: str):
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

