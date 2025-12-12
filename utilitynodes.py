from comfy_api.latest import io


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
