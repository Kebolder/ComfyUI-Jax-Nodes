from comfy_api.latest import io


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
