from comfy_api.latest import ComfyExtension, io

from .utilitynodes import (
    ConditionalSelect,
    SamplerPipeIn,
    SamplerPipeOut,
    ImageSizeMultiplier,
)
from .conditioningnodes import SimpleLoraPromptEncoder, LoraPromptEncoder
from .kritanodes import (
    KritaOutput,
    KritaSendText,
    KritaCanvas,
    KritaSelection,
    KritaImageLayer,
    KritaMaskLayer,
    Parameter,
    KritaStyle,
    KritaStrength,
)


class JaxNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ConditionalSelect,
            SamplerPipeIn,
            SamplerPipeOut,
            ImageSizeMultiplier,
            SimpleLoraPromptEncoder,
            LoraPromptEncoder,
            KritaOutput,
            KritaSendText,
            KritaCanvas,
            KritaSelection,
            KritaImageLayer,
            KritaMaskLayer,
            Parameter,
            KritaStyle,
            KritaStrength,
        ]


async def comfy_entrypoint():
    return JaxNodesExtension()
