from __future__ import annotations

from typing import Any
from io import BytesIO

import numpy as np
from PIL import Image
import torch

from server import PromptServer, BinaryEventTypes
from comfy_api.latest import io


class SendImageWebSocket(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JAX_SendImageWebSocket",
            display_name="Send Image (WebSocket)",
            category="krita_internal",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("format", options=["PNG", "JPEG"], default="PNG"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images: torch.Tensor, format: str):
        results: list[dict[str, Any]] = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                [format, image, None],
                server.client_id,
            )
            results.append(
                {
                    "source": "websocket",
                    "content-type": f"image/{format.lower()}",
                    "type": "output",
                }
            )

        return io.NodeOutput(ui={"images": results})

