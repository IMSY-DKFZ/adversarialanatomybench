import torch
import torch.nn as nn
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, pipeline

class Gemma3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_params = {
            "hf_id": self.cfg.model.hf_id,
        }
        self.device = cfg.model.device
        self._init_model()

    def _init_model(self):
        # Completely disable torch.compile to avoid generator tracing issues
        torch._dynamo.config.disable = True
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_params["hf_id"],
            device=self.device,
            dtype=torch.bfloat16
        )

    def forward(self, images, questions):
        outputs = []
        for image, question in zip(images, questions):
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question}
                    ]
                }]

            output = self.pipe(messages, max_new_tokens=self.cfg.model.max_new_tokens, temperature=self.cfg.model.temperature)
            outputs.append(output[0]["generated_text"][-1]["content"])
        return outputs
