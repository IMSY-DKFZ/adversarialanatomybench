import torch.nn as nn
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

class Qwen2_5VL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_params = {
            "hf_id": self.cfg.model.hf_id,
        }
        self.device = cfg.model.device
        self._init_model()

    def _init_model(self):
        if self.cfg.model.attn_impl == "flash-attn":
            attn_impl = "flash_attention_2"
        else:
            attn_impl = self.cfg.model.attn_impl
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_params["hf_id"],
            dtype="auto",
            attn_implementation=attn_impl,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_params["hf_id"])

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

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.cfg.model.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs.append(output_text)
        return outputs
