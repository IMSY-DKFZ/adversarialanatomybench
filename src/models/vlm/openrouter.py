import io
import base64
import torch.nn as nn
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
load_dotenv("api_keys.env")
api_key = os.getenv('OPENROUTER_API_KEY')


class OpenRouter(nn.Module):
    # Provider routing configuration

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_params = {
            "hf_id": self.cfg.model.hf_id,
            "temperature": self.cfg.model.temperature
        }
        self.reasoning_effort = self.cfg.model['reasoning_effort']
        self.reasoning_max_tokens = self.cfg.model['reasoning_max_tokens']
        self.max_new_tokens = self.cfg.model['max_new_tokens']

        # Ensure only one reasoning method is configured
        assert not (self.reasoning_max_tokens is not None and self.reasoning_effort is not None)

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )

    async def _query_single(self, image, question):
        """Query a single image-question pair asynchronously"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64_str = base64.b64encode(img_bytes).decode('utf-8')

        # Build reasoning config
        reasoning_config = {}
        if self.reasoning_max_tokens is not None:
            assert self.reasoning_max_tokens >= 1
            reasoning_config["max_tokens"] = int(self.reasoning_max_tokens)
        elif self.reasoning_effort is not None:
            reasoning_config["effort"] = self.reasoning_effort

        # Build API call kwargs
        kwargs = {
            "model": self.model_params["hf_id"],
            "temperature": self.model_params["temperature"],
            "max_tokens": self.max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_base64_str}"
                        }
                    ]
                }
            ]
        }

        # Build extra_body with reasoning and provider configs
        extra_body = {}
        if reasoning_config:
            extra_body["reasoning"] = reasoning_config

        if extra_body:
            kwargs["extra_body"] = extra_body

        completion = await self.client.chat.completions.create(**kwargs)
        if completion.choices == None:
            return {"answer": "empty_return", "usage": None, "reasoning": None, "provider": None}
        else:
            # Extract usage information
            usage_info = None
            if completion.usage:
                usage_info = {
                    "completion_tokens": completion.usage.completion_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
                # Add reasoning tokens if available
                if hasattr(completion.usage, 'completion_tokens_details') and completion.usage.completion_tokens_details:
                    if hasattr(completion.usage.completion_tokens_details, 'reasoning_tokens'):
                        usage_info["reasoning_tokens"] = completion.usage.completion_tokens_details.reasoning_tokens

            # Extract reasoning trace if available
            reasoning_trace = None
            if hasattr(completion.choices[0].message, 'model_extra') and completion.choices[0].message.model_extra:
                reasoning_trace = completion.choices[0].message.model_extra.get('reasoning')

            # Extract provider information if available
            provider_info = None
            if hasattr(completion, 'model_extra') and completion.model_extra:
                provider_info = completion.model_extra.get('provider')

            return {
                "answer": completion.choices[0].message.content,
                "usage": usage_info,
                "reasoning": reasoning_trace,
                "provider": provider_info
            }

    def forward(self, images, questions):
        """Forward pass with parallel API calls"""
        async def _forward_async():
            tasks = [self._query_single(img, q) for img, q in zip(images, questions)]
            return await asyncio.gather(*tasks)

        return asyncio.run(_forward_async())
