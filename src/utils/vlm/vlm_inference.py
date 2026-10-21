import json
from datetime import datetime
from pathlib import Path
import logging
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

class VLMInferenceEngine():
    def __init__(self, cfg, model):
            self.cfg = cfg
            self.model = model

    def _resize_image_if_needed(self, img):
        """
        Resize image if needed based on configuration settings.

        Args:
            img (PIL.Image): Input image

        Returns:
            PIL.Image: Resized image or original if no resize needed
        """
        # Check if resizing is enabled
        if not self.cfg.vlm.image_processing.resize.enabled: 
            return img

        resize_config = self.cfg.vlm.image_processing.resize
        max_size = resize_config["max_size"]

        # Get image dimensions
        width, height = img.size

        # Check if resizing is needed
        if width <= max_size and height <= max_size:
            return img

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)

        # Resize the image with high-quality resampling
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        logging.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        return resized_img

    def infer(self, qa_pairs):
        logging.info(f"Started inference process with model {self.cfg.model.hf_id}\n\n".upper())

        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # Extract company folder from base_model (e.g., "openai.OpenAIAPI" -> "openai")
        company_folder = self.cfg.model.base_model.split('.')[0]
        model_filename = self.cfg.model.hf_id.replace('/', '_').replace('.', '_')

        output_dict = {
            'metadata': OmegaConf.to_container(self.cfg, resolve=True),
            'model_id': self.cfg.model.hf_id,
            'dataset': self.cfg.dataset.name,
            'qa_pairs': qa_pairs
        }

        # Get batch size from config, default to 1 for sequential processing
        # Only use batching for OpenRouter models (they have parallel async implementation)
        is_openrouter = company_folder == 'openrouter'
        batch_size = self.cfg.vlm["batch_size"] if is_openrouter else 1

        # Process in batches
        for batch_start in tqdm(range(0, len(qa_pairs), batch_size)):
            batch_end = min(batch_start + batch_size, len(qa_pairs))
            batch = qa_pairs[batch_start:batch_end]

            # Collect images and questions for this batch
            images = []
            questions = []
            batch_indices = []

            for idx, qa_pair in enumerate(batch):
                if "model_answer" not in qa_pair:
                    question = qa_pair["question"]
                    with Image.open(qa_pair["img_path"]) as img:
                        # Apply dynamic resizing if configured
                        img = self._resize_image_if_needed(img)
                        # Create a copy to use outside the context manager
                        img_copy = img.copy()
                        images.append(img_copy)
                    questions.append(question)
                    batch_indices.append(idx)

            outputs = self.model(images, questions)

            # Assign outputs back to qa_pairs
            for idx, output in zip(batch_indices, outputs):
                batch[idx]['model'] = self.cfg.model.hf_id

                # Handle dict output (e.g., from OpenRouter with usage info)
                if isinstance(output, dict) and 'answer' in output:
                    batch[idx]['model_answer'] = output['answer']
                    if output.get('usage'):
                        batch[idx]['usage'] = output['usage']
                        # Log usage information
                        logging.debug(f"Token usage for Q{batch_start + idx}: {output['usage']}")
                    if output.get('reasoning'):
                        batch[idx]['reasoning'] = output['reasoning']
                        # Log that reasoning was captured
                        logging.debug(f"Reasoning trace captured for Q{batch_start + idx}")
                    if output.get('provider'):
                        batch[idx]['provider'] = output['provider']
                        # Log provider information
                        logging.debug(f"Provider used for Q{batch_start + idx}: {output['provider']}")
                else:
                    # Handle string output (from other models)
                    batch[idx]['model_answer'] = output

            # Print outputs if configured
            for idx, qa_pair in enumerate(batch):
                if self.cfg.paths.vlm.print_inference_outputs and "model_answer" in qa_pair:
                    logging.info(f"Q: {qa_pair['question']}")
                    logging.info(f"gt: {qa_pair['gt_answer']}")
                    logging.info(f"Model: {qa_pair['model_answer']}\n")
                print(batch_start + idx, end=" ")

        # Retry failed requests
        error_indices = [i for i, qa in enumerate(qa_pairs) if qa.get('model_answer') == 'empty_return']
        if error_indices:
            logging.info(f"\n\nRetrying {len(error_indices)} failed requests...")
            print(f"\nRetrying {len(error_indices)} errors: ", end="")

            for retry_idx in error_indices:
                qa_pair = qa_pairs[retry_idx]
                question = qa_pair["question"]

                with Image.open(qa_pair["img_path"]) as img:
                    img = self._resize_image_if_needed(img)
                    img_copy = img.copy()
                    outputs = self.model([img_copy], [question])

                # Handle dict output (e.g., from OpenRouter with usage info)
                output = outputs[0]
                if isinstance(output, dict) and 'answer' in output:
                    qa_pair['model_answer'] = output['answer']
                    if output.get('usage'):
                        qa_pair['usage'] = output['usage']
                        logging.debug(f"Token usage for retry Q{retry_idx}: {output['usage']}")
                    if output.get('reasoning'):
                        qa_pair['reasoning'] = output['reasoning']
                        logging.debug(f"Reasoning trace captured for retry Q{retry_idx}")
                else:
                    qa_pair['model_answer'] = output
                print(retry_idx, end=" ")

                if self.cfg.paths.vlm.print_inference_outputs:
                    logging.info(f"RETRY Q: {qa_pair['question']}")
                    logging.info(f"RETRY Model: {qa_pair['model_answer']}\n")

            # Count remaining errors after retry
            final_errors = sum(1 for qa in qa_pairs if qa.get('model_answer') == 'empty_return')
            if final_errors > 0:
                logging.warning(f"\n{final_errors} requests still failed after retry")

        # Use metatask_id from dataset config if available, otherwise fall back to vlm config
        metatask_id = getattr(self.cfg.dataset, 'metatask_id', self.cfg.vlm.metatask_id)
        qa_save_path = Path(self.cfg.paths.vlm.result_save_path, metatask_id, company_folder, f"{model_filename}_{self.cfg.dataset.name}_len{len(qa_pairs)}_time{current_time}" + ".json")
        qa_save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(qa_save_path, 'w') as f:
            json.dump(output_dict, f, indent=4)

        logging.info(f"Saved model outputs to {qa_save_path}")
        return qa_save_path