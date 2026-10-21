import pandas as pd
import json
from pathlib import Path
import logging

class MedicalBiasDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = Path(cfg.dataset.data_path)
        self.images_path = Path(cfg.dataset.images_path)
        
        # Load CSV files
        self.questions_df = pd.read_csv(self.data_path / cfg.dataset.questions_file)
        self.images_df = pd.read_csv(self.data_path / cfg.dataset.images_file)
        self.mapping_df = pd.read_csv(self.data_path / cfg.dataset.mapping_file)
        
        logging.info(f"Loaded {len(self.questions_df)} questions, {len(self.images_df)} images, {len(self.mapping_df)} mappings")

    def generate_full_question(self, question, question_type):
        """Add type-specific specifications and additional context to questions based on config settings"""
        full_question = question

        # Add question prefix if configured (e.g., rare condition hint)
        if self.cfg.dataset.question_prefix:
            full_question = self.cfg.dataset.question_prefix + full_question

        # Add question suffix if configured (e.g., rare condition hint)
        if self.cfg.dataset.question_suffix:
            full_question = full_question + " " + self.cfg.dataset.question_suffix

        # Add question specifications if enabled
        if self.cfg.dataset.use_question_specifications:
            specifications = self.cfg.dataset.question_specifications
            spec = specifications[question_type]
            full_question = f"{full_question} {spec}"

        return full_question

    def create_qa_pairs(self):
        """Create question-answer pairs by joining the three datasets"""
        qa_pairs = []
        
        for _, mapping_row in self.mapping_df.iterrows():
            # Get question
            question_row = self.questions_df[
                self.questions_df['question_uuid'] == mapping_row['question_uuid']
            ]
            if question_row.empty:
                raise ValueError(f"Question UUID {mapping_row['question_uuid']} not found")
            
            # Get image
            image_row = self.images_df[
                self.images_df['image_uuid'] == mapping_row['image_uuid']
            ]
            if image_row.empty:
                raise ValueError(f"Image UUID {mapping_row['image_uuid']} not found")
                
            question = question_row.iloc[0]['question']
            question_type = question_row.iloc[0]['question_type']
            full_question = self.generate_full_question(question, question_type)
            full_image_path = self.images_path / image_row.iloc[0]['filepath']
            
            # Check if image file exists
            if not full_image_path.exists():
                raise FileNotFoundError(f"Image file not found: {full_image_path}")
            
            qa_pair = {
                'mapping_uuid': mapping_row['mapping_uuid'],
                'question_uuid': mapping_row['question_uuid'],
                'image_uuid': mapping_row['image_uuid'],
                'question': full_question,
                'img_path': str(full_image_path),
                'gt_answer': str(mapping_row['gt_answer']),
                'exp_bias_answer': str(mapping_row['exp_bias_answer']),
                'question_type': question_row.iloc[0]['question_type'],
                'failure_mode': question_row.iloc[0]['failure_mode'],
                'reference': str(image_row.iloc[0]['typical'])
            }
            
            qa_pairs.append(qa_pair)
        
        logging.info(f"Created {len(qa_pairs)} valid QA pairs")
        return qa_pairs

def get_medical_bias_qa_pairs(cfg):
    """Main function to get QA pairs for medical bias benchmark"""
    dataset = MedicalBiasDataset(cfg)
    return dataset.create_qa_pairs()