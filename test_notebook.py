#!/usr/bin/env python3
"""
Test script to verify notebook functionality
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import numpy as np
import pandas as pd
import re
from collections import Counter

from hypothesaes.quickstart import train_sae, interpret_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_local_embeddings

print("✅ Basic imports successful")

# Test model loading
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    model_path = '/data/qingpengkong/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1'
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={'': 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: '8GB'},
    )
    
    # Create wrapper
    class TransformersEngine:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            
        def generate(self, prompts, **kwargs):
            results = []
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors='pt')
                if self.device == 'cuda':
                    inputs = inputs.to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_length=kwargs.get('max_tokens', 100) + inputs['input_ids'].shape[1],
                        do_sample=True,
                        temperature=kwargs.get('temperature', 0.7),
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                results.append(response)
            return results
    
    engine = TransformersEngine(model, tokenizer, 'cuda')
    print("✅ Model and engine created successfully!")
    
    # Test generation
    test_prompt = 'What is cancer?'
    response = engine.generate([test_prompt], max_tokens=50, temperature=0.7)[0]
    print(f"Test response: {response}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()



