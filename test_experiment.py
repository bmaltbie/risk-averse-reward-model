"""
Quick Test for Risk-Averse Reward Model Training

This test script runs a minimal version of the experiment for quick end-to-end testing.
Uses test_data.csv with only 5 scenarios and minimal training parameters.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
import json
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Import classes from main experiment
from risk_averse_experiment import (
    RiskAversionDataLoader,
    PairwiseRiskAversionDataset,
    PairwiseDataCollator,
    RiskAverseRewardModel,
    evaluate_model
)

def run_quick_test():
    """Run a quick test of the experiment with minimal parameters"""
    print("=== Quick Test - Risk-Averse Reward Model ===")
    
    # Test data loading
    print("\n1. Testing data loading...")
    loader = RiskAversionDataLoader("data/test_data.csv")
    try:
        dataset_df = loader.load_and_process_data()
        print(f"✓ Successfully loaded {len(dataset_df)} test scenarios")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Check if we have enough data
    if len(dataset_df) < 3:
        print(f"✗ Not enough data for testing (need at least 3 scenarios, got {len(dataset_df)})")
        return False
    
    # Test model initialization
    print("\n2. Testing model initialization...")
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use TinyLlama for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = RiskAverseRewardModel(model_name)
        print(f"✓ Model initialized successfully")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    
    # Test dataset creation
    print("\n3. Testing dataset creation...")
    try:
        # Use minimal split - just 2 scenarios for train, 1 for validation
        if len(dataset_df) >= 4:
            train_df = dataset_df.iloc[:3]  # Use first 3 for training
            val_df = dataset_df.iloc[3:]    # Use rest for validation
        else:
            train_df = dataset_df.iloc[:2]  # Use first 2 for training
            val_df = dataset_df.iloc[2:]    # Use last 1 for validation
        
        train_dataset = PairwiseRiskAversionDataset(train_df, tokenizer, max_length=128)  # Keep shorter for testing
        val_dataset = PairwiseRiskAversionDataset(val_df, tokenizer, max_length=128)
        
        print(f"✓ Datasets created: {len(train_dataset)} train, {len(val_dataset)} validation situation pairs")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False
    
    # Test training setup
    print("\n4. Testing training setup...")
    try:
        training_args = TrainingArguments(
            output_dir="./test_model",
            num_train_epochs=1,  # Just 1 epoch for quick test
            per_device_train_batch_size=1,  # Smallest possible batch
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,   # Small accumulation for test
            warmup_steps=1,
            weight_decay=0.01,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=2,
            save_steps=2,                    # Must be multiple of eval_steps
            load_best_model_at_end=False,    # Disable for pairwise training (no eval_loss available)
            fp16=torch.cuda.is_available(),  # Only enable fp16 on CUDA, not MPS
            dataloader_pin_memory=False,    # Reduce memory overhead
            report_to=[],  # Disable wandb/tensorboard logging
        )
        
        data_collator = PairwiseDataCollator(tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print("✓ Training setup completed successfully")
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        return False
    
    # Test training (just a few steps)
    print("\n5. Testing training...")
    try:
        # Run minimal training
        trainer.train()
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Test evaluation
    print("\n6. Testing evaluation...")
    try:
        accuracy, eval_results = evaluate_model(model, tokenizer, val_df, return_detailed=True)
        print(f"✓ Evaluation completed successfully - Accuracy: {accuracy:.3f}")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False
    
    # Test inference on a single example
    print("\n7. Testing inference...")
    try:
        test_row = dataset_df.iloc[0]
        model.eval()
        
        with torch.no_grad():
            # Get the device the model is on
            device = next(model.parameters()).device
            
            # Test risk-averse option
            correct_input = f"{test_row['prompt_text']}\\n\\nChosen option: {test_row['correct_label']}"
            encoding = tokenizer(correct_input, truncation=True, padding='max_length', 
                               max_length=128, return_tensors='pt')
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            correct_score = torch.sigmoid(outputs["logits"]).item()
            
            # Test risk-neutral option
            incorrect_input = f"{test_row['prompt_text']}\\n\\nChosen option: {test_row['incorrect_label']}"
            encoding = tokenizer(incorrect_input, truncation=True, padding='max_length',
                               max_length=128, return_tensors='pt')
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            incorrect_score = torch.sigmoid(outputs["logits"]).item()
            
            print(f"✓ Inference successful:")
            print(f"  Risk-averse option {test_row['correct_label']}: {correct_score:.3f}")
            print(f"  Risk-neutral option {test_row['incorrect_label']}: {incorrect_score:.3f}")
            
            if correct_score > incorrect_score:
                print("  ✓ Model prefers risk-averse option")
            else:
                print("  ⚠ Model prefers risk-neutral option (may need more training)")
                
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    # Test plotting
    print("\n8. Testing plotting...")
    try:
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Create a simple test plot
        plt.figure(figsize=(8, 6))
        
        # Plot training loss if available
        if hasattr(trainer, 'state') and trainer.state.log_history:
            train_losses = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
            if train_losses:
                plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Test Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save test plot to outputs directory
        test_plot_filename = "outputs/test_training_plot.png"
        plt.savefig(test_plot_filename, dpi=150, bbox_inches='tight')
        plt.close()  # Close to avoid display issues
        
        print(f"✓ Plotting successful - saved to {test_plot_filename}")
        
        # Show evaluation metrics
        if eval_results:
            risk_averse_scores = eval_results['risk_averse_scores']
            risk_neutral_scores = eval_results['risk_neutral_scores']
            if risk_averse_scores and risk_neutral_scores:
                avg_diff = np.mean(risk_averse_scores) - np.mean(risk_neutral_scores)
                print(f"  Average score difference: {avg_diff:+.3f}")
        
    except Exception as e:
        print(f"✗ Plotting failed: {e}")
        return False
    
    # Cleanup
    print("\n9. Cleaning up...")
    try:
        import shutil
        if os.path.exists("./test_model"):
            shutil.rmtree("./test_model")
        # Clean up test plot from outputs directory
        if os.path.exists("outputs/test_training_plot.png"):
            os.remove("outputs/test_training_plot.png")
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")
    
    print("\n=== Quick Test PASSED ===")
    print("All components are working correctly!")
    return True

if __name__ == "__main__":
    # Check if test data exists
    if not os.path.exists("data/test_data.csv"):
        print("Error: data/test_data.csv not found. Please ensure the test data file is present.")
        sys.exit(1)
    
    success = run_quick_test()
    if not success:
        print("\n=== Quick Test FAILED ===")
        print("Please check the error messages above.")
        sys.exit(1)