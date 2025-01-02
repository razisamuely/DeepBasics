import os
import sys
import unittest
import matplotlib.pyplot as plt
import h5py
import subprocess

class TestLSTMAutoencoder(unittest.TestCase):
    def setUp(self):
        # Ensure the artifacts directory exists
        os.makedirs('RNNBasics/artifacts', exist_ok=True)
        
    def test_data_visualization(self):
        """Test running data visualization script"""
        print("\nTest 1: Running data visualization script...")
        
        cmd = [
            sys.executable,
            'RNNBasics/src/lstm_ae_toy_3_1/lstm_ae_toy_3_1_1.py'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Verify the visualization was created
            self.assertTrue(os.path.exists('RNNBasics/artifacts/synthetic_data_3_1_1.png'),
                          "Visualization file not created")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running visualization script: {e}")
            print(f"Script output: {e.output}")
            raise
        
        print("Test 1 completed successfully")

    def test_lstm_training(self):
        """Test LSTM autoencoder training with 1 trial"""
        print("\nTest 2: Training LSTM autoencoder with 1 trial...")
        
        # Run training script with 1 trial
        cmd = [
            sys.executable,
            'RNNBasics/src/lstm_ae_toy_3_1/lstm_ae_toy_3_1_2.py',
            '--n-trials', '1',
            '--n-epochs', '2'  # Reduced epochs for testing
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Check if results directory contains expected files
            results_path = 'RNNBasics/artifacts/results_3_1_2'
            self.assertTrue(os.path.exists(results_path), "Results directory not created")
            
            # Basic check for output files
            self.assertTrue(any(fname.endswith('.pth') for fname in os.listdir(results_path)),
                          "No model file found")
            self.assertTrue(any(fname.endswith('.json') for fname in os.listdir(results_path)),
                          "No results JSON found")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running training script: {e}")
            print(f"Script output: {e.output}")
            raise
        
        print("Test 2 completed successfully")

if __name__ == '__main__':
    unittest.main()