import os
import sys
import unittest
import subprocess

class TestMNISTReconstruction(unittest.TestCase):
    def setUp(self):
        os.makedirs('RNNBasics/artifacts', exist_ok=True)
        
    def test_row_by_row_reconstruction(self):
        cmd = [
            sys.executable,
            'RNNBasics/src/mnsit_reconstruction_3_2/reconstruct_mnist_by_row_3_2_1.py',
            '--n-trials', '1',
            '--n-epochs', '1'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            artifacts_path = 'RNNBasics/artifacts/mnist_3_2_1'
            self.assertTrue(os.path.exists(artifacts_path))
            
            self.assertTrue(os.path.exists(os.path.join(artifacts_path, 'reconstructions_final.png')))
            self.assertTrue(os.path.exists(os.path.join(artifacts_path, 'lstm_ae_mnist.pt')))
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Output: {e.output}")
            raise

    def test_row_by_row_with_classification(self):
        cmd = [
            sys.executable,
            'RNNBasics/src/mnsit_reconstruction_3_2/reconstruct_and_classify_mnist_3_2_2.py',
            '--n-trials', '1',
            '--n-epochs', '1'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            artifacts_path = 'RNNBasics/artifacts/mnist_3_2_2'
            self.assertTrue(os.path.exists(artifacts_path))
            
            self.assertTrue(os.path.exists(os.path.join(artifacts_path, 'reconstructions_final.png')))
            self.assertTrue(os.path.exists(os.path.join(artifacts_path, 'training_history.png')))
            self.assertTrue(os.path.exists(os.path.join(artifacts_path, 'lstm_ae_classifier.pt')))
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Output: {e.output}")
            raise

    def test_pixel_by_pixel(self):
        cmd = [
            sys.executable,
            'RNNBasics/src/mnsit_reconstruction_3_2/reconstruct_mnist_by_pixel_3_2_3.py',
            '--n-trials', '1',
            '--n-epochs', '1'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            artifacts_path = 'RNNBasics/artifacts/mnist_3_2_2'
            self.assertTrue(os.path.exists(artifacts_path))
            
            self.assertTrue(os.path.exists(os.path.join(
                artifacts_path, 'reconstructions_final_pixel_by_pixel_3_2_3.png')))
            self.assertTrue(os.path.exists(os.path.join(
                artifacts_path, 'training_history_pixel_by_pixel_3_2_3.png')))
            self.assertTrue(os.path.exists(os.path.join(artifacts_path, 'lstm_ae_classifier.pt')))
            
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print(f"Output: {e.output}")
            raise

if __name__ == '__main__':
    unittest.main()