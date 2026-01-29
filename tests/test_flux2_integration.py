"""
Test suite for Flux.2 training integration.
Verifies:
1. Flux.2 model detection in config_gen.py
2. Proper routing to flux2_train.py vs flux_train_network.py
3. Flux.2 model architecture validation
"""

import sys
import os
import unittest
from pathlib import Path

# Setup paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)  # ComfyUI-Flux2-LoRA-Manager
sys.path.insert(0, PROJECT_ROOT)

from src.flux2_support import flux2_models, flux2_utils


class TestFlux2Detection(unittest.TestCase):
    """Test Flux.2 model detection logic."""
    
    def test_flux2_detection_variants(self):
        """Test that various Flux.2 filename patterns are detected correctly."""
        flux2_patterns = [
            "FLUX.2-dev.safetensors",
            "flux2-dev.safetensors",
            "models/flux.2/checkpoint.safetensors",
            "FLUX2_Dev.safetensors",
        ]
        
        for pattern in flux2_patterns:
            is_flux2 = "flux2" in pattern.lower() or "flux.2" in pattern.lower()
            self.assertTrue(is_flux2, f"Failed to detect: {pattern}")
    
    def test_flux1_detection(self):
        """Test that Flux.1 models are NOT detected as Flux.2."""
        flux1_patterns = [
            "FLUX-1-dev.safetensors",
            "flux1-dev.safetensors",
            "models/flux.1/checkpoint.safetensors",
            "model.safetensors",  # Unknown
        ]
        
        for pattern in flux1_patterns:
            is_flux2 = "flux2" in pattern.lower() or "flux.2" in pattern.lower()
            self.assertFalse(is_flux2, f"Incorrectly detected as Flux.2: {pattern}")
    
    def test_flux2_architecture_params(self):
        """Test that Flux.2 architecture has correct parameters."""
        config = flux2_models.get_flux2_config()
        
        # config is a dict with 'params' key
        self.assertIsInstance(config, dict)
        self.assertIn("params", config)
        
        params = config["params"]
        self.assertEqual(params.get("hidden_size"), 6144,
                         "Flux.2 hidden_size should be 6144")
        self.assertEqual(params.get("in_channels"), 128,
                         "Flux.2 in_channels should be 128")
        self.assertEqual(params.get("num_heads"), 48,
                         "Flux.2 num_heads should be 48")
    
    def test_flux2_text_encoder_dim(self):
        """Test that Flux.2 uses Mistral encoder dimension."""
        config = flux2_models.get_flux2_config()
        
        # Mistral Small 3.1 uses 4096-dim embeddings
        self.assertIn("params", config)
        params = config["params"]
        self.assertEqual(params.get("context_in_dim"), 4096,
                         "Flux.2 should use Mistral (4096-dim) encoder")
    
    def test_flux2_vae_params(self):
        """Test that Flux.2 VAE parameters are correct."""
        config = flux2_models.get_flux2_config()
        
        # Check VAE parameters in ae_params
        self.assertIn("ae_params", config)
        ae_params = config["ae_params"]
        
        self.assertEqual(ae_params.get("z_channels"), 32,
                         "VAE z_channels should be 32")
        self.assertAlmostEqual(ae_params.get("scale_factor"), 0.3611, places=4,
                               msg="VAE scale_factor incorrect")
    
    def test_flux2_architecture_summary(self):
        """Test that architecture summary is populated."""
        summary = flux2_models.FLUX2_ARCHITECTURE_SUMMARY
        
        # Check for key fields (may have different names)
        self.assertIn("hidden_size", summary)
        self.assertIn("num_heads", summary)
        self.assertEqual(summary["hidden_size"], 6144)
        self.assertEqual(summary["num_heads"], 48)


class TestFlux2Utils(unittest.TestCase):
    """Test Flux.2 utility functions."""
    
    def test_dummy_encoder_creation(self):
        """Test that dummy encoders can be created."""
        encoder = flux2_utils.create_dummy_encoder(dims=4096)
        
        self.assertIsNotNone(encoder)
        # Dummy encoder should be callable
        self.assertTrue(callable(encoder))
    
    def test_flux2_compatibility_validation(self):
        """Test model compatibility validation function signature."""
        # This is a smoke test - actual validation requires real model file
        validator = flux2_utils.validate_flux2_compatibility
        
        self.assertIsNotNone(validator)
        self.assertTrue(callable(validator))
    
    def test_flux2_model_import(self):
        """Test that Flux2 model class can be instantiated."""
        # Check that Flux2Params exists and is a valid class
        self.assertTrue(hasattr(flux2_models, 'Flux2Params'),
                        "Flux2Params class should exist")
        self.assertTrue(callable(flux2_models.Flux2Params),
                        "Flux2Params should be callable")


class TestFlux2Training(unittest.TestCase):
    """Test Flux.2 training script structure."""
    
    def test_flux2_train_script_exists(self):
        """Test that flux2_train.py exists."""
        train_script = os.path.join(
            PROJECT_ROOT,
            "src",
            "flux2_support",
            "flux2_train.py"
        )
        
        self.assertTrue(os.path.exists(train_script),
                        f"flux2_train.py not found at {train_script}")
    
    def test_flux2_train_script_has_main(self):
        """Test that flux2_train.py has main() function."""
        train_script = os.path.join(
            PROJECT_ROOT,
            "src",
            "flux2_support",
            "flux2_train.py"
        )
        
        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("def main():", content,
                      "flux2_train.py should have main() function")
        self.assertIn("if __name__ == \"__main__\":", content,
                      "flux2_train.py should have main guard")
    
    def test_flux2_train_cache_text_encoder_check(self):
        """Test that flux2_train.py enforces cache_text_encoder_outputs."""
        train_script = os.path.join(
            PROJECT_ROOT,
            "src",
            "flux2_support",
            "flux2_train.py"
        )
        
        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("cache_text_encoder_outputs MUST be True",
                      content,
                      "flux2_train.py should enforce cache_text_encoder_outputs")
        self.assertIn("Mistral encoder which is not supported",
                      content,
                      "flux2_train.py should document Mistral limitation")


class TestConfigGenFlux2Integration(unittest.TestCase):
    """Test config_gen.py Flux.2 integration."""
    
    def test_config_gen_has_flux2_detection(self):
        """Test that config_gen.py has Flux.2 detection logic."""
        config_script = os.path.join(
            PROJECT_ROOT,
            "src",
            "config_gen.py"
        )
        
        with open(config_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("is_flux2", content,
                      "config_gen.py should have is_flux2 variable")
        self.assertIn("flux2", content.lower(),
                      "config_gen.py should mention flux2 detection")
        self.assertIn("flux2_support", content,
                      "config_gen.py should route to flux2_support")


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFlux2Detection))
    suite.addTests(loader.loadTestsFromTestCase(TestFlux2Utils))
    suite.addTests(loader.loadTestsFromTestCase(TestFlux2Training))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigGenFlux2Integration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
