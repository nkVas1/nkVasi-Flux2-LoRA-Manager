"""
Unit tests for configuration validation
Tests ConfigValidator against various config structures and edge cases
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_validator import ConfigValidator


def test_valid_config():
    """Test that valid configuration passes validation."""
    config = {
        "general": {
            "shuffle_caption": True,
            "keep_tokens": 1,
            "enable_bucket": True,
        },
        "datasets": [
            {
                "resolution": 512,
                "batch_size": 1,
                "bucket_reso_steps": 64,
                "subsets": [
                    {
                        "image_dir": "/path/to/images",
                        "num_repeats": 10,
                        "caption_extension": ".txt",
                    }
                ]
            }
        ]
    }
    
    is_valid, errors = ConfigValidator.validate_config(config)
    assert is_valid, f"Valid config failed validation: {errors}"
    print("✓ Valid config test passed")


def test_deprecated_enable_bucket_reso_steps():
    """Test that deprecated enable_bucket_reso_steps is detected."""
    config = {
        "general": {
            "enable_bucket": True,
        },
        "datasets": [
            {
                "resolution": 512,
                "enable_bucket_reso_steps": True,  # DEPRECATED
                "subsets": [
                    {
                        "image_dir": "/path",
                        "num_repeats": 10,
                    }
                ]
            }
        ]
    }
    
    is_valid, errors = ConfigValidator.validate_config(config)
    assert not is_valid, "Deprecated parameter not detected"
    assert any("enable_bucket_reso_steps" in err for err in errors), \
        f"Expected enable_bucket_reso_steps error, got: {errors}"
    print("✓ Deprecated enable_bucket_reso_steps test passed")


def test_missing_subsets():
    """Test that missing subsets is detected."""
    config = {
        "general": {"enable_bucket": True},
        "datasets": [
            {
                "resolution": 512,
                # Missing 'subsets'
            }
        ]
    }
    
    is_valid, errors = ConfigValidator.validate_config(config)
    assert not is_valid, "Missing subsets not detected"
    assert any("subsets" in err.lower() for err in errors), \
        f"Expected subsets error, got: {errors}"
    print("✓ Missing subsets test passed")


def test_missing_image_dir():
    """Test that missing image_dir in subset is detected."""
    config = {
        "general": {"enable_bucket": True},
        "datasets": [
            {
                "resolution": 512,
                "subsets": [
                    {
                        # Missing image_dir
                        "num_repeats": 10,
                    }
                ]
            }
        ]
    }
    
    is_valid, errors = ConfigValidator.validate_config(config)
    assert not is_valid, "Missing image_dir not detected"
    assert any("image_dir" in err.lower() for err in errors), \
        f"Expected image_dir error, got: {errors}"
    print("✓ Missing image_dir test passed")


def test_auto_fix_deprecated_enable_bucket():
    """Test that auto-fix repairs enable_bucket_reso_steps."""
    config = {
        "general": {
            "shuffle_caption": True,
        },
        "datasets": [
            {
                "resolution": 512,
                "enable_bucket_reso_steps": True,  # Should be converted
                "subsets": [
                    {
                        "image_dir": "/path",
                        "num_repeats": 10,
                    }
                ]
            }
        ]
    }
    
    fixed, changes = ConfigValidator.auto_fix_config(config)
    
    # Check that fixes were applied
    assert "enable_bucket_reso_steps" not in fixed["datasets"][0], \
        "enable_bucket_reso_steps not removed"
    assert len(changes) > 0, "No changes recorded"
    
    # Verify fixed config is valid
    is_valid, errors = ConfigValidator.validate_config(fixed)
    assert is_valid, f"Fixed config still invalid: {errors}"
    
    print("✓ Auto-fix enable_bucket_reso_steps test passed")
    print(f"  Changes made: {changes}")


def test_auto_fix_deprecated_seed():
    """Test that auto-fix removes deprecated seed parameter."""
    config = {
        "general": {
            "seed": 42,  # Deprecated
            "enable_bucket": True,
        },
        "datasets": [
            {
                "resolution": 512,
                "subsets": [
                    {
                        "image_dir": "/path",
                        "num_repeats": 10,
                    }
                ]
            }
        ]
    }
    
    fixed, changes = ConfigValidator.auto_fix_config(config)
    
    assert "seed" not in fixed.get("general", {}), "seed not removed from general"
    assert len(changes) > 0, "No changes recorded"
    
    # Verify fixed config is valid
    is_valid, errors = ConfigValidator.validate_config(fixed)
    assert is_valid, f"Fixed config still invalid: {errors}"
    
    print("✓ Auto-fix deprecated seed test passed")
    print(f"  Changes made: {changes}")


def test_multiple_datasets():
    """Test validation with multiple datasets."""
    config = {
        "general": {
            "enable_bucket": True,
        },
        "datasets": [
            {
                "resolution": 512,
                "subsets": [
                    {"image_dir": "/dataset1", "num_repeats": 10}
                ]
            },
            {
                "resolution": 768,
                "subsets": [
                    {"image_dir": "/dataset2", "num_repeats": 5}
                ]
            }
        ]
    }
    
    is_valid, errors = ConfigValidator.validate_config(config)
    assert is_valid, f"Multi-dataset config failed: {errors}"
    print("✓ Multiple datasets test passed")


def test_multiple_subsets():
    """Test validation with multiple subsets in one dataset."""
    config = {
        "general": {
            "enable_bucket": True,
        },
        "datasets": [
            {
                "resolution": 512,
                "subsets": [
                    {"image_dir": "/subset1", "num_repeats": 10},
                    {"image_dir": "/subset2", "num_repeats": 5},
                    {"image_dir": "/subset3", "num_repeats": 15},
                ]
            }
        ]
    }
    
    is_valid, errors = ConfigValidator.validate_config(config)
    assert is_valid, f"Multi-subset config failed: {errors}"
    print("✓ Multiple subsets test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("Running configuration validation tests...")
    print("=" * 70)
    
    try:
        test_valid_config()
        test_deprecated_enable_bucket_reso_steps()
        test_missing_subsets()
        test_missing_image_dir()
        test_auto_fix_deprecated_enable_bucket()
        test_auto_fix_deprecated_seed()
        test_multiple_datasets()
        test_multiple_subsets()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
