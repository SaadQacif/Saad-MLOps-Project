import pickle
import sys
import os
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))

try:
    from scripts.extract_potato_features import extract_features
    from src.complex_feature_extraction import create_feature_extractor
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import feature extraction: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False

def check_model_features(model_path):
    """Check the number of features expected by a model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if hasattr(model, 'n_features_in_'):
            logger.info(f"Model: {os.path.basename(model_path)}")
            logger.info(f"Expected features: {model.n_features_in_}")
            return model.n_features_in_
        else:
            logger.warning(f"Model {os.path.basename(model_path)} doesn't have n_features_in_ attribute")
            return None
    except Exception as e:
        logger.error(f"Error loading {model_path}: {e}")
        return None

def check_scaler_features(scaler_path):
    """Check the number of features expected by a scaler"""
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        if hasattr(scaler, 'n_features_in_'):
            logger.info(f"Scaler: {os.path.basename(scaler_path)}")
            logger.info(f"Expected features: {scaler.n_features_in_}")
            return scaler.n_features_in_
        else:
            logger.warning(f"Scaler {os.path.basename(scaler_path)} doesn't have n_features_in_ attribute")
            return None
    except Exception as e:
        logger.error(f"Error loading {scaler_path}: {e}")
        return None

def validate_feature_extraction():
    """Validate that feature extraction produces the correct dimensions"""
    if not FEATURE_EXTRACTION_AVAILABLE:
        logger.error("Feature extraction modules not available")
        return False
        
    results = {}
    
    # Test simple feature extraction (should produce 30 features)
    try:
        # Create a test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        simple_features = extract_features(test_image)
        
        if simple_features is not None:
            logger.info(f"Simple feature extractor: {len(simple_features)} features")
            results["simple_features"] = len(simple_features)
        else:
            logger.error("Simple feature extraction failed")
            results["simple_features"] = None
    except Exception as e:
        logger.error(f"Simple feature extraction error: {e}")
        results["simple_features"] = None
    
    # Test complex feature extraction with 121 features
    try:
        complex_extractor = create_feature_extractor(feature_count=121)
        complex_features = complex_extractor.extract_all_features(test_image, max_features=121)
        
        if complex_features is not None:
            logger.info(f"Complex feature extractor (121): {len(complex_features)} features")
            results["complex_features_121"] = len(complex_features)
        else:
            logger.error("Complex feature extraction failed")
            results["complex_features_121"] = None
    except Exception as e:
        logger.error(f"Complex feature extraction error: {e}")
        results["complex_features_121"] = None
        
    # Test complex feature extraction with 30 features
    try:
        complex_extractor = create_feature_extractor(feature_count=30)
        complex_features = complex_extractor.extract_all_features(test_image, max_features=30)
        
        if complex_features is not None:
            logger.info(f"Complex feature extractor (30): {len(complex_features)} features")
            results["complex_features_30"] = len(complex_features)
        else:
            logger.error("Complex feature extraction failed")
            results["complex_features_30"] = None
    except Exception as e:
        logger.error(f"Complex feature extraction error: {e}")
        results["complex_features_30"] = None
    
    return results

def check_model_and_scaler_compatibility():
    """Check compatibility between models and scalers"""
    models_dir = os.path.join(os.getcwd(), "models", "bin")
    model_info = {}
    scaler_info = {}
    
    # Check models
    for model_file in os.listdir(models_dir):
        if model_file.endswith("_model.pkl"):
            model_path = os.path.join(models_dir, model_file)
            model_name = os.path.basename(model_file).replace("_model.pkl", "")
            expected_features = check_model_features(model_path)
            model_info[model_name] = expected_features
            logger.info("-" * 30)
    
    # Check scalers
    for scaler_file in ["scaler.pkl", "scaler_multi.pkl"]:
        scaler_path = os.path.join(models_dir, scaler_file)
        if os.path.exists(scaler_path):
            expected_features = check_scaler_features(scaler_path)
            scaler_info[os.path.basename(scaler_file)] = expected_features
            logger.info("-" * 30)
    
    # Validate feature extraction
    feature_extraction_results = validate_feature_extraction()
    logger.info("-" * 30)
    
    # Check compatibility
    compatibility_issues = []
    
    # Check if each model has a compatible scaler
    for model_name, feature_count in model_info.items():
        if feature_count == 30 and scaler_info.get("scaler.pkl") == 30:
            logger.info(f"✅ Model {model_name} (30 features) is compatible with scaler.pkl")
        elif feature_count == 121 and scaler_info.get("scaler_multi.pkl") == 121:
            logger.info(f"✅ Model {model_name} (121 features) is compatible with scaler_multi.pkl")
        elif feature_count is None:
            logger.warning(f"⚠️ Cannot determine compatibility for {model_name} (unknown feature count)")
            compatibility_issues.append(f"Unknown feature count for model {model_name}")
        else:
            logger.error(f"❌ Model {model_name} ({feature_count} features) has no compatible scaler!")
            compatibility_issues.append(f"No compatible scaler for {model_name} ({feature_count} features)")
    
    # Check feature extraction compatibility
    if feature_extraction_results:
        if feature_extraction_results.get("simple_features") == 30:
            logger.info("✅ Simple feature extractor produces 30 features as expected")
        else:
            logger.error(f"❌ Simple feature extractor produces {feature_extraction_results.get('simple_features')} features, expected 30")
            compatibility_issues.append(f"Simple feature extraction: expected 30, got {feature_extraction_results.get('simple_features')}")
        
        if feature_extraction_results.get("complex_features_121") == 121:
            logger.info("✅ Complex feature extractor produces 121 features as expected")
        else:
            logger.error(f"❌ Complex feature extractor produces {feature_extraction_results.get('complex_features_121')} features, expected 121")
            compatibility_issues.append(f"Complex feature extraction: expected 121, got {feature_extraction_results.get('complex_features_121')}")
    
    # Final report
    if compatibility_issues:
        logger.warning("⚠️ Found compatibility issues:")
        for issue in compatibility_issues:
            logger.warning(f" - {issue}")
    else:
        logger.info("✅ All models and scalers are compatible!")
    
    return {
        "models": model_info,
        "scalers": scaler_info,
        "feature_extraction": feature_extraction_results,
        "issues": compatibility_issues
    }

if __name__ == "__main__":
    logger.info("Performing comprehensive model and feature compatibility check...")
    results = check_model_and_scaler_compatibility()
    
    # Count issues
    issue_count = len(results["issues"])
    if issue_count == 0:
        logger.info("✅ All checks passed successfully!")
    else:
        logger.warning(f"⚠️ Found {issue_count} compatibility issues. Please fix before deployment.")
