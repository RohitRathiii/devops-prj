#!/usr/bin/env python3
"""
Generate real metrics for midsem presentation
"""

import torch
import time
import json
from fed_drift.models import BERTClassifier, get_device
from fed_drift.data import FederatedDataLoader

def get_model_metrics():
    """Get actual model metrics"""
    print("🔄 Getting model metrics...")
    
    device = get_device()
    model = BERTClassifier(num_classes=4)
    
    # Calculate actual metrics
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    return {
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "device": str(device),
        "model_name": "prajjwal1/bert-tiny"
    }

def get_data_metrics():
    """Get actual data pipeline metrics"""
    print("🔄 Getting data metrics...")
    
    data_loader = FederatedDataLoader(num_clients=2)
    
    # Load dataset
    dataset = data_loader.load_ag_news()
    
    # Create federated splits
    client_datasets, test_dataset = data_loader.create_federated_splits()
    
    # Get statistics
    stats = data_loader.get_dataset_statistics(client_datasets)
    
    return {
        "total_train_samples": len(dataset['train']) if 'train' in dataset else 0,
        "total_test_samples": len(dataset['test']) if 'test' in dataset else 0,
        "num_clients": len(client_datasets),
        "samples_per_client": {str(k): v for k, v in stats['client_sizes'].items()},
        "label_distributions": {str(k): v for k, v in stats['label_distributions'].items()},
        "total_federated_samples": stats['total_samples']
    }

def test_training_step():
    """Test actual training step to get training metrics"""
    print("🔄 Testing training step...")
    
    try:
        device = get_device()
        model = BERTClassifier(num_classes=4)
        model.to(device)
        
        # Create dummy batch
        batch_size = 16
        seq_length = 128
        
        dummy_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
            'attention_mask': torch.ones(batch_size, seq_length).to(device),
            'labels': torch.randint(0, 4, (batch_size,)).to(device)
        }
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Time training step
        start_time = time.time()
        
        # Forward pass
        model.train()
        optimizer.zero_grad()
        
        logits = model(dummy_batch['input_ids'], dummy_batch['attention_mask'])
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, dummy_batch['labels'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == dummy_batch['labels']).float().mean()
        
        training_time = time.time() - start_time
        
        return {
            "training_step_successful": True,
            "initial_loss": round(loss.item(), 4),
            "initial_accuracy": round(accuracy.item(), 4),
            "training_time_seconds": round(training_time, 4),
            "batch_size": batch_size,
            "sequence_length": seq_length
        }
        
    except Exception as e:
        return {
            "training_step_successful": False,
            "error": str(e)
        }

def test_drift_detection():
    """Test drift detection components"""
    print("🔄 Testing drift detection...")
    
    try:
        from fed_drift.drift_detection import ADWINDriftDetector, MultiLevelDriftDetector
        
        # Test ADWIN
        adwin = ADWINDriftDetector(delta=0.002)
        adwin.update_performance(0.85)
        
        # Test multi-level detector
        config = {
            'adwin_delta': 0.002,
            'mmd_p_val': 0.05,
            'evidently_threshold': 0.25
        }
        
        detector = MultiLevelDriftDetector(config)
        
        return {
            "adwin_initialized": True,
            "multilevel_detector_initialized": True,
            "adwin_delta": 0.002,
            "mmd_p_value_threshold": 0.05,
            "evidently_threshold": 0.25,
            "detection_methods": ["ADWIN", "MMD", "Evidently"]
        }
        
    except Exception as e:
        return {
            "drift_detection_working": False,
            "error": str(e)
        }

def main():
    """Generate complete metrics report"""
    print("=" * 60)
    print("🎯 GENERATING REAL METRICS FOR MIDSEM PRESENTATION")
    print("=" * 60)
    
    metrics = {}
    
    # Get model metrics
    try:
        metrics["model_metrics"] = get_model_metrics()
        print("✅ Model metrics collected")
    except Exception as e:
        metrics["model_metrics"] = {"error": str(e)}
        print(f"❌ Model metrics failed: {e}")
    
    # Get data metrics
    try:
        metrics["data_metrics"] = get_data_metrics()
        print("✅ Data metrics collected")
    except Exception as e:
        metrics["data_metrics"] = {"error": str(e)}
        print(f"❌ Data metrics failed: {e}")
    
    # Test training
    try:
        metrics["training_metrics"] = test_training_step()
        print("✅ Training metrics collected")
    except Exception as e:
        metrics["training_metrics"] = {"error": str(e)}
        print(f"❌ Training metrics failed: {e}")
    
    # Test drift detection
    try:
        metrics["drift_detection_metrics"] = test_drift_detection()
        print("✅ Drift detection metrics collected")
    except Exception as e:
        metrics["drift_detection_metrics"] = {"error": str(e)}
        print(f"❌ Drift detection metrics failed: {e}")
    
    # Add system info
    metrics["system_info"] = {
        "pytorch_version": torch.__version__,
        "device_available": {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available()
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save metrics
    with open("real_metrics_midsem.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 REAL METRICS SUMMARY")
    print("=" * 60)
    
    # Print summary
    if "model_metrics" in metrics and "error" not in metrics["model_metrics"]:
        model = metrics["model_metrics"]
        print(f"📦 Model: {model['model_name']}")
        print(f"   - Parameters: {model['total_parameters']:,}")
        print(f"   - Size: {model['model_size_mb']} MB")
        print(f"   - Device: {model['device']}")
    
    if "data_metrics" in metrics and "error" not in metrics["data_metrics"]:
        data = metrics["data_metrics"]
        print(f"📊 Data: {data['total_train_samples']:,} train, {data['total_test_samples']:,} test")
        print(f"   - Clients: {data['num_clients']}")
        print(f"   - Federated samples: {data['total_federated_samples']:,}")
    
    if "training_metrics" in metrics and metrics["training_metrics"].get("training_step_successful"):
        train = metrics["training_metrics"]
        print(f"🏋️ Training: Loss={train['initial_loss']}, Acc={train['initial_accuracy']}")
        print(f"   - Time per step: {train['training_time_seconds']}s")
        print(f"   - Batch size: {train['batch_size']}")
    
    if "drift_detection_metrics" in metrics and "error" not in metrics["drift_detection_metrics"]:
        drift = metrics["drift_detection_metrics"]
        methods = ", ".join(drift["detection_methods"])
        print(f"🔍 Drift Detection: {methods}")
        print(f"   - ADWIN delta: {drift['adwin_delta']}")
        print(f"   - MMD p-value: {drift['mmd_p_value_threshold']}")
    
    print(f"\n💾 Metrics saved to: real_metrics_midsem.json")
    print("🎯 Use these metrics in your presentation!")

if __name__ == "__main__":
    main()