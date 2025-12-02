# Complete Federated Learning Drift Detection System - All Missing Components
# This file contains all the remaining components to complete the notebook

# =====================================
# Advanced Drift Injection System
# =====================================

class AdvancedDriftInjector:
    """Comprehensive drift injection system matching your original DriftInjector class."""

    def __init__(self, config):
        self.config = config
        self.drift_types = config['drift']['drift_types']
        self.affected_clients = set(config['drift']['affected_clients'])

        # Initialize augmenters if available
        self.vocab_augmenter = None
        if CAPABILITIES['nlpaug']:
            try:
                self.vocab_augmenter = naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=config['drift']['vocab_shift_rate']
                )
                print("✅ NLP Augmenter initialized for vocabulary shift")
            except Exception as e:
                print(f"⚠️ NLP Augmenter failed: {e}, using fallback")

    def inject_drift(self, client_id, dataset, round_num):
        """Inject various types of drift into client dataset."""
        if (round_num < self.config['drift']['injection_round'] or
            client_id not in self.affected_clients):
            return dataset

        print(f"💥 Injecting drift into Client {client_id} at Round {round_num}")

        # Apply different drift types
        modified_dataset = dataset

        for drift_type in self.drift_types:
            if drift_type == 'label_noise':
                modified_dataset = self._inject_label_noise(modified_dataset, client_id)
            elif drift_type == 'vocab_shift':
                modified_dataset = self._inject_vocabulary_shift(modified_dataset, client_id)
            elif drift_type == 'distribution_shift':
                modified_dataset = self._inject_distribution_shift(modified_dataset, client_id)

        return modified_dataset

    def _inject_label_noise(self, dataset, client_id):
        """Inject label noise drift."""
        noise_rate = self.config['drift']['label_noise_rate']
        num_samples = len(dataset)
        num_noisy = int(num_samples * noise_rate)

        print(f"🔀 Injecting label noise: {num_noisy}/{num_samples} samples for Client {client_id}")

        # Create new dataset with noisy labels
        texts = []
        labels = []

        for i in range(num_samples):
            sample = dataset[i]
            text = sample['input_ids']  # Keep original encoding
            attention_mask = sample['attention_mask']
            original_label = sample['labels'].item()

            # Add noise to some labels
            if i < num_noisy:
                # Randomly change to different class
                new_label = random.choice([l for l in range(4) if l != original_label])
                labels.append(new_label)
            else:
                labels.append(original_label)

            texts.append((text, attention_mask))

        # Create new dataset with modified labels
        return self._create_modified_dataset(texts, labels, dataset.tokenizer, dataset.max_length)

    def _inject_vocabulary_shift(self, dataset, client_id):
        """Inject vocabulary shift drift."""
        print(f"📝 Injecting vocabulary shift for Client {client_id}")

        if self.vocab_augmenter is None:
            return self._inject_vocabulary_shift_fallback(dataset, client_id)

        # Extract original texts and labels
        original_texts = []
        labels = []

        # This is a simplified approach - in practice, you'd need to decode tokenized text
        # For demo purposes, we'll apply fallback method
        return self._inject_vocabulary_shift_fallback(dataset, client_id)

    def _inject_vocabulary_shift_fallback(self, dataset, client_id):
        """Fallback vocabulary shift without nlpaug."""
        shift_rate = self.config['drift']['vocab_shift_rate']

        # Simple word substitution dictionary
        word_substitutions = {
            'good': 'excellent', 'bad': 'terrible', 'big': 'large', 'small': 'tiny',
            'fast': 'quick', 'slow': 'sluggish', 'new': 'fresh', 'old': 'ancient',
            'high': 'elevated', 'low': 'minimal', 'strong': 'powerful', 'weak': 'feeble'
        }

        # For demonstration, we'll modify some token IDs randomly
        modified_data = []
        for i, sample in enumerate(dataset):
            input_ids = sample['input_ids'].clone()
            attention_mask = sample['attention_mask']
            labels = sample['labels']

            # Randomly modify some tokens (simulate vocabulary shift)
            if random.random() < shift_rate:
                # Find non-padding tokens
                non_pad_indices = (input_ids != 0).nonzero(as_tuple=True)[0]
                if len(non_pad_indices) > 2:  # Skip CLS and SEP
                    # Randomly select a token to modify
                    modify_idx = random.choice(non_pad_indices[1:-1])  # Skip CLS and SEP
                    # Add small random offset to simulate vocabulary shift
                    input_ids[modify_idx] = max(1, input_ids[modify_idx] + random.randint(-5, 5))

            modified_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })

        print(f"📝 Applied fallback vocabulary shift to ~{int(len(dataset) * shift_rate)} samples")
        return modified_data

    def _inject_distribution_shift(self, dataset, client_id):
        """Inject distribution shift by changing class balance."""
        severity = self.config['drift']['distribution_shift_severity']

        print(f"📊 Injecting distribution shift (severity: {severity}) for Client {client_id}")

        # Analyze current class distribution
        class_counts = {}
        class_samples = {}

        for i, sample in enumerate(dataset):
            label = sample['labels'].item()
            if label not in class_counts:
                class_counts[label] = 0
                class_samples[label] = []
            class_counts[label] += 1
            class_samples[label].append(i)

        print(f"   Original distribution: {class_counts}")

        # Create imbalanced distribution
        # Reduce samples from some classes, increase others
        target_samples = []

        for class_id in range(4):
            if class_id in class_samples:
                current_samples = class_samples[class_id]

                if class_id % 2 == 0:  # Reduce even classes
                    keep_fraction = 1.0 - severity
                    num_keep = max(1, int(len(current_samples) * keep_fraction))
                    selected_samples = random.sample(current_samples, num_keep)
                else:  # Keep odd classes as is
                    selected_samples = current_samples

                target_samples.extend(selected_samples)

        # Create new dataset with modified distribution
        modified_dataset = [dataset[i] for i in target_samples]

        # Calculate new distribution
        new_class_counts = {}
        for sample in modified_dataset:
            label = sample['labels'].item()
            new_class_counts[label] = new_class_counts.get(label, 0) + 1

        print(f"   New distribution: {new_class_counts}")
        return modified_dataset

    def _create_modified_dataset(self, texts, labels, tokenizer, max_length):
        """Create new dataset from modified texts and labels."""
        class ModifiedDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text_data, attention_mask = self.texts[idx]
                label = self.labels[idx]

                return {
                    'input_ids': text_data,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        return ModifiedDataset(texts, labels)


# =====================================
# Sophisticated Federated Client
# =====================================

class DriftDetectionClient:
    """Advanced federated client with integrated drift detection."""

    def __init__(self, client_id, dataset, model, tokenizer, config):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize drift detection
        self.drift_detector = MultiLevelDriftDetector(config)
        self.drift_injector = AdvancedDriftInjector(config)

        # Performance tracking
        self.round_metrics = []
        self.embeddings_history = []
        self.predictions_history = []

        # Training setup
        self.optimizer = None
        self.loss_fn = nn.CrossEntropyLoss()

        print(f"👤 Client {client_id} initialized with {len(dataset)} samples")

    def set_parameters(self, parameters_dict):
        """Set model parameters from server."""
        self.model.set_parameters_dict(parameters_dict)

    def get_parameters(self):
        """Get current model parameters."""
        return self.model.get_parameters_dict()

    def fit(self, parameters_dict, round_num):
        """Train the model and return updated parameters with drift info."""
        print(f"🏋️ Client {self.client_id} training for Round {round_num}")

        # Set parameters from server
        self.set_parameters(parameters_dict)

        # Inject drift if applicable
        current_dataset = self.drift_injector.inject_drift(
            self.client_id, self.dataset, round_num
        )

        # Train the model
        train_loss, train_accuracy = self._train_epoch(current_dataset)

        # Extract embeddings for drift detection
        embeddings = self._extract_embeddings(current_dataset)

        # Get predictions for drift analysis
        predictions = self._get_predictions(current_dataset)

        # Update drift detector
        drift_result = self.drift_detector.update(
            accuracy=train_accuracy,
            embeddings=embeddings,
            predictions=predictions
        )

        # Store metrics
        metrics = {
            'round': round_num,
            'loss': train_loss,
            'accuracy': train_accuracy,
            'drift_detected': drift_result['drift_detected'],
            'drift_signals': drift_result['signals'],
            'num_samples': len(current_dataset),
            'client_id': self.client_id  # Add client ID for server analysis
        }
        self.round_metrics.append(metrics)

        print(f"   📊 Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        if drift_result['drift_detected']:
            print(f"   🚨 DRIFT DETECTED! Signals: {drift_result['signals']}")

        return self.get_parameters(), len(current_dataset), metrics

    def evaluate(self, parameters_dict, test_dataset):
        """Evaluate the model and return metrics."""
        # Set parameters
        self.set_parameters(parameters_dict)

        # Evaluate
        test_loss, test_accuracy = self._evaluate_dataset(test_dataset)

        return len(test_dataset), test_loss, test_accuracy

    def _train_epoch(self, dataset):
        """Train for one epoch."""
        self.model.train()

        # Setup optimizer
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['model']['learning_rate']
            )

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=True
        )

        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate_dataset(self, dataset):
        """Evaluate model on dataset."""
        self.model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False
        )

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']

                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _extract_embeddings(self, dataset):
        """Extract embeddings for drift detection."""
        self.model.eval()

        # Sample subset for efficiency
        sample_size = min(100, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)

        embeddings = []

        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

                embedding = self.model.get_embeddings(input_ids, attention_mask)
                embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings) if embeddings else np.array([])

    def _get_predictions(self, dataset):
        """Get model predictions for drift analysis."""
        self.model.eval()

        # Sample subset for efficiency
        sample_size = min(100, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)

        predictions = []

        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

                outputs = self.model(input_ids, attention_mask)
                pred_probs = torch.softmax(outputs['logits'], dim=1)
                predictions.extend(pred_probs.cpu().numpy().flatten())

        return np.array(predictions)

    def get_drift_history(self):
        """Get complete drift detection history."""
        return {
            'detection_history': self.drift_detector.detection_history,
            'round_metrics': self.round_metrics
        }


# =====================================
# Advanced Server Strategy
# =====================================

class FedTrimmedAvg:
    """FedTrimmedAvg implementation for robust aggregation."""

    def __init__(self, beta=0.2):
        self.beta = beta  # Fraction to trim

    def aggregate(self, client_updates):
        """Aggregate client updates using trimmed mean."""
        if not client_updates:
            return None

        # Convert to tensor format for easier manipulation
        stacked_updates = {}

        # Stack all client parameters
        for param_name in client_updates[0][0].keys():
            param_stack = torch.stack([
                update[0][param_name] for update in client_updates
            ])
            stacked_updates[param_name] = param_stack

        # Apply trimmed mean to each parameter
        aggregated_params = {}

        for param_name, param_tensor in stacked_updates.items():
            # Calculate trimmed mean
            trimmed_mean = self._trimmed_mean(param_tensor, self.beta)
            aggregated_params[param_name] = trimmed_mean

        return aggregated_params

    def _trimmed_mean(self, tensor, beta):
        """Calculate trimmed mean along client dimension."""
        # tensor shape: [num_clients, ...]
        num_clients = tensor.shape[0]
        trim_count = int(num_clients * beta)

        if trim_count == 0:
            return torch.mean(tensor, dim=0)

        # Flatten for easier sorting
        original_shape = tensor.shape
        flattened = tensor.view(num_clients, -1)

        # Sort along client dimension
        sorted_tensor, _ = torch.sort(flattened, dim=0)

        # Trim top and bottom
        trim_bottom = trim_count // 2
        trim_top = trim_count - trim_bottom

        if trim_top > 0:
            trimmed_tensor = sorted_tensor[trim_bottom:-trim_top]
        else:
            trimmed_tensor = sorted_tensor[trim_bottom:]

        # Calculate mean of remaining values
        result = torch.mean(trimmed_tensor, dim=0)

        # Reshape back to original parameter shape
        return result.view(original_shape[1:])


class DriftAwareFedAvg:
    """Advanced server strategy with drift-aware aggregation."""

    def __init__(self, config):
        self.config = config

        # Aggregation strategies
        self.fed_avg = self._weighted_average
        self.fed_trimmed_avg = FedTrimmedAvg(
            beta=config['drift_detection']['trimmed_beta']
        )

        # Drift monitoring
        self.global_drift_detector = MultiLevelDriftDetector(config)
        self.client_drift_reports = {}
        self.mitigation_active = False
        self.mitigation_threshold = config['simulation']['mitigation_threshold']

        # Performance tracking
        self.global_metrics = []
        self.aggregation_history = []

        print("🏛️ Drift-aware server strategy initialized")

    def aggregate_fit(self, round_num, client_updates, test_dataset=None):
        """Aggregate client updates with drift awareness."""
        print(f"🏛️ Server aggregating {len(client_updates)} client updates for Round {round_num}")

        # Extract client parameters and metrics
        parameters_updates = [(params, num_samples) for params, num_samples, metrics in client_updates]
        client_metrics = [metrics for params, num_samples, metrics in client_updates]

        # Analyze client drift reports
        drift_ratio = self._analyze_client_drift(client_metrics, round_num)

        # Decide on aggregation strategy
        use_mitigation = self._should_use_mitigation(drift_ratio, round_num)

        if use_mitigation and not self.mitigation_active:
            print(f"🚨 ACTIVATING MITIGATION: {drift_ratio:.1%} clients report drift")
            self.mitigation_active = True
        elif not use_mitigation and self.mitigation_active:
            print(f"✅ DEACTIVATING MITIGATION: drift ratio below threshold")
            self.mitigation_active = False

        # Aggregate parameters
        if self.mitigation_active:
            # Use robust FedTrimmedAvg
            aggregated_params = self.fed_trimmed_avg.aggregate(parameters_updates)
            strategy_used = "FedTrimmedAvg"
        else:
            # Use standard FedAvg
            aggregated_params = self.fed_avg(parameters_updates)
            strategy_used = "FedAvg"

        # Evaluate global model if test dataset provided
        global_metrics = None
        if test_dataset is not None:
            global_metrics = self._evaluate_global_model(
                aggregated_params, test_dataset, round_num
            )

        # Store aggregation info
        aggregation_info = {
            'round': round_num,
            'strategy': strategy_used,
            'drift_ratio': drift_ratio,
            'mitigation_active': self.mitigation_active,
            'num_clients': len(client_updates),
            'global_metrics': global_metrics
        }
        self.aggregation_history.append(aggregation_info)

        print(f"   📊 Strategy: {strategy_used}, Drift ratio: {drift_ratio:.1%}")
        if global_metrics:
            print(f"   🎯 Global accuracy: {global_metrics['accuracy']:.4f}")

        return aggregated_params, aggregation_info

    def _analyze_client_drift(self, client_metrics, round_num):
        """Analyze drift reports from clients."""
        drift_reports = []

        for metrics in client_metrics:
            if 'drift_detected' in metrics:
                drift_reports.append(metrics['drift_detected'])

                # Store client drift info
                client_id = metrics.get('client_id', len(self.client_drift_reports))
                if client_id not in self.client_drift_reports:
                    self.client_drift_reports[client_id] = []

                self.client_drift_reports[client_id].append({
                    'round': round_num,
                    'drift_detected': metrics['drift_detected'],
                    'drift_signals': metrics.get('drift_signals', {})
                })

        # Calculate drift ratio
        if drift_reports:
            drift_ratio = sum(drift_reports) / len(drift_reports)
        else:
            drift_ratio = 0.0

        return drift_ratio

    def _should_use_mitigation(self, drift_ratio, round_num):
        """Decide whether to use mitigation strategy."""
        # Use mitigation if drift ratio exceeds threshold
        return drift_ratio > self.mitigation_threshold

    def _weighted_average(self, client_updates):
        """Standard FedAvg weighted averaging."""
        if not client_updates:
            return None

        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in client_updates)

        # Initialize aggregated parameters
        aggregated_params = None

        for params, num_samples in client_updates:
            weight = num_samples / total_samples

            if aggregated_params is None:
                # Initialize with first client's parameters
                aggregated_params = {}
                for name, param in params.items():
                    aggregated_params[name] = param * weight
            else:
                # Add weighted parameters
                for name, param in params.items():
                    aggregated_params[name] += param * weight

        return aggregated_params

    def _evaluate_global_model(self, parameters, test_dataset, round_num):
        """Evaluate global model performance."""
        # Create temporary model for evaluation
        temp_model, _ = create_model_and_tokenizer()
        temp_model.set_parameters_dict(parameters)
        temp_model.eval()

        # Evaluate on test set
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False
        )

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = temp_model(input_ids, attention_mask, labels)
                loss = outputs['loss']

                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        metrics = {
            'round': round_num,
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total
        }

        self.global_metrics.append(metrics)
        return metrics

    def get_server_metrics(self):
        """Get complete server metrics and drift analysis."""
        return {
            'aggregation_history': self.aggregation_history,
            'global_metrics': self.global_metrics,
            'client_drift_reports': self.client_drift_reports,
            'mitigation_active': self.mitigation_active
        }


# =====================================
# Complete Simulation Engine
# =====================================

class FederatedDriftSimulation:
    """Complete simulation orchestration matching your original simulation.py"""

    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.server_strategy = DriftAwareFedAvg(config)

        # Simulation state
        self.current_round = 0
        self.simulation_results = {
            'rounds': [],
            'global_metrics': [],
            'client_metrics': [],
            'drift_events': [],
            'aggregation_history': []
        }

    def setup_simulation(self):
        """Initialize clients and datasets."""
        print("🔧 Setting up federated simulation...")

        # Create datasets
        client_datasets, test_dataset, tokenizer = create_federated_datasets()
        self.test_dataset = test_dataset

        # Create clients
        for client_id, dataset in client_datasets.items():
            # Create model for each client
            model, _ = create_model_and_tokenizer()

            # Create client
            client = DriftDetectionClient(
                client_id=client_id,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                config=self.config
            )

            self.clients[client_id] = client

        print(f"✅ Simulation setup complete: {len(self.clients)} clients, {len(test_dataset)} test samples")

    def run_simulation(self):
        """Run the complete federated learning simulation."""
        print(f"🚀 Starting federated simulation for {self.config['simulation']['num_rounds']} rounds...")

        # Setup simulation
        self.setup_simulation()

        # Initialize global model parameters
        global_model, _ = create_model_and_tokenizer()
        global_params = global_model.get_parameters_dict()

        # Run simulation rounds
        for round_num in range(1, self.config['simulation']['num_rounds'] + 1):
            print(f"\n🔄 === ROUND {round_num} ===")

            round_results = self._run_round(round_num, global_params)

            # Update global parameters
            global_params = round_results['aggregated_params']

            # Store results
            self.simulation_results['rounds'].append(round_num)
            self.simulation_results['global_metrics'].append(round_results['global_metrics'])
            self.simulation_results['client_metrics'].append(round_results['client_metrics'])
            self.simulation_results['aggregation_history'].append(round_results['aggregation_info'])

            # Track drift events
            if round_results['drift_ratio'] > 0:
                self.simulation_results['drift_events'].append({
                    'round': round_num,
                    'drift_ratio': round_results['drift_ratio'],
                    'mitigation_active': round_results['aggregation_info']['mitigation_active']
                })

            # Print round summary
            self._print_round_summary(round_num, round_results)

        print(f"\n🏁 Simulation complete! Results ready for analysis.")
        return self.simulation_results

    def _run_round(self, round_num, global_params):
        """Execute one federated learning round."""
        # Client training phase
        client_updates = []
        client_metrics = []

        participating_clients = list(self.clients.keys())  # All clients participate

        for client_id in participating_clients:
            client = self.clients[client_id]

            # Client training
            params, num_samples, metrics = client.fit(global_params, round_num)
            client_updates.append((params, num_samples, metrics))
            client_metrics.append(metrics)

        # Server aggregation phase
        aggregated_params, aggregation_info = self.server_strategy.aggregate_fit(
            round_num, client_updates, self.test_dataset
        )

        # Calculate drift ratio for this round
        drift_ratio = sum(m.get('drift_detected', False) for m in client_metrics) / len(client_metrics)

        return {
            'aggregated_params': aggregated_params,
            'global_metrics': aggregation_info['global_metrics'],
            'client_metrics': client_metrics,
            'aggregation_info': aggregation_info,
            'drift_ratio': drift_ratio
        }

    def _print_round_summary(self, round_num, results):
        """Print summary of round results."""
        global_metrics = results['global_metrics']
        drift_ratio = results['drift_ratio']
        mitigation = results['aggregation_info']['mitigation_active']

        if global_metrics:
            print(f"📊 Global Accuracy: {global_metrics['accuracy']:.4f}")
            print(f"📊 Global Loss: {global_metrics['loss']:.4f}")

        print(f"🚨 Drift Ratio: {drift_ratio:.1%}")
        print(f"🛡️ Mitigation: {'ACTIVE' if mitigation else 'inactive'}")

        # Check if this is drift injection round
        if round_num == self.config['drift']['injection_round']:
            print("💥 DRIFT INJECTION ROUND!")

    def get_comprehensive_results(self):
        """Get complete simulation analysis."""
        server_metrics = self.server_strategy.get_server_metrics()

        # Collect client drift histories
        client_drift_histories = {}
        for client_id, client in self.clients.items():
            client_drift_histories[client_id] = client.get_drift_history()

        return {
            'simulation_results': self.simulation_results,
            'server_metrics': server_metrics,
            'client_drift_histories': client_drift_histories,
            'config': self.config
        }


# =====================================
# Advanced Visualization System
# =====================================

class ComprehensiveVisualizer:
    """Advanced visualization system for all metrics and analysis."""

    def __init__(self, results):
        self.results = results
        self.simulation_results = results['simulation_results']
        self.server_metrics = results['server_metrics']
        self.client_drift_histories = results['client_drift_histories']
        self.config = results['config']

    def create_comprehensive_dashboard(self):
        """Create complete dashboard with all visualizations."""
        print("📊 Creating comprehensive visualization dashboard...")

        # Create large figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Global Performance Over Time
        ax1 = plt.subplot(3, 4, 1)
        self._plot_global_performance(ax1)

        # 2. Drift Detection Timeline
        ax2 = plt.subplot(3, 4, 2)
        self._plot_drift_timeline(ax2)

        # 3. Aggregation Strategy Usage
        ax3 = plt.subplot(3, 4, 3)
        self._plot_aggregation_strategy(ax3)

        # 4. Client Drift Distribution
        ax4 = plt.subplot(3, 4, 4)
        self._plot_client_drift_distribution(ax4)

        # 5. Performance vs Drift Correlation
        ax5 = plt.subplot(3, 4, 5)
        self._plot_performance_drift_correlation(ax5)

        # 6. Recovery Analysis
        ax6 = plt.subplot(3, 4, 6)
        self._plot_recovery_analysis(ax6)

        # 7. Client Performance Comparison
        ax7 = plt.subplot(3, 4, 7)
        self._plot_client_performance_comparison(ax7)

        # 8. Drift Detection Methods Comparison
        ax8 = plt.subplot(3, 4, 8)
        self._plot_drift_methods_comparison(ax8)

        # 9. Fairness Analysis
        ax9 = plt.subplot(3, 4, 9)
        self._plot_fairness_analysis(ax9)

        # 10. System Timeline Overview
        ax10 = plt.subplot(3, 4, (10, 12))  # Span 3 columns
        self._plot_system_timeline(ax10)

        plt.tight_layout()
        plt.suptitle('🔄 Federated Learning Drift Detection & Recovery Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.show()

        # Create detailed analysis plots
        self._create_detailed_analysis_plots()

    def _plot_global_performance(self, ax):
        """Plot global model performance over time."""
        rounds = self.simulation_results['rounds']
        global_metrics = self.simulation_results['global_metrics']

        accuracies = [m['accuracy'] if m else 0 for m in global_metrics]
        losses = [m['loss'] if m else 0 for m in global_metrics]

        ax.plot(rounds, accuracies, 'b-', label='Accuracy', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(rounds, losses, 'r--', label='Loss', linewidth=2)

        # Mark drift injection
        injection_round = self.config['drift']['injection_round']
        ax.axvline(x=injection_round, color='orange', linestyle=':',
                  label='Drift Injection', linewidth=2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy', color='blue')
        ax2.set_ylabel('Loss', color='red')
        ax.set_title('Global Model Performance')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_drift_timeline(self, ax):
        """Plot drift detection timeline."""
        rounds = self.simulation_results['rounds']

        # Collect drift ratios
        drift_ratios = []
        for round_data in self.simulation_results['aggregation_history']:
            if round_data:
                drift_ratios.append(round_data.get('drift_ratio', 0))
            else:
                drift_ratios.append(0)

        ax.plot(rounds, drift_ratios, 'r-', linewidth=2, label='Drift Ratio')\n        \n        # Mark mitigation periods\n        mitigation_rounds = []\n        for i, round_data in enumerate(self.simulation_results['aggregation_history']):\n            if round_data and round_data.get('mitigation_active', False):\n                mitigation_rounds.append(rounds[i])\n        \n        if mitigation_rounds:\n            ax.scatter(mitigation_rounds, [drift_ratios[rounds.index(r)] for r in mitigation_rounds],\n                      color='red', s=100, marker='s', label='Mitigation Active', zorder=5)\n        \n        # Mark drift injection\n        injection_round = self.config['drift']['injection_round']\n        ax.axvline(x=injection_round, color='orange', linestyle=':', \n                  label='Drift Injection', linewidth=2)\n        \n        ax.set_xlabel('Round')\n        ax.set_ylabel('Drift Ratio')\n        ax.set_title('Drift Detection Timeline')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        ax.set_ylim(0, 1)\n    \n    def _plot_aggregation_strategy(self, ax):\n        \"\"\"Plot aggregation strategy usage over time.\"\"\"\n        rounds = self.simulation_results['rounds']\n        strategies = []\n        \n        for round_data in self.simulation_results['aggregation_history']:\n            if round_data:\n                strategies.append(round_data.get('strategy', 'FedAvg'))\n            else:\n                strategies.append('FedAvg')\n        \n        # Create binary representation\n        strategy_binary = [1 if s == 'FedTrimmedAvg' else 0 for s in strategies]\n        \n        ax.fill_between(rounds, strategy_binary, alpha=0.7, \n                       label='FedTrimmedAvg', color='red')\n        ax.fill_between(rounds, [1-x for x in strategy_binary], alpha=0.7,\n                       label='FedAvg', color='blue')\n        \n        ax.set_xlabel('Round')\n        ax.set_ylabel('Strategy')\n        ax.set_title('Aggregation Strategy Usage')\n        ax.legend()\n        ax.set_ylim(0, 1)\n        ax.set_yticks([0, 1])\n        ax.set_yticklabels(['FedAvg', 'FedTrimmedAvg'])\n    \n    def _plot_client_drift_distribution(self, ax):\n        \"\"\"Plot distribution of drift across clients.\"\"\"\n        client_drift_counts = {}\n        \n        for client_id, history in self.client_drift_histories.items():\n            drift_count = sum(history['detection_history']['combined'])\n            client_drift_counts[client_id] = drift_count\n        \n        clients = list(client_drift_counts.keys())\n        counts = list(client_drift_counts.values())\n        \n        bars = ax.bar(clients, counts, alpha=0.7)\n        \n        # Color affected clients differently\n        affected_clients = self.config['drift']['affected_clients']\n        for i, client_id in enumerate(clients):\n            if client_id in affected_clients:\n                bars[i].set_color('red')\n                bars[i].set_label('Affected by Drift' if i == 0 else \"\")\n            else:\n                bars[i].set_color('blue')\n                bars[i].set_label('Not Affected' if i == len(affected_clients) else \"\")\n        \n        ax.set_xlabel('Client ID')\n        ax.set_ylabel('Drift Detections')\n        ax.set_title('Client Drift Distribution')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n    \n    def _plot_performance_drift_correlation(self, ax):\n        \"\"\"Plot correlation between performance and drift.\"\"\"\n        rounds = self.simulation_results['rounds']\n        global_metrics = self.simulation_results['global_metrics']\n        \n        accuracies = [m['accuracy'] if m else 0 for m in global_metrics]\n        \n        # Get drift ratios\n        drift_ratios = []\n        for round_data in self.simulation_results['aggregation_history']:\n            if round_data:\n                drift_ratios.append(round_data.get('drift_ratio', 0))\n            else:\n                drift_ratios.append(0)\n        \n        # Create scatter plot\n        scatter = ax.scatter(drift_ratios, accuracies, c=rounds, \n                           cmap='viridis', alpha=0.7, s=50)\n        \n        ax.set_xlabel('Drift Ratio')\n        ax.set_ylabel('Global Accuracy')\n        ax.set_title('Performance vs Drift Correlation')\n        \n        # Add colorbar for rounds\n        cbar = plt.colorbar(scatter, ax=ax)\n        cbar.set_label('Round')\n        \n        ax.grid(True, alpha=0.3)\n    \n    def _plot_recovery_analysis(self, ax):\n        \"\"\"Analyze recovery after drift injection.\"\"\"\n        injection_round = self.config['drift']['injection_round']\n        recovery_window = self.config['simulation']['recovery_window']\n        \n        rounds = self.simulation_results['rounds']\n        global_metrics = self.simulation_results['global_metrics']\n        \n        accuracies = [m['accuracy'] if m else 0 for m in global_metrics]\n        \n        # Find baseline, drift, and recovery accuracies\n        baseline_acc = np.mean(accuracies[:injection_round-1]) if injection_round > 1 else 0\n        \n        drift_start = injection_round\n        drift_end = min(injection_round + 5, len(accuracies))\n        drift_acc = np.mean(accuracies[drift_start:drift_end]) if drift_end > drift_start else 0\n        \n        recovery_start = injection_round + 5\n        recovery_end = min(recovery_start + recovery_window, len(accuracies))\n        recovery_acc = np.mean(accuracies[recovery_start:recovery_end]) if recovery_end > recovery_start else 0\n        \n        # Create bar plot\n        phases = ['Baseline', 'Drift Impact', 'Recovery']\n        values = [baseline_acc, drift_acc, recovery_acc]\n        colors = ['green', 'red', 'blue']\n        \n        bars = ax.bar(phases, values, color=colors, alpha=0.7)\n        \n        # Add value labels on bars\n        for bar, value in zip(bars, values):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n                   f'{value:.3f}', ha='center', va='bottom')\n        \n        ax.set_ylabel('Accuracy')\n        ax.set_title('Recovery Analysis')\n        ax.grid(True, alpha=0.3)\n        \n        # Calculate recovery rate\n        if baseline_acc > drift_acc:\n            recovery_rate = (recovery_acc - drift_acc) / (baseline_acc - drift_acc)\n            ax.text(0.5, 0.9, f'Recovery Rate: {recovery_rate:.1%}', \n                   transform=ax.transAxes, ha='center', \n                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))\n    \n    def _plot_client_performance_comparison(self, ax):\n        \"\"\"Compare performance across clients.\"\"\"\n        client_accuracies = {}\n        \n        for client_id, history in self.client_drift_histories.items():\n            metrics = history['round_metrics']\n            if metrics:\n                recent_acc = np.mean([m['accuracy'] for m in metrics[-5:]])  # Last 5 rounds\n                client_accuracies[client_id] = recent_acc\n        \n        clients = list(client_accuracies.keys())\n        accuracies = list(client_accuracies.values())\n        \n        bars = ax.bar(clients, accuracies, alpha=0.7)\n        \n        # Color affected clients differently\n        affected_clients = self.config['drift']['affected_clients']\n        for i, client_id in enumerate(clients):\n            if client_id in affected_clients:\n                bars[i].set_color('red')\n            else:\n                bars[i].set_color('blue')\n        \n        ax.set_xlabel('Client ID')\n        ax.set_ylabel('Average Accuracy (Last 5 Rounds)')\n        ax.set_title('Client Performance Comparison')\n        ax.grid(True, alpha=0.3)\n        \n        # Add fairness metrics\n        if accuracies:\n            fairness_gap = max(accuracies) - min(accuracies)\n            ax.text(0.5, 0.9, f'Fairness Gap: {fairness_gap:.3f}', \n                   transform=ax.transAxes, ha='center',\n                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))\n    \n    def _plot_drift_methods_comparison(self, ax):\n        \"\"\"Compare different drift detection methods.\"\"\"\n        # Aggregate drift detection counts by method\n        method_counts = {'ADWIN': 0, 'MMD': 0, 'Statistical': 0}\n        \n        for client_id, history in self.client_drift_histories.items():\n            detection_history = history['detection_history']\n            method_counts['ADWIN'] += sum(detection_history.get('adwin', []))\n            method_counts['MMD'] += sum(detection_history.get('mmd', []))\n            method_counts['Statistical'] += sum(detection_history.get('statistical', []))\n        \n        methods = list(method_counts.keys())\n        counts = list(method_counts.values())\n        \n        bars = ax.bar(methods, counts, color=['blue', 'green', 'orange'], alpha=0.7)\n        \n        # Add value labels\n        for bar, count in zip(bars, counts):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,\n                   str(count), ha='center', va='bottom')\n        \n        ax.set_ylabel('Total Detections')\n        ax.set_title('Drift Detection Methods Comparison')\n        ax.grid(True, alpha=0.3)\n    \n    def _plot_fairness_analysis(self, ax):\n        \"\"\"Analyze fairness across training rounds.\"\"\"\n        rounds = self.simulation_results['rounds']\n        fairness_gaps = []\n        \n        for round_metrics in self.simulation_results['client_metrics']:\n            if round_metrics:\n                accuracies = [m.get('accuracy', 0) for m in round_metrics]\n                if accuracies:\n                    fairness_gap = max(accuracies) - min(accuracies)\n                    fairness_gaps.append(fairness_gap)\n                else:\n                    fairness_gaps.append(0)\n            else:\n                fairness_gaps.append(0)\n        \n        ax.plot(rounds, fairness_gaps, 'purple', linewidth=2, label='Fairness Gap')\n        \n        # Mark drift injection\n        injection_round = self.config['drift']['injection_round']\n        ax.axvline(x=injection_round, color='orange', linestyle=':', \n                  label='Drift Injection', linewidth=2)\n        \n        ax.set_xlabel('Round')\n        ax.set_ylabel('Fairness Gap (Max - Min Accuracy)')\n        ax.set_title('Fairness Analysis Over Time')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n    \n    def _plot_system_timeline(self, ax):\n        \"\"\"Create comprehensive system timeline.\"\"\"\n        rounds = self.simulation_results['rounds']\n        \n        # Plot multiple metrics on same timeline\n        global_metrics = self.simulation_results['global_metrics']\n        accuracies = [m['accuracy'] if m else 0 for m in global_metrics]\n        \n        # Normalize accuracy to 0-1 range for plotting\n        norm_accuracies = [(a - min(accuracies)) / (max(accuracies) - min(accuracies)) \n                          if max(accuracies) > min(accuracies) else 0.5 for a in accuracies]\n        \n        # Get drift ratios\n        drift_ratios = []\n        mitigation_status = []\n        for round_data in self.simulation_results['aggregation_history']:\n            if round_data:\n                drift_ratios.append(round_data.get('drift_ratio', 0))\n                mitigation_status.append(1 if round_data.get('mitigation_active', False) else 0)\n            else:\n                drift_ratios.append(0)\n                mitigation_status.append(0)\n        \n        # Plot all metrics\n        ax.plot(rounds, norm_accuracies, 'b-', linewidth=3, label='Normalized Accuracy', alpha=0.8)\n        ax.plot(rounds, drift_ratios, 'r-', linewidth=2, label='Drift Ratio', alpha=0.8)\n        ax.fill_between(rounds, mitigation_status, alpha=0.3, \n                       label='Mitigation Active', color='red')\n        \n        # Mark important events\n        injection_round = self.config['drift']['injection_round']\n        ax.axvline(x=injection_round, color='orange', linestyle='--', \n                  linewidth=3, label='Drift Injection')\n        \n        # Add annotations for key phases\n        ax.annotate('Baseline Training', xy=(injection_round//2, 0.8), \n                   xytext=(injection_round//2, 0.9),\n                   arrowprops=dict(arrowstyle='->', color='black'),\n                   ha='center', fontsize=10, fontweight='bold')\n        \n        ax.annotate('Drift & Recovery', xy=(injection_round + 10, 0.8), \n                   xytext=(injection_round + 10, 0.9),\n                   arrowprops=dict(arrowstyle='->', color='black'),\n                   ha='center', fontsize=10, fontweight='bold')\n        \n        ax.set_xlabel('Training Round')\n        ax.set_ylabel('Normalized Values')\n        ax.set_title('Complete System Timeline - All Key Metrics')\n        ax.legend(loc='upper right')\n        ax.grid(True, alpha=0.3)\n        ax.set_ylim(0, 1)\n    \n    def _create_detailed_analysis_plots(self):\n        \"\"\"Create additional detailed analysis plots.\"\"\"\n        # Detailed drift detection analysis\n        fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n        \n        # 1. Client-by-client drift timeline\n        ax = axes[0, 0]\n        for client_id, history in self.client_drift_histories.items():\n            combined_drift = history['detection_history']['combined']\n            rounds = range(1, len(combined_drift) + 1)\n            \n            # Create step plot for drift events\n            cumulative_drift = np.cumsum(combined_drift)\n            ax.step(rounds, cumulative_drift, where='post', \n                   label=f'Client {client_id}', linewidth=2)\n        \n        ax.set_xlabel('Round')\n        ax.set_ylabel('Cumulative Drift Detections')\n        ax.set_title('Client-by-Client Drift Timeline')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        \n        # 2. Detection method effectiveness\n        ax = axes[0, 1]\n        methods = ['ADWIN', 'MMD', 'Statistical']\n        true_positives = [0, 0, 0]  # Simplified for demo\n        false_positives = [0, 0, 0]\n        \n        # Calculate based on injection round\n        injection_round = self.config['drift']['injection_round']\n        \n        for client_id, history in self.client_drift_histories.items():\n            detection_history = history['detection_history']\n            \n            for i, method in enumerate(['adwin', 'mmd', 'statistical']):\n                detections = detection_history.get(method, [])\n                for round_idx, detected in enumerate(detections):\n                    if detected:\n                        if round_idx >= injection_round - 1:  # True positive\n                            true_positives[i] += 1\n                        else:  # False positive\n                            false_positives[i] += 1\n        \n        x = np.arange(len(methods))\n        width = 0.35\n        \n        ax.bar(x - width/2, true_positives, width, label='True Positives', color='green', alpha=0.7)\n        ax.bar(x + width/2, false_positives, width, label='False Positives', color='red', alpha=0.7)\n        \n        ax.set_xlabel('Detection Method')\n        ax.set_ylabel('Count')\n        ax.set_title('Detection Method Effectiveness')\n        ax.set_xticks(x)\n        ax.set_xticklabels(methods)\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        \n        # 3. Performance distribution analysis\n        ax = axes[1, 0]\n        \n        # Collect all client accuracies by phase\n        baseline_accs = []\n        drift_accs = []\n        recovery_accs = []\n        \n        injection_round = self.config['drift']['injection_round']\n        \n        for client_id, history in self.client_drift_histories.items():\n            metrics = history['round_metrics']\n            if metrics:\n                # Baseline phase\n                baseline_metrics = [m for m in metrics if m['round'] < injection_round]\n                if baseline_metrics:\n                    baseline_accs.extend([m['accuracy'] for m in baseline_metrics])\n                \n                # Drift phase\n                drift_metrics = [m for m in metrics if injection_round <= m['round'] < injection_round + 5]\n                if drift_metrics:\n                    drift_accs.extend([m['accuracy'] for m in drift_metrics])\n                \n                # Recovery phase\n                recovery_metrics = [m for m in metrics if m['round'] >= injection_round + 5]\n                if recovery_metrics:\n                    recovery_accs.extend([m['accuracy'] for m in recovery_metrics])\n        \n        # Create box plot\n        data_to_plot = [baseline_accs, drift_accs, recovery_accs]\n        labels = ['Baseline', 'Drift Impact', 'Recovery']\n        \n        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)\n        colors = ['lightblue', 'lightcoral', 'lightgreen']\n        for patch, color in zip(box_plot['boxes'], colors):\n            patch.set_facecolor(color)\n        \n        ax.set_ylabel('Accuracy')\n        ax.set_title('Performance Distribution by Phase')\n        ax.grid(True, alpha=0.3)\n        \n        # 4. System resilience metrics\n        ax = axes[1, 1]\n        \n        # Calculate key resilience metrics\n        metrics_names = []\n        metrics_values = []\n        \n        # Detection delay\n        injection_round = self.config['drift']['injection_round']\n        first_detection_round = None\n        \n        for round_data in self.simulation_results['aggregation_history']:\n            if round_data and round_data.get('drift_ratio', 0) > 0:\n                round_num = round_data['round']\n                if round_num >= injection_round:\n                    first_detection_round = round_num\n                    break\n        \n        detection_delay = first_detection_round - injection_round if first_detection_round else float('inf')\n        metrics_names.append('Detection Delay\\n(rounds)')\n        metrics_values.append(detection_delay if detection_delay != float('inf') else 0)\n        \n        # Recovery time\n        baseline_acc = np.mean([m['accuracy'] for m in self.simulation_results['global_metrics'][:injection_round-1] if m])\n        recovery_threshold = baseline_acc * 0.95  # 95% of baseline\n        \n        recovery_round = None\n        for i, metrics in enumerate(self.simulation_results['global_metrics'][injection_round:], injection_round):\n            if metrics and metrics['accuracy'] >= recovery_threshold:\n                recovery_round = i\n                break\n        \n        recovery_time = recovery_round - injection_round if recovery_round else float('inf')\n        metrics_names.append('Recovery Time\\n(rounds)')\n        metrics_values.append(recovery_time if recovery_time != float('inf') else 0)\n        \n        # Final recovery rate\n        final_acc = np.mean([m['accuracy'] for m in self.simulation_results['global_metrics'][-5:] if m])\n        recovery_rate = final_acc / baseline_acc if baseline_acc > 0 else 0\n        metrics_names.append('Recovery Rate\\n(%)')\n        metrics_values.append(recovery_rate * 100)\n        \n        # System robustness (1 - max performance drop)\n        min_acc_during_drift = min([m['accuracy'] for m in self.simulation_results['global_metrics'][injection_round:injection_round+5] if m])\n        performance_drop = (baseline_acc - min_acc_during_drift) / baseline_acc if baseline_acc > 0 else 0\n        robustness = (1 - performance_drop) * 100\n        metrics_names.append('Robustness\\n(%)')\n        metrics_values.append(robustness)\n        \n        # Create bar plot\n        bars = ax.bar(metrics_names, metrics_values, \n                     color=['blue', 'orange', 'green', 'purple'], alpha=0.7)\n        \n        # Add value labels\n        for bar, value in zip(bars, metrics_values):\n            height = bar.get_height()\n            ax.text(bar.get_x() + bar.get_width()/2., height + max(metrics_values)*0.01,\n                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')\n        \n        ax.set_ylabel('Value')\n        ax.set_title('System Resilience Metrics')\n        ax.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.suptitle('🔍 Detailed Drift Analysis & System Resilience', \n                    fontsize=14, fontweight='bold', y=0.98)\n        plt.show()\n\n    def print_comprehensive_summary(self):\n        \"\"\"Print detailed text summary of simulation results.\"\"\"\n        print(\"\\n\" + \"=\"*80)\n        print(\"📊 COMPREHENSIVE SIMULATION ANALYSIS SUMMARY\")\n        print(\"=\"*80)\n        \n        # Basic simulation info\n        print(f\"\\n🎯 SIMULATION CONFIGURATION:\")\n        print(f\"   • Execution Mode: {EXECUTION_MODE}\")\n        print(f\"   • Total Rounds: {self.config['simulation']['num_rounds']}\")\n        print(f\"   • Number of Clients: {self.config['federated']['num_clients']}\")\n        print(f\"   • Drift Injection Round: {self.config['drift']['injection_round']}\")\n        print(f\"   • Affected Clients: {self.config['drift']['affected_clients']}\")\n        print(f\"   • Drift Types: {', '.join(self.config['drift']['drift_types'])}\")\n        \n        # Performance analysis\n        print(f\"\\n🎯 PERFORMANCE ANALYSIS:\")\n        injection_round = self.config['drift']['injection_round']\n        \n        # Baseline performance\n        baseline_metrics = [m for m in self.simulation_results['global_metrics'][:injection_round-1] if m]\n        baseline_acc = np.mean([m['accuracy'] for m in baseline_metrics]) if baseline_metrics else 0\n        baseline_loss = np.mean([m['loss'] for m in baseline_metrics]) if baseline_metrics else 0\n        \n        print(f\"   • Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)\")\n        print(f\"   • Baseline Loss: {baseline_loss:.4f}\")\n        \n        # Performance during drift\n        drift_metrics = [m for m in self.simulation_results['global_metrics'][injection_round:injection_round+5] if m]\n        if drift_metrics:\n            drift_acc = np.mean([m['accuracy'] for m in drift_metrics])\n            drift_loss = np.mean([m['loss'] for m in drift_metrics])\n            performance_drop = ((baseline_acc - drift_acc) / baseline_acc * 100) if baseline_acc > 0 else 0\n            \n            print(f\"   • Performance During Drift: {drift_acc:.4f} ({drift_acc*100:.1f}%)\")\n            print(f\"   • Performance Drop: {performance_drop:.1f}%\")\n        \n        # Recovery performance\n        recovery_metrics = [m for m in self.simulation_results['global_metrics'][-5:] if m]\n        if recovery_metrics:\n            recovery_acc = np.mean([m['accuracy'] for m in recovery_metrics])\n            recovery_rate = ((recovery_acc - drift_acc) / (baseline_acc - drift_acc) * 100) if (baseline_acc > drift_acc) else 0\n            \n            print(f\"   • Final Recovery Accuracy: {recovery_acc:.4f} ({recovery_acc*100:.1f}%)\")\n            print(f\"   • Recovery Rate: {recovery_rate:.1f}%\")\n        \n        # Drift detection analysis\n        print(f\"\\n🚨 DRIFT DETECTION ANALYSIS:\")\n        \n        total_drift_detections = 0\n        method_detections = {'ADWIN': 0, 'MMD': 0, 'Statistical': 0}\n        \n        for client_id, history in self.client_drift_histories.items():\n            detection_history = history['detection_history']\n            total_drift_detections += sum(detection_history.get('combined', []))\n            method_detections['ADWIN'] += sum(detection_history.get('adwin', []))\n            method_detections['MMD'] += sum(detection_history.get('mmd', []))\n            method_detections['Statistical'] += sum(detection_history.get('statistical', []))\n        \n        print(f\"   • Total Drift Detections: {total_drift_detections}\")\n        print(f\"   • ADWIN Detections: {method_detections['ADWIN']}\")\n        print(f\"   • MMD Detections: {method_detections['MMD']}\")\n        print(f\"   • Statistical Detections: {method_detections['Statistical']}\")\n        \n        # Calculate detection delay\n        first_detection_round = None\n        for round_data in self.simulation_results['aggregation_history']:\n            if round_data and round_data.get('drift_ratio', 0) > 0:\n                round_num = round_data['round']\n                if round_num >= injection_round:\n                    first_detection_round = round_num\n                    break\n        \n        detection_delay = first_detection_round - injection_round if first_detection_round else float('inf')\n        print(f\"   • Detection Delay: {detection_delay if detection_delay != float('inf') else 'Not detected'} rounds\")\n        \n        # Mitigation analysis\n        print(f\"\\n🛡️ MITIGATION ANALYSIS:\")\n        \n        mitigation_rounds = []\n        for round_data in self.simulation_results['aggregation_history']:\n            if round_data and round_data.get('mitigation_active', False):\n                mitigation_rounds.append(round_data['round'])\n        \n        print(f\"   • Mitigation Strategy: FedTrimmedAvg (β={self.config['drift_detection']['trimmed_beta']})\")\n        print(f\"   • Mitigation Triggered: {'Yes' if mitigation_rounds else 'No'}\")\n        if mitigation_rounds:\n            print(f\"   • Mitigation Duration: {len(mitigation_rounds)} rounds\")\n            print(f\"   • Mitigation Period: Rounds {min(mitigation_rounds)} - {max(mitigation_rounds)}\")\n        \n        # Fairness analysis\n        print(f\"\\n⚖️ FAIRNESS ANALYSIS:\")\n        \n        # Calculate fairness metrics\n        final_client_accs = []\n        for client_id, history in self.client_drift_histories.items():\n            metrics = history['round_metrics']\n            if metrics:\n                final_acc = np.mean([m['accuracy'] for m in metrics[-3:]])  # Last 3 rounds\n                final_client_accs.append(final_acc)\n        \n        if final_client_accs:\n            fairness_gap = max(final_client_accs) - min(final_client_accs)\n            fairness_ratio = min(final_client_accs) / max(final_client_accs) if max(final_client_accs) > 0 else 0\n            \n            print(f\"   • Final Fairness Gap: {fairness_gap:.4f} ({fairness_gap*100:.1f}%)\")\n            print(f\"   • Fairness Ratio: {fairness_ratio:.3f}\")\n            print(f\"   • Min Client Accuracy: {min(final_client_accs):.4f}\")\n            print(f\"   • Max Client Accuracy: {max(final_client_accs):.4f}\")\n        \n        # System resilience summary\n        print(f\"\\n🏗️ SYSTEM RESILIENCE:\")\n        \n        # Overall system rating\n        resilience_score = 0\n        \n        # Detection capability (30%)\n        if detection_delay < 3:\n            resilience_score += 30\n        elif detection_delay < 5:\n            resilience_score += 20\n        elif detection_delay < 10:\n            resilience_score += 10\n        \n        # Recovery capability (40%)\n        if recovery_rate > 80:\n            resilience_score += 40\n        elif recovery_rate > 60:\n            resilience_score += 30\n        elif recovery_rate > 40:\n            resilience_score += 20\n        elif recovery_rate > 20:\n            resilience_score += 10\n        \n        # Fairness maintenance (20%)\n        if fairness_gap < 0.05:\n            resilience_score += 20\n        elif fairness_gap < 0.10:\n            resilience_score += 15\n        elif fairness_gap < 0.15:\n            resilience_score += 10\n        elif fairness_gap < 0.20:\n            resilience_score += 5\n        \n        # Robustness (10%)\n        if performance_drop < 5:\n            resilience_score += 10\n        elif performance_drop < 10:\n            resilience_score += 8\n        elif performance_drop < 15:\n            resilience_score += 6\n        elif performance_drop < 20:\n            resilience_score += 4\n        \n        print(f\"   • Overall Resilience Score: {resilience_score}/100\")\n        \n        if resilience_score >= 80:\n            rating = \"EXCELLENT 🏆\"\n        elif resilience_score >= 60:\n            rating = \"GOOD ✅\"\n        elif resilience_score >= 40:\n            rating = \"MODERATE ⚖️\"\n        else:\n            rating = \"NEEDS IMPROVEMENT ⚠️\"\n        \n        print(f\"   • System Rating: {rating}\")\n        \n        # Key findings\n        print(f\"\\n🔍 KEY FINDINGS:\")\n        findings = []\n        \n        if detection_delay <= 3:\n            findings.append(\"✅ Fast drift detection capability\")\n        elif detection_delay > 10:\n            findings.append(\"⚠️ Slow drift detection - consider tuning parameters\")\n        \n        if recovery_rate > 80:\n            findings.append(\"✅ Excellent recovery performance\")\n        elif recovery_rate < 50:\n            findings.append(\"⚠️ Poor recovery - mitigation strategy needs improvement\")\n        \n        if fairness_gap < 0.1:\n            findings.append(\"✅ Good fairness across clients\")\n        else:\n            findings.append(\"⚠️ High fairness gap - consider client-specific adaptations\")\n        \n        if performance_drop < 10:\n            findings.append(\"✅ System robust to drift\")\n        else:\n            findings.append(\"⚠️ Significant performance impact from drift\")\n        \n        if mitigation_rounds:\n            findings.append(\"✅ Automatic mitigation successfully triggered\")\n        else:\n            findings.append(\"⚠️ Mitigation not triggered - check threshold settings\")\n        \n        for i, finding in enumerate(findings, 1):\n            print(f\"   {i}. {finding}\")\n        \n        print(f\"\\n🎉 SIMULATION COMPLETED SUCCESSFULLY!\")\n        print(\"=\"*80)\n\n\n# =====================================\n# Main Execution Cell\n# =====================================\n\ndef run_complete_federated_simulation():\n    \"\"\"Main execution function to run the complete simulation.\"\"\"\n    print(\"🚀 Starting Complete Federated Learning Drift Detection Simulation...\")\n    print(f\"📊 Mode: {EXECUTION_MODE}\")\n    print(f\"🎯 Configuration: {CONFIG['simulation']['num_rounds']} rounds, {CONFIG['federated']['num_clients']} clients\")\n    \n    # Create and run simulation\n    simulation = FederatedDriftSimulation(CONFIG)\n    \n    try:\n        # Run the simulation\n        start_time = time.time()\n        results = simulation.run_simulation()\n        end_time = time.time()\n        \n        print(f\"\\n⏱️ Simulation completed in {end_time - start_time:.1f} seconds\")\n        \n        # Get comprehensive results\n        comprehensive_results = simulation.get_comprehensive_results()\n        \n        # Create visualizations\n        visualizer = ComprehensiveVisualizer(comprehensive_results)\n        \n        # Print summary\n        visualizer.print_comprehensive_summary()\n        \n        # Create dashboard\n        visualizer.create_comprehensive_dashboard()\n        \n        return comprehensive_results\n        \n    except Exception as e:\n        print(f\"❌ Simulation failed: {str(e)}\")\n        import traceback\n        traceback.print_exc()\n        return None\n\n\nprint(\"✅ Complete federated learning system ready!\")\nprint(\"\\n🎯 To run the simulation, call: run_complete_federated_simulation()\")\nprint(\"\\n📊 This will execute:\")\nprint(\"   1. Complete federated learning simulation\")\nprint(\"   2. Multi-level drift detection\")\nprint(\"   3. Adaptive mitigation with FedTrimmedAvg\")\nprint(\"   4. Comprehensive performance analysis\")\nprint(\"   5. Advanced visualization dashboard\")\nprint(\"   6. Detailed results summary\")