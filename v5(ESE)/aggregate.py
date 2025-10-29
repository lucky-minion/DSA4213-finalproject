"""
Memory-Optimized Model Aggregator with Original BERT Comparison
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
from collections import defaultdict
from config import ModelConfig, DatasetConfig, TrainingConfig, SystemConfig
from split_model import ModelPart0, ModelPart1, ModelPart2
from dataset import get_test_dataloaders

class MemorySafeAggregator:
    def __init__(self):
        self.device = SystemConfig.DEVICE
        self.aggregation_interval = TrainingConfig.AGGREGATION_INTERVAL
        
        # Store only metadata initially
        self.model_metadata = []
        self._scan_model_files()
        
        # Original untrained models
        self.original_part0 = ModelPart0().to(self.device).eval()
        self.original_part1 = ModelPart1().to(self.device).eval()
        
        # Track results per method
        self.results = {
            'fedavg': defaultdict(list),
            'elite_weighted': defaultdict(list),
            'client_avg': defaultdict(list),
            'original_bert': defaultdict(list)  # For original BERT training results
        }
        
        os.makedirs("./aggregated_models", exist_ok=True)
        self._load_original_bert_results()

    def _load_original_bert_results(self):
        """Load accuracy results from original BERT training"""
        original_files = glob(os.path.join(SystemConfig.FULL_MODEL_SAVE_DIR, "fullbert_epoch-*.pt"))
        for f in original_files:
            try:
                # Extract epoch and accuracy from filename
                match = re.search(r'epoch-(\d+)_acc-([\d.]+)', f)
                if match:
                    epoch = int(match.group(1))
                    acc = float(match.group(2))
                    self.results['original_bert']['epoch'].append(epoch)
                    self.results['original_bert']['accuracy'].append(acc * 100)  # Convert to percentage
            except Exception as e:
                print(f"Error parsing {f}: {str(e)}")

    def _scan_model_files(self):
        """Scan model files and extract metadata without loading"""
        model_files = glob("./client_saved_models/modelpart2/*.pt")
        for f in model_files:
            try:
                fname = os.path.basename(f)
                parts = fname.split('_')
                client_id = int(parts[0].split('-')[1])
                epoch = int(parts[1].split('-')[1])
                acc = float(parts[2].split('-')[1])
                self.model_metadata.append({
                    'path': f,
                    'client_id': client_id,
                    'epoch': epoch,
                    'accuracy': acc
                })
            except Exception as e:
                print(f"Error parsing {f}: {str(e)}")

    def _load_models_for_epoch(self, target_epoch):
        """Yield model paths for specific epoch one at a time"""
        for meta in self.model_metadata:
            if meta['epoch'] == target_epoch:
                yield meta

    def _load_single_model(self, meta):
        """Load a single model from metadata"""
        try:
            state_dict = torch.load(meta['path'], map_location='cpu')
            model = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to('cpu')
            model.load_state_dict(state_dict)
            return {
                'model': model.to(self.device),
                'accuracy': meta['accuracy'],
                'client_id': meta['client_id']
            }
        except Exception as e:
            print(f"Error loading {meta['path']}: {str(e)}")
            return None

    def _fedavg(self, model_paths):
        """Basic federated averaging - processes models one at a time"""
        if not model_paths:
            return None
            
        first_model = self._load_single_model(next(model_paths))
        if not first_model:
            return None
            
        avg_state = {}
        model_state = first_model['model'].state_dict()
        for name, param in model_state.items():
            avg_state[name] = param.data.clone()
        
        total_models = 1
        
        for meta in model_paths:
            model = self._load_single_model(meta)
            if not model:
                continue
                
            model_state = model['model'].state_dict()
            for name in avg_state:
                avg_state[name] += model_state[name].data
            total_models += 1
            
            del model['model']
            torch.cuda.empty_cache()
        
        for name in avg_state:
            avg_state[name] = avg_state[name] / total_models
        
        return avg_state

    def _elite_weighted_avg(self, model_paths):
        """Accuracy-weighted averaging - processes models one at a time"""
        if not model_paths:
            return None
            
        model_infos = []
        for meta in model_paths:
            model_infos.append({
                'path': meta['path'],
                'accuracy': meta['accuracy'],
                'client_id': meta['client_id']
            })
        
        accuracies = np.array([m['accuracy'] for m in model_infos])
        temperature = 0.5  # Lower T => higher sharpness (more elite emphasis)
        exp_scores = np.exp(accuracies / temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        print("Model weights:", ["%.2f" % w for w in weights])
        
        avg_state = None
        for i, meta in enumerate(model_infos):
            model = self._load_single_model(meta)
            if not model:
                continue
                
            model_state = model['model'].state_dict()
            if avg_state is None:
                avg_state = {}
                for name, param in model_state.items():
                    avg_state[name] = param.data.clone() * weights[i]
            else:
                for name in avg_state:
                    avg_state[name] += model_state[name].data * weights[i]
            
            del model['model']
            torch.cuda.empty_cache()
        
        return avg_state

    def _save_model(self, state_dict, method, epoch):
        """Save aggregated model"""
        model = ModelPart2(DatasetConfig.DATASET_NUM_CLASSES).to(self.device)
        model.load_state_dict(state_dict)
        path = f"./aggregated_models/{method}_epoch{epoch}.pt"
        return model

    def _evaluate(self, model, epoch, method):
        """Evaluate model and store results"""
        test_loader = get_test_dataloaders(batch_size=32)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                hidden0 = self.original_part0(inputs, masks)
                hidden1 = self.original_part1(hidden0, masks)
                outputs = model(hidden1, masks)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        self.results[method]['epoch'].append(epoch)
        self.results[method]['accuracy'].append(accuracy)
        print(f"{method.upper()} @ Epoch {epoch}: {accuracy:.2f}%")
        
        del model
        torch.cuda.empty_cache()

    def _evaluate_client_models(self, model_paths, epoch):
        """Evaluate all client models and calculate average accuracy"""
        test_loader = get_test_dataloaders(batch_size=32)
        total_acc = 0.0
        valid_models = 0
        
        for meta in model_paths:
            model = self._load_single_model(meta)
            if not model:
                continue
                
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["input_ids"].to(self.device)
                    masks = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    hidden0 = self.original_part0(inputs, masks)
                    hidden1 = self.original_part1(hidden0, masks)
                    outputs = model['model'](hidden1, masks)
                    
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            acc = 100 * correct / total
            total_acc += acc
            valid_models += 1
            
            del model['model']
            torch.cuda.empty_cache()
        
        if valid_models > 0:
            avg_acc = total_acc / valid_models
            self.results['client_avg']['epoch'].append(epoch)
            self.results['client_avg']['accuracy'].append(avg_acc)
            print(f"CLIENT AVG @ Epoch {epoch}: {avg_acc:.2f}%")

    def run(self):
        """Execute aggregation pipeline"""
        max_epoch = TrainingConfig.NUM_EPOCHS
        
        for epoch in range(self.aggregation_interval, max_epoch+1, self.aggregation_interval):
            print(f"\n=== Processing Epoch {epoch} ===")
            
            model_paths = list(self._load_models_for_epoch(epoch))
            if not model_paths:
                print(f"No models found for epoch {epoch}")
                continue
                
            self._evaluate_client_models(model_paths, epoch)
            
            fedavg_state = self._fedavg(iter(model_paths))
            if fedavg_state:
                fedavg_model = self._save_model(fedavg_state, 'fedavg', epoch)
                self._evaluate(fedavg_model, epoch, 'fedavg')
            
            weighted_state = self._elite_weighted_avg(iter(model_paths))
            if weighted_state:
                weighted_model = self._save_model(weighted_state, 'elite_weighted', epoch)
                self._evaluate(weighted_model, epoch, 'elite_weighted')
        
        self._plot_comparison()
        self._plot_bert_comparison()

    def _plot_comparison(self):
        """Plot all FL method results"""
        plt.figure(figsize=(12, 6))
        
        colors = {
            'fedavg': 'blue',
            'elite_weighted': 'purple',
            'client_avg': 'orange'
        }
        
        labels = {
            'fedavg': 'FedAvg',
            'elite_weighted': 'Elite Weighted',
            'client_avg': 'Client Average'
        }
        
        for method, data in self.results.items():
            if method == 'original_bert' or not data['epoch']:
                continue
                
            plt.plot(data['epoch'], data['accuracy'],
                    label=labels[method],
                    color=colors[method],
                    marker='o',
                    linestyle='-',
                    linewidth=2)
        
        plt.title("Federated Learning Method Comparison", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plot_path = "./aggregated_models/fl_method_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved FL comparison plot to {plot_path}")

    def _plot_bert_comparison(self):
        """Plot comparison between original BERT and elite-weighted FL"""
        if not self.results['original_bert']['epoch'] or not self.results['elite_weighted']['epoch']:
            print("Warning: Not enough data for BERT comparison plot")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Sort both results by epoch for proper plotting
        bert_epochs, bert_accs = zip(*sorted(zip(self.results['original_bert']['epoch'], 
                                            self.results['original_bert']['accuracy'])))
        fl_epochs, fl_accs = zip(*sorted(zip(self.results['elite_weighted']['epoch'], 
                                        self.results['elite_weighted']['accuracy'])))
        
        # Original BERT curve
        plt.plot(bert_epochs, 
                bert_accs,
                label='Original BERT',
                color='red',
                marker='o',
                linestyle='-',
                linewidth=2)
        
        # Elite-weighted FL curve
        plt.plot(fl_epochs,
                fl_accs,
                label='Elite Weighted FL',
                color='purple',
                marker='s',
                linestyle='--',
                linewidth=2)
        
        plt.title("Original BERT vs Federated Learning Performance", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Test Accuracy (%)", fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set proper axis limits
        min_epoch = min(min(bert_epochs), min(fl_epochs))
        max_epoch = max(max(bert_epochs), max(fl_epochs))
        plt.xlim(min_epoch - 1, max_epoch + 1)
        
        # Set y-axis to show typical accuracy range (adjust as needed)
        min_acc = min(min(bert_accs), min(fl_accs))
        max_acc = max(max(bert_accs), max(fl_accs))
        plt.ylim(max(0, min_acc - 5), min(100, max_acc + 5))
        
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plot_path = "./aggregated_models/bert_fl_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved BERT vs FL comparison plot to {plot_path}")

if __name__ == "__main__":
    aggregator = MemorySafeAggregator()
    aggregator.run()