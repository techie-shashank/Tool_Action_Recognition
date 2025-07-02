import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import random

class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization system for ML models.
    Handles model-specific hyperparameters, training parameters, and data processing options.
    """
    
    def __init__(self, config_path: str = "../config.json", results_dir: str = "../hyperopt_results"):
        self.config_path = config_path
        self.results_dir = results_dir
        self.best_results = defaultdict(dict)
        self.all_results = []
        
        # results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load base config
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)
    
    def define_hyperparameter_space(self):
        """Define the hyperparameter search space for each model and component."""
        
        # Model-specific hyperparameters
        model_hyperparams = {
            "lstm": {
                "hidden_size": [32, 64, 128, 256],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "tcn": {
                "num_channels": [
                    [32, 32, 32],
                    [64, 64, 64], 
                    [32, 64, 32],
                    [64, 128, 64],
                    [32, 64, 128],
                    [128, 64, 32]
                ],
                "kernel_size": [2, 3, 4, 5],
                "dropout": [0.1, 0.2, 0.3, 0.4]
            }
        }
        
        # Training hyperparameters
        training_hyperparams = {
            "learning_rate": [0.0001, 0.0005, 0.001, 0.002, 0.005],
            "batch_size": [16, 32, 64, 128],
            "epochs": [20, 30, 50, 75, 100],
            "l1_lambda": [0.0, 0.1, 0.5, 1.0],
            "gradient_clip": [0.5, 1.0, 2.0, 5.0]
        }
        
        # Data balancing strategies
        data_balancing_options = [
            [],
            ["oversample"],
            ["focal_loss"],
            ["resample"],
            ["weighted_sampling"],
            ["augment"],
            ["focal_loss", "weighted_sampling"],
            ["oversample", "augment"]
        ]
        
        # Semi-supervised learning configurations
        semi_supervised_configs = [
            {"active": False},
            {
                "active": True,
                "strategy": "contrastive",
                "labelled_ratio": 0.1,
                "temperature": 0.3
            },
            {
                "active": True,
                "strategy": "contrastive", 
                "labelled_ratio": 0.25,
                "temperature": 0.5
            },
            {
                "active": True,
                "strategy": "pseudo_labeling",
                "labelled_ratio": 0.1,
                "threshold": 0.8
            },
            {
                "active": True,
                "strategy": "pseudo_labeling",
                "labelled_ratio": 0.25,
                "threshold": 0.9
            },
            {
            "active": True,
            "strategy": "mean_teacher",
            "labelled_ratio": 0.1,
            "alpha": 0.99,
            "lambda_consistency": 1.0
        }
        ]
        
        # Focal Loss specific hyperparameters
        focal_loss_params = {
            "gamma": [0.5, 1.0, 1.5, 2.0, 2.5],
            "alpha_strategy": ["balanced", "manual"]
        }
        
        return {
            "model_hyperparams": model_hyperparams,
            "training_hyperparams": training_hyperparams,
            "data_balancing_options": data_balancing_options,
            "semi_supervised_configs": semi_supervised_configs,
            "focal_loss_params": focal_loss_params
        }
    
    def generate_configurations(self, models: List[str], tools: List[str], 
                              max_configs_per_model: int = 50) -> List[Dict]:
        """Generate hyperparameter configurations using different strategies."""
        
        hyperparams = self.define_hyperparameter_space()
        configurations = []
        
        for model in models:
            for tool in tools:
                # Strategy 1: Grid Search (limited)
                grid_configs = self._generate_grid_search_configs(
                    model, tool, hyperparams, max_configs_per_model // 3 # //3 to balance with other strategies
                )
                
                # Strategy 2: Random Search
                random_configs = self._generate_random_search_configs(
                    model, tool, hyperparams, max_configs_per_model // 3
                )
                
                # Strategy 3: Focused Search
                focused_configs = self._generate_focused_search_configs(
                    model, tool, hyperparams, max_configs_per_model // 3
                )
                
                configurations.extend(grid_configs + random_configs + focused_configs)
        
        return configurations
    
    def _generate_grid_search_configs(self, model: str, tool: str, 
                                    hyperparams: Dict, max_configs: int) -> List[Dict]:
        """Generate configurations using grid search (limited combinations)."""
        configs = []
        
        # Select key hyperparameters for grid search
        model_params = hyperparams["model_hyperparams"][model]
        training_params = {
            "learning_rate": hyperparams["training_hyperparams"]["learning_rate"][:3],
            "batch_size": hyperparams["training_hyperparams"]["batch_size"][:2] ,
            "epochs": [30, 50]
        }
        
        # Generate combinations
        for model_combo in itertools.product(*model_params.values()):
            for train_combo in itertools.product(*training_params.values()):
                if len(configs) >= max_configs:
                    break
                    
                config = self._create_config(
                    model, tool, model_combo, train_combo, 
                    model_params.keys(), training_params.keys(), hyperparams
                )
                configs.append(config)
        
        return configs[:max_configs]
    
    def _generate_random_search_configs(self, model: str, tool: str,
                                      hyperparams: Dict, max_configs: int) -> List[Dict]:
        """Generate configurations using random search."""
        configs = []
        random.seed(42)  # For reproducibility
        
        model_params = hyperparams["model_hyperparams"][model]
        training_params = hyperparams["training_hyperparams"]
        
        for _ in range(max_configs):
            # Randomly sample model hyperparameters
            model_combo = tuple(
                random.choice(param_values) 
                for param_values in model_params.values()
            )
            
            # Randomly sample training hyperparameters
            train_combo = tuple(
                random.choice(param_values)
                for param_values in training_params.values()
            )
            
            config = self._create_config(
                model, tool, model_combo, train_combo,
                model_params.keys(), training_params.keys(), hyperparams
            )
            configs.append(config)
        
        return configs
    
    def _generate_focused_search_configs(self, model: str, tool: str,
                                       hyperparams: Dict, max_configs: int) -> List[Dict]:
        """Generate configurations based on best practices and model-specific recommendations."""
        configs = []
        
        # Model-specific best practice configurations
        if model == "lstm":
            focused_combinations = [
                {"hidden_size": 64, "num_layers": 2, "dropout": 0.3},
                {"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
                {"hidden_size": 256, "num_layers": 1, "dropout": 0.4},
                {"hidden_size": 128, "num_layers": 3, "dropout": 0.3},
            ]
        elif model == "tcn":
            focused_combinations = [
                {"num_channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.2},
                {"num_channels": [32, 64, 128], "kernel_size": 4, "dropout": 0.3},
                {"num_channels": [128, 64, 32], "kernel_size": 3, "dropout": 0.2},
                {"num_channels": [64, 128, 64], "kernel_size": 5, "dropout": 0.3},
            ]
        else:  # fcn
            focused_combinations = [
                {"conv_layers": [128, 256, 128], "dropout": 0.2},
                {"conv_layers": [64, 128, 256], "dropout": 0.3},
                {"conv_layers": [256, 128, 64], "dropout": 0.2},
            ]
        
        # Best practice training configurations
        training_combinations = [
            {"learning_rate": 0.001, "batch_size": 64, "epochs": 50},
            {"learning_rate": 0.0005, "batch_size": 32, "epochs": 75},
            {"learning_rate": 0.002, "batch_size": 128, "epochs": 30},
        ]
        
        for model_params in focused_combinations:
            for train_params in training_combinations:
                if len(configs) >= max_configs:
                    break
                
                config = self._create_focused_config(
                    model, tool, model_params, train_params, hyperparams
                )
                configs.append(config)
        
        return configs[:max_configs]
    
    def _create_config(self, model: str, tool: str, model_combo: tuple, 
                      train_combo: tuple, model_keys: List, train_keys: List,
                      hyperparams: Dict) -> Dict:
        """Create a configuration dictionary from parameter combinations."""
        
        config = self.base_config.copy()
        
        # Update model-specific parameters
        model_params = dict(zip(model_keys, model_combo))
        
        # Update training parameters
        train_params = dict(zip(train_keys, train_combo))
        config.update(train_params)
        
        # Randomly select data balancing and semi-supervised options
        # config["data_balancing"] = np.random.choice(
        #     hyperparams["data_balancing_options"]
        # )
        # config["semi_supervised"] = np.random.choice(
        #     hyperparams["semi_supervised_configs"]
        # )
        config["data_balancing"] = random.choice(hyperparams["data_balancing_options"])
        config["semi_supervised"] = random.choice(hyperparams["semi_supervised_configs"])
        
        # Add focal loss parameters if focal loss is selected
        if "focal_loss" in config["data_balancing"]:
            config["focal_loss_gamma"] = np.random.choice(
                hyperparams["focal_loss_params"]["gamma"]
            )
        
        return {
            "model": model,
            "tool": tool,
            "sensor": "all",
            "config": config,
            "model_params": model_params
        }
    
    def _create_focused_config(self, model: str, tool: str, 
                             model_params: Dict, train_params: Dict,
                             hyperparams: Dict) -> Dict:
        """Create a focused configuration with best practices."""
        
        config = self.base_config.copy()
        config.update(train_params)
        
        # Use effective data balancing strategies
        effective_balancing = [
            ["weighted_sampling"],
            ["focal_loss", "weighted_sampling"],
            ["oversample", "augment"]
        ]
        config["data_balancing"] = random.choice(effective_balancing)
        
        
        # Use promising semi-supervised configurations
        effective_semi = [
            {"active": False},
            {
                "active": True,
                "strategy": "contrastive",
                "labelled_ratio": 0.25,
                "temperature": 0.5
            }
        ]
        config["semi_supervised"] = random.choice(effective_semi)
        
        return {
            "model": model,
            "tool": tool,
            "sensor": "all",
            "config": config,
            "model_params": model_params
        }
    
    def run_optimization(self, models: List[str], tools: List[str], 
                        max_configs_per_model: int = 50):
        """Run the hyperparameter optimization process."""
        
        print(f"Starting hyperparameter optimization at {datetime.now()}")
        print(f"Models: {models}")
        print(f"Tools: {tools}")
        print(f"Max configurations per model: {max_configs_per_model}")
        
        # Generate configurations
        configurations = self.generate_configurations(
            models, tools, max_configs_per_model
        )
        
        print(f"Generated {len(configurations)} configurations")
        
        # Run experiments
        for i, config_dict in enumerate(configurations):
            print(f"\n{'='*60}")
            print(f"Running configuration {i+1}/{len(configurations)}")
            print(f"Model: {config_dict['model']}, Tool: {config_dict['tool']}")
            print(f"{'='*60}")
            
            try:
                result = self._run_single_experiment(config_dict, i+1)
                self.all_results.append(result)
                
                # Update best results
                key = f"{config_dict['model']}_{config_dict['tool']}"
                if key not in self.best_results or result['accuracy'] > self.best_results[key]['accuracy']:
                    self.best_results[key] = result
                
                # Save intermediate results
                self._save_results()
                
            except Exception as e:
                print(f"❌ Configuration {i+1} failed: {str(e)}")
                continue
            
            time.sleep(1)  # Prevent system overload
        
        # Generate final report
        self._generate_final_report()
        print(f"\n🎉 Optimization completed! Results saved to {self.results_dir}")
    
    def _run_single_experiment(self, config_dict: Dict, run_id: int) -> Dict:
        """Run a single experiment with the given configuration."""
        
        # Update config file
        self._update_config_file(config_dict['config'])
        
        # Update model-specific configuration if needed
        if config_dict['model_params']:
            self._update_model_config(config_dict['model'], config_dict['model_params'])
        
        # Build command
        cmd = [
            sys.executable, "main.py",
            "--model", config_dict['model'],
            "--tool", config_dict['tool'],
            "--sensor", config_dict['sensor']
        ]
        
        # Directory setup
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # Copy current environment variables and add PYTHONPATH pointing to src/
        env = os.environ.copy()
        env["PYTHONPATH"] = src_dir
        env["VIRTUAL_ENV"] = os.path.dirname(sys.executable)
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
        # Run experiment
        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8',
            cwd=src_dir, env=env, timeout=1800  # 30 minute timeout
        )
        duration = time.time() - start_time
        
        if result.returncode != 0:
            raise Exception(f"Subprocess failed: {result.stderr}")
        
        # Extract metrics from output
        metrics = self._extract_metrics_from_output(result.stdout)
        
        return {
            'run_id': run_id,
            'model': config_dict['model'],
            'tool': config_dict['tool'],
            'config': config_dict['config'],
            'model_params': config_dict['model_params'],
            'duration': duration,
            'success': True,
            **metrics
        }
    
    def _update_config_file(self, config: Dict):
        """Update the config.json file with new parameters."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _update_model_config(self, model: str, model_params: Dict):
        """Update model-specific configuration files."""
        # This would require modifications to your model files
        # For now, we'll add the parameters to the main config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        if 'model_params' not in config:
            config['model_params'] = {}
        config['model_params'][model] = model_params
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _extract_metrics_from_output(self, output: str) -> Dict:
        """Extract performance metrics from the subprocess output."""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        lines = output.split('\n')
        for line in lines:
            if '[RESULT] Accuracy:' in line:
                metrics['accuracy'] = float(line.split(':')[1].strip())
            elif '[RESULT] Precision (macro):' in line:
                metrics['precision'] = float(line.split(':')[1].strip())
            elif '[RESULT] Recall (macro):' in line:
                metrics['recall'] = float(line.split(':')[1].strip())
            elif '[RESULT] F1 Score (macro):' in line:
                metrics['f1_score'] = float(line.split(':')[1].strip())
        
        return metrics
    
    def _save_results(self):
        """Save current results to files."""
        
        # Save all results
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv(
            os.path.join(self.results_dir, 'all_results.csv'), 
            index=False
        )
        
        # Save best results
        best_results_list = []
        for key, result in self.best_results.items():
            best_results_list.append(result)
        
        if best_results_list:
            best_df = pd.DataFrame(best_results_list)
            best_df.to_csv(
                os.path.join(self.results_dir, 'best_results.csv'),
                index=False
            )
    
    def _generate_final_report(self):
        """Generate a comprehensive final report."""
        
        report_path = os.path.join(self.results_dir, 'optimization_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("HYPERPARAMETER OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write(f"Total experiments: {len(self.all_results)}\n")
            f.write(f"Successful experiments: {sum(1 for r in self.all_results if r['success'])}\n\n")
            
            # Best results by model-tool combination
            f.write("BEST RESULTS BY MODEL-TOOL COMBINATION:\n")
            f.write("-" * 40 + "\n")
            
            for key, result in self.best_results.items():
                f.write(f"\n{key}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1 Score: {result['f1_score']:.4f}\n")
                f.write(f"  Duration: {result['duration']:.2f}s\n")
                f.write(f"  Config: {json.dumps(result['config'], indent=4)}\n")
            
            # Overall best result
            if self.all_results:
                best_overall = max(self.all_results, key=lambda x: x['accuracy'])
                f.write(f"\nOVERALL BEST RESULT:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Model: {best_overall['model']}\n")
                f.write(f"Tool: {best_overall['tool']}\n")
                f.write(f"Accuracy: {best_overall['accuracy']:.4f}\n")
                f.write(f"Configuration: {json.dumps(best_overall['config'], indent=2)}\n")
        
        print(f"📊 Final report saved to {report_path}")


def main():
    """Main function to run hyperparameter optimization."""
    
    # Configuration
    models = ["lstm", "tcn"]  # Add "fcn" if you have it implemented
    tools = ["electric_screwdriver", "pneumatic_rivet_gun", "pneumatic_screwdriver"]
    max_configs_per_model = 30  # Adjust based on computational resources
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer()
    
    # Run optimization
    optimizer.run_optimization(models, tools, max_configs_per_model)


if __name__ == "__main__":
    main()