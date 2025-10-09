"""
Training orchestrator for PHAZE framework.

This module coordinates the complete training, benchmarking, and plotting pipeline
for generating real benchmark data across all model architectures and complexities.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .enhanced_crypto_benchmark import EnhancedCryptoBenchmark
from .enhanced_zkml_benchmark import EnhancedZKMLBenchmark
from .mnist_trainer import MNISTTrainer
from .multi_exit_generator import MultiExitGenerator
from .plotting.plot_suite import PHAZEPlotSuite
from .training_config import PHAZEConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Main orchestrator for PHAZE training and benchmarking pipeline."""

    def __init__(self, config: PHAZEConfig):
        """Initialize orchestrator with configuration.

        Args:
            config: PHAZE configuration object
        """
        self.config = config
        self.results = {
            'training_results': {},
            'early_exit_models': [],
            'zkml_results': [],
            'crypto_results': [],
            'timing_info': {},
            'config': config.to_dict()
        }

        # Initialize components
        self.trainer = MNISTTrainer(config)
        self.exit_generator = MultiExitGenerator(config)
        self.zkml_benchmarker = EnhancedZKMLBenchmark(config)
        self.crypto_benchmarker = EnhancedCryptoBenchmark(config)

        # Create output directories
        config.create_output_directories()

        logger.info("Training orchestrator initialized")
        logger.info(f"Output directory: {config.output.output_dir}")
        logger.info(f"Planned architectures: {config.model.architectures}")
        logger.info(f"Planned complexities: {config.model.complexities}")
        logger.info(f"Random seeds: {config.experiment.seeds}")

    def run_training_phase(self) -> Dict[str, Any]:
        """Run the training phase for all model configurations.

        Returns:
            Dictionary containing all training results
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: MODEL TRAINING")
        logger.info("=" * 60)

        phase_start_time = time.time()
        training_results = {}

        # Calculate total number of models to train
        total_models = (len(self.config.model.architectures) *
                       len(self.config.model.complexities) *
                       len(self.config.experiment.seeds))

        logger.info(f"Training {total_models} model variants...")

        # Progress tracking
        progress_bar = tqdm(total=total_models, desc="Training models")

        for architecture in self.config.model.architectures:
            for complexity in self.config.model.complexities:
                for seed in self.config.experiment.seeds:
                    try:
                        model_start_time = time.time()

                        # Create and train model
                        model = self.trainer.create_model(architecture, complexity)
                        results = self.trainer.train_model(model, architecture, complexity, seed)

                        # Save model if configured
                        if self.config.output.save_models or self.config.output.export_onnx:
                            output_dir = Path(self.config.output.output_dir) / self.config.output.models_dir
                            saved_files = self.trainer.save_model(model, results, output_dir)
                            results['saved_files'] = saved_files

                            # Export to ONNX
                            onnx_path = self.trainer.export_to_onnx(model, results, output_dir)
                            if onnx_path:
                                results['onnx_path'] = onnx_path

                        # Store model for later use
                        results['model'] = model
                        results['model_training_time'] = time.time() - model_start_time

                        # Store results
                        model_id = results['model_id']
                        training_results[model_id] = results

                        progress_bar.set_postfix({
                            'current': f"{architecture}_{complexity}",
                            'acc': f"{results['final_test_acc']:.1f}%"
                        })
                        progress_bar.update(1)

                    except Exception as e:
                        logger.error(f"Training failed for {architecture}_{complexity}_seed{seed}: {e}")
                        progress_bar.update(1)
                        continue

        progress_bar.close()

        phase_time = time.time() - phase_start_time
        successful_models = len(training_results)

        logger.info(f"Training phase completed in {phase_time:.2f}s")
        logger.info(f"Successfully trained: {successful_models}/{total_models} models")

        if successful_models == 0:
            raise RuntimeError("No models were successfully trained")

        # Update results
        self.results['training_results'] = training_results
        self.results['timing_info']['training_phase'] = phase_time

        return training_results

    def run_early_exit_generation(self, training_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate early exit models from the largest trained models.

        Args:
            training_results: Results from training phase

        Returns:
            List of early exit model information
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: EARLY EXIT MODEL GENERATION")
        logger.info("=" * 60)

        phase_start_time = time.time()

        # Select largest models from each architecture
        largest_models = self.exit_generator.select_largest_models(training_results)

        all_early_exit_models = []

        for architecture, model_info in largest_models.items():
            try:
                logger.info(f"Generating early exit variants for {architecture}")

                model = model_info['model']
                early_variants = self.exit_generator.generate_early_exit_variants(
                    model, architecture, model_info['complexity'], model_info
                )

                # Save early exit models
                if self.config.output.save_models:
                    output_dir = Path(self.config.output.output_dir) / self.config.output.models_dir / "early_exit"

                    for early_info in early_variants:
                        saved_files = self.exit_generator.save_early_exit_model(early_info, output_dir)
                        early_info['saved_files'] = saved_files

                all_early_exit_models.extend(early_variants)

                logger.info(f"Generated {len(early_variants)} early exit variants for {architecture}")

            except Exception as e:
                logger.error(f"Early exit generation failed for {architecture}: {e}")
                continue

        phase_time = time.time() - phase_start_time

        logger.info(f"Early exit generation completed in {phase_time:.2f}s")
        logger.info(f"Generated {len(all_early_exit_models)} early exit models total")

        # Update results
        self.results['early_exit_models'] = all_early_exit_models
        self.results['timing_info']['early_exit_phase'] = phase_time

        return all_early_exit_models

    async def run_zkml_benchmarking(self, training_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run zkML benchmarking on trained models.

        Args:
            training_results: Results from training phase

        Returns:
            List of zkML benchmark results
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: ZKML BENCHMARKING")
        logger.info("=" * 60)

        phase_start_time = time.time()

        # Prepare models for benchmarking (limit to avoid overwhelming the system)
        models_to_benchmark = []

        # Select representative models: one from each architecture-complexity combination
        arch_complexity_combinations = set()
        for model_info in training_results.values():
            arch_complexity = (model_info['architecture'], model_info['complexity'])
            if arch_complexity not in arch_complexity_combinations:
                arch_complexity_combinations.add(arch_complexity)
                models_to_benchmark.append(model_info)

        logger.info(f"Benchmarking {len(models_to_benchmark)} representative models")

        # Run zkML benchmarks
        zkml_results = await self.zkml_benchmarker.benchmark_multiple_models(
            models_to_benchmark,
            self.config.experiment.zkml_frameworks
        )

        phase_time = time.time() - phase_start_time

        successful_benchmarks = len([r for r in zkml_results if r.get('success', False)])

        logger.info(f"zkML benchmarking completed in {phase_time:.2f}s")
        logger.info(f"Successful benchmarks: {successful_benchmarks}/{len(zkml_results)}")

        # Update results
        self.results['zkml_results'] = zkml_results
        self.results['timing_info']['zkml_phase'] = phase_time

        return zkml_results

    def run_crypto_benchmarking(self, early_exit_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run crypto benchmarking on early exit models.

        Args:
            early_exit_models: List of early exit model information

        Returns:
            List of crypto benchmark results
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: CRYPTO BENCHMARKING")
        logger.info("=" * 60)

        phase_start_time = time.time()

        # Run crypto benchmarks on early exit models
        crypto_results = self.crypto_benchmarker.benchmark_multiple_models(early_exit_models)

        phase_time = time.time() - phase_start_time

        successful_benchmarks = len([r for r in crypto_results if r.get('success', False)])

        logger.info(f"Crypto benchmarking completed in {phase_time:.2f}s")
        logger.info(f"Successful benchmarks: {successful_benchmarks}/{len(crypto_results)}")

        # Update results
        self.results['crypto_results'] = crypto_results
        self.results['timing_info']['crypto_phase'] = phase_time

        return crypto_results

    def generate_plots(self) -> Dict[str, Any]:
        """Generate plots from benchmark results.

        Returns:
            Dictionary containing plot generation results
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: PLOT GENERATION")
        logger.info("=" * 60)

        phase_start_time = time.time()

        if not self.config.output.generate_plots:
            logger.info("Plot generation disabled in configuration")
            return {}

        try:
            # Initialize plot suite
            plot_suite = PHAZEPlotSuite(config=self.config.plotting)

            # Prepare data for plotting
            plot_data = {
                'zkml_results': self.results['zkml_results'],
                'crypto_results': self.results['crypto_results'],
                'training_results': self.results['training_results'],
                'early_exit_models': self.results['early_exit_models']
            }

            # Generate plots
            output_dir = Path(self.config.output.output_dir) / self.config.output.plots_dir
            plot_results = plot_suite.generate_plots(
                data=plot_data,
                output_dir=output_dir,
                save_plots=self.config.output.save_plots
            )

            # Generate summary report
            if plot_results:
                report_path = output_dir / "plotting_summary.md"
                plot_suite.generate_summary_report(plot_results, report_path)
                logger.info(f"Plot summary report saved to {report_path}")

            phase_time = time.time() - phase_start_time

            logger.info(f"Plot generation completed in {phase_time:.2f}s")

            if plot_results and 'plots' in plot_results:
                num_plot_types = len(plot_results['plots'])
                total_plots = sum(p.get('num_plots', 0) for p in plot_results['plots'].values())
                logger.info(f"Generated {total_plots} plots across {num_plot_types} plot types")

            # Update results
            self.results['plot_results'] = plot_results
            self.results['timing_info']['plotting_phase'] = phase_time

            return plot_results

        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return {'error': str(e)}

    def save_final_results(self) -> str:
        """Save complete results to file.

        Returns:
            Path to saved results file
        """
        output_dir = Path(self.config.output.output_dir)
        results_file = output_dir / "complete_benchmark_results.json"

        # Prepare results for JSON serialization (remove model objects)
        serializable_results = {}

        for key, value in self.results.items():
            if key == 'training_results':
                # Remove model objects from training results
                serializable_results[key] = {}
                for model_id, model_info in value.items():
                    clean_info = {k: v for k, v in model_info.items() if k != 'model'}
                    serializable_results[key][model_id] = clean_info

            elif key == 'early_exit_models':
                # Remove model objects from early exit models
                serializable_results[key] = []
                for model_info in value:
                    clean_info = {k: v for k, v in model_info.items() if k != 'model'}
                    serializable_results[key].append(clean_info)

            else:
                serializable_results[key] = value

        # Add timestamp and summary
        serializable_results['completion_timestamp'] = time.time()
        serializable_results['total_runtime'] = sum(
            self.results['timing_info'].values()
        )

        # Save to file
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Complete results saved to {results_file}")
        return str(results_file)

    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete PHAZE training and benchmarking pipeline.

        Returns:
            Dictionary containing all results
        """
        pipeline_start_time = time.time()

        logger.info("🚀 Starting PHAZE complete pipeline")
        logger.info(f"Configuration: {self.config.project_name} v{self.config.version}")

        try:
            # Phase 1: Train models
            training_results = self.run_training_phase()

            # Phase 2: Generate early exit models
            early_exit_models = self.run_early_exit_generation(training_results)

            # Phase 3: zkML benchmarking
            zkml_results = await self.run_zkml_benchmarking(training_results)

            # Phase 4: Crypto benchmarking
            crypto_results = self.run_crypto_benchmarking(early_exit_models)

            # Phase 5: Generate plots
            self.generate_plots()

            # Save complete results
            results_file = self.save_final_results()

            total_time = time.time() - pipeline_start_time

            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total runtime: {total_time:.2f}s ({total_time/3600:.2f} hours)")
            logger.info(f"Models trained: {len(training_results)}")
            logger.info(f"Early exit models: {len(early_exit_models)}")
            logger.info(f"zkML benchmarks: {len([r for r in zkml_results if r.get('success')])}")
            logger.info(f"Crypto benchmarks: {len([r for r in crypto_results if r.get('success')])}")
            logger.info(f"Results saved to: {results_file}")

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

            # Save partial results
            partial_results_file = self.save_final_results()
            logger.info(f"Partial results saved to: {partial_results_file}")

            raise

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary.

        Returns:
            Dictionary with progress information
        """
        summary = {
            'trained_models': len(self.results['training_results']),
            'early_exit_models': len(self.results['early_exit_models']),
            'zkml_benchmarks': len(self.results['zkml_results']),
            'crypto_benchmarks': len(self.results['crypto_results']),
            'timing_info': self.results['timing_info']
        }

        return summary


# Convenience functions
async def run_full_pipeline(config: PHAZEConfig) -> Dict[str, Any]:
    """Run the complete PHAZE pipeline with given configuration.

    Args:
        config: PHAZE configuration

    Returns:
        Complete results dictionary
    """
    orchestrator = TrainingOrchestrator(config)
    return await orchestrator.run_complete_pipeline()


def run_training_only(config: PHAZEConfig) -> Dict[str, Any]:
    """Run only the training phase.

    Args:
        config: PHAZE configuration

    Returns:
        Training results dictionary
    """
    orchestrator = TrainingOrchestrator(config)
    return orchestrator.run_training_phase()


if __name__ == "__main__":
    # Example usage
    from .training_config import create_default_config

    async def main():
        # Create configuration
        config = create_default_config()

        # Reduce for quick testing
        config.training.epochs = 1
        config.dataset.dataset_size = 500
        config.experiment.benchmark_iterations = 3
        config.model.architectures = ["simple", "conv"]  # Test subset
        config.model.complexities = ["minimal", "light"]  # Test subset
        config.experiment.seeds = [42, 123]  # Fewer seeds

        # Run pipeline
        try:
            await run_full_pipeline(config)
            print("✅ Pipeline completed successfully!")

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")

    asyncio.run(main())
