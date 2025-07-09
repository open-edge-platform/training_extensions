#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test script for the OTX Benchmark Dashboard."""

import sys
import threading
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parents[2]))

from plot_benchmark import BenchmarkDashboard  # noqa: E402

from otx.types.task import OTXTaskType  # noqa: E402


def test_dashboard_basic():
    """Test basic dashboard functionality."""
    print("Testing dashboard initialization...")

    # Use default benchmark directory
    benchmark_dir = Path("tests/perf_v2/perf_history")

    if not benchmark_dir.exists():
        print(f"Warning: Benchmark directory {benchmark_dir} does not exist")
        print("Creating dashboard anyway for testing...")

    try:
        dashboard = BenchmarkDashboard(benchmark_dir)
        print("✓ Dashboard initialized successfully")
    except Exception as e:
        print(f"✗ Dashboard initialization failed: {e}")
        return False

    # Test data loading
    print("Testing data loading...")
    try:
        dashboard.load_all_benchmark_data()
        print(f"✓ Found {len(dashboard.available_versions)} versions")
        print(f"✓ Loaded data for {len(dashboard.benchmark_data)} version(s)")

        # Print available data summary
        for version, tasks in dashboard.benchmark_data.items():
            print(f"  Version {version}: {len(tasks)} tasks")
            for task_name, datasets in tasks.items():
                print(f"    {task_name}: {len(datasets)} datasets")

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

    # Test plot creation (if data available)
    if dashboard.benchmark_data:
        print("Testing plot creation...")
        try:
            # Get first available task and dataset
            first_version = next(iter(dashboard.benchmark_data.keys()))
            first_task_name = next(iter(dashboard.benchmark_data[first_version].keys()))
            first_task = OTXTaskType(first_task_name)
            first_dataset = next(iter(dashboard.benchmark_data[first_version][first_task_name].keys()))

            # Test accuracy comparison plots
            dashboard.create_accuracy_comparison_plots(first_task, first_dataset)
            print(f"✓ Created accuracy comparison plots for {first_task.value} - {first_dataset}")

            # Test latency comparison plots
            dashboard.create_latency_comparison_plots(first_task, first_dataset)
            print(f"✓ Created latency comparison plots for {first_task.value} - {first_dataset}")

            # Test training time comparison
            dashboard.create_training_time_comparison_plot(first_task, first_dataset)
            print(f"✓ Created training time comparison plot for {first_task.value} - {first_dataset}")

            # Test epoch comparison
            dashboard.create_epoch_comparison_plot(first_task, first_dataset)
            print(f"✓ Created epoch comparison plot for {first_task.value} - {first_dataset}")

            # Test iteration time comparison
            dashboard.create_iter_time_comparison_plot(first_task, first_dataset)
            print(f"✓ Created iteration time comparison plot for {first_task.value} - {first_dataset}")

            # Test scatter plot
            dashboard.create_latency_accuracy_scatter(first_task, first_dataset)
            print(f"✓ Created scatter plot for {first_task.value} - {first_dataset}")

            # Test averaged plots
            print("Testing averaged plots...")
            dashboard.create_averaged_accuracy_comparison_plots(first_task)
            print(f"✓ Created averaged accuracy comparison plots for {first_task.value}")

            dashboard.create_averaged_latency_comparison_plots(first_task)
            print(f"✓ Created averaged latency comparison plots for {first_task.value}")

            dashboard.create_averaged_training_metric_plot(
                first_task,
                "training:e2e_time_mean",
                "Training Time",
                "Training Time (seconds)",
            )
            print(f"✓ Created averaged training time comparison plot for {first_task.value}")

            dashboard.create_averaged_latency_accuracy_scatter(first_task)
            print(f"✓ Created averaged scatter plot for {first_task.value}")

        except Exception as e:
            print(f"✗ Plot creation failed: {e}")
            return False
    else:
        print("⚠ No data available for plot testing")

    print("\n✓ All tests passed!")
    return True


def test_dashboard_run():
    """Test dashboard server startup."""
    print("Testing dashboard server startup...")

    benchmark_dir = Path("perf_history")
    dashboard = BenchmarkDashboard(benchmark_dir)

    print("Starting dashboard server (will run for 5 seconds)...")
    print("Access the dashboard at: http://localhost:8050")

    # Create a timer to stop the server after 5 seconds
    def stop_server() -> None:
        time.sleep(5)
        print("\n5 seconds elapsed. Stopping server...")
        # Force exit the process (since Dash doesn't have a clean shutdown method)
        import os

        os._exit(0)

    # Start the timer thread
    timer_thread = threading.Thread(target=stop_server, daemon=True)
    timer_thread.start()

    try:
        # This would normally run indefinitely, but timer will stop it after 5 seconds
        dashboard.app.run_server(debug=True, port=8050)
    except Exception as e:
        print(f"Server startup test completed: {e}")


if __name__ == "__main__":
    print("OTX Benchmark Dashboard Test")
    print("=" * 40)

    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        test_dashboard_run()
    else:
        success = test_dashboard_basic()

        if success:
            print("\nTo run the dashboard server, use:")
            print("python plot_benchmark.py")
            print("or")
            print("python test_dashboard.py --run")
        else:
            print("\nPlease fix the issues above before running the dashboard.")
            sys.exit(1)
