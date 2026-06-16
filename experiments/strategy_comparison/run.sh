#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

echo "Starting Experiment: strategy_comparison ..."
python -m experiments.strategy_comparison.test_strategy_comparison

echo "Experiment complete."
