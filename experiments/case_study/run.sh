#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

echo "Starting Experiment: test_multiple_station_numbers..."
python -m experiments.case_study.test_multiple_station_numbers

echo "Plotting: test_multiple_station_numbers..."
python -m experiments.case_study.plot_multiple_station_numbers

echo "Experiment Complete."