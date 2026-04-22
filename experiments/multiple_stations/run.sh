#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# echo "Starting Experiment: test_multiple_station_numbers..."
# python -m experiments.multiple_stations.test_multiple_station_numbers

echo "Plotting: test_multiple_station_numbers..."
python -m experiments.multiple_stations.plot_multiple_station_numbers

echo "Experiment Complete."