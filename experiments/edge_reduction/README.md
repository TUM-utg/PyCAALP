# Check edge reduction effect on solution quality

## Basic run

Just run `python -m edge_reduction.run` for the basic results run.

## Extra tests

* For further example tests:
    Create a new test case folder e.g., edge_reduction/test_assembly_X
    Create a config file based on edge_reduction/template_config.py and save it in the new folder
    Then run:
    `$ python -m experiments.edge_reduction.run --config-file <new-config-path-filename>`

* Only plots run:
    For occasions where the results are already generated and only the plots are needed.
    `$ python -m experiments.edge_reduction.run --config-file <config-path-filename> --only-plots`

## Results

Results and plots can be found inside the results directory specified in the config file:

* `<result-dirname>/results`
* `<result-dirname>/plots`
