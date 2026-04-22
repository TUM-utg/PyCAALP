# Check attribute changes

## Technology, Handling, Tolerance, Time

This test uses the assembly 1 part while varying the `w_tech`, `w_hand`, `w_tol`, and `w_mass` constants.
We want to confirm that not favoring technology changes, i.e., $w\_tech \approx 1$ will result in less technology changes. Similarly, higher handling and tolerance weights should ensure that accumulated attribute plot areas are minimized.

## Basic run

Just run `python -m experiments.multiple_attributes.run` for the basic results run.

## Extra tests

* For further example tests:
    Create a new test case folder e.g., multiple_attributes/test_assembly_X
    Create a config file based on multiple_attributes/template_config.py and save it in the new folder
    Then run:
    `$ python -m experiments.multiple_attributes.run --config-file <new-config-path-filename>`

* Only plots run:
    For occasions where the results are already generated and only the plots are needed.
    `$ python -m experiments.multiple_attributes.run --config-file <config-path-filename> --only-plots`

## Results

Results and plots can be found inside the results directory specified in the config file: 

* `<result-dirname>/results`
* `<result-dirname>/plots`
