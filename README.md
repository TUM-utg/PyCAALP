# PyCAALP (Computer-Aided Assembly Line Planning)

A Unified Framework for Automated Assembly Sequence and Production Line Planning using Graph-based Optimization

## About

__PyCAALP__ is an open-source, Python-based framework designed to solve the Assembly Sequence Planning (ASP) and Production Line Planning (PLP) problems simultaneously.

Traditionally, product design and production planning are treated as separate sequential stages.
__PyCAALP__ bridges this gap by transforming the physical assembly definition into a graph-based optimization problem.
It uses a Mixed-Integer Programming (MIP) formulation to balance manufacturing station times while respecting critical engineering constraints.

This framework is capable of handling complex industrial assemblies by integrating kinematic boundary conditions (collision detection) and applying heuristic reduction techniques to ensure scalability.

## Features

__Unified Optimization__: Solves ASP and PLP in a single workflow using a weighted directed graph approach.

__Geometric Feasibility__: Built-in collision detection using Degree-of-Freedom (DoF) matrices to prune infeasible sequences automatically.

__Constraint Management__: Supports Engineering constraints (handling fragility, tolerance accumulation, technology switching - engineering factor $\mu$).

__Scalable__: Includes a randomized edge-reduction algorithm and shortest-path heuristics to handle complex assemblies with massive solution spaces.

__Customizable__: Flexible trade-off between sequence quality and assembly line efficiency via the time-balancing factor ($\lambda$)

## Installation

### Requirements

__python__ `>= 3.6`

Libraries:

* __numpy__ `==1.26.4`
* __networkx__ `==3.4.2`
* __matplotlib__ `==3.8.4`
* __Pillow__ `==10.3.0`
* __pandas__ `==2.2.2`
* __openpyxl__ `==3.1.2`
* __loguru__ `==0.7.2`
* __PySCIPOpt__ `==5.0.1`

### Setup

```bash
# Clone the repository
git clone https://github.com/TUM-utg/PyCAALP.git
cd PyCAALP

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> A PyPI package release is planned for the near future.

## Usage

```python
from pycaalp.run import create_assembly_digraph, optimize

FILE_NAME = "data/assembly_1/assembly_1_parts.json"

# engineering constraint constants (mu)
MU_TECH = 1.0
MU_HAND = 0.0
MU_TOL = 0.0

# MIP options
NUM_PHASES = 3
LAMBDA_TIME_BAL = 0.5  # time balancing coefficient

# Generate the weighted assembly directed graph
assembly_digraph = create_assembly_digraph(
        file_name=FILE_NAME,
        w_tech=MU_TECH,
        w_hand=MU_HAND,
        w_tol=MU_TOL,
        # dmf_file=DFM_FILE # if provided
    )

# Solve the phase time balancing problem
result, best_path = optimize(
    assembly_digraph=assembly_digraph,
    num_phases=NUM_PHASES,
    w_balanced=LAMBDA_TIME_BAL,
    full_result_output=True, 
    hide_output=False, # Print MIPs output
)

```

`result` consists of all the solution's attributes:  _'operations', 'technology', 'handling', 'tolerance', 'time', 'absolute_handling', 'absolute_tolerance', 'absolute_time', 'phase', 'operations_per_phase', 'time_per_phase', 'absolute_time_per_phase', 'alpha'_

`best path` is a list consisting of `num_phases` lists. Each list consists of the joints/operations that take place at the current phase, as part tuples, i.e., `[[(partA, partB), (partB, partC)], [...], ... ]`

## Examples

* `examples/run_assembly_1.py`: This example uses Assembly 1 (14 parts, 13 joints) to show a simple assembly sequence calculation with balanced phase times.  

## API Reference

## Citation

```bash
@misc{hartmann2025unifiedframeworkautomatedassembly,
      title={A Unified Framework for Automated Assembly Sequence and Production Line Planning using Graph-based Optimization}, 
      author={Christoph Hartmann and Marios Demetriades and Kevin Prüfer and Zichen Zhang and Klaus Spindler and Stefan Weltge},
      year={2025},
      eprint={2512.13219},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.13219}, 
}
```

## License

This project is released under the MIT License.  
Copyright (c) 2025  
Chair of Metal Forming and Casting (UTG),
Technical University of Munich (TUM).

## Contact

__Project Maintainer:__ Marios Demetriades [@MariosDem](https://github.com/MariosDem)

* __Bug Reports & Feature Requests:__ Please open an issue on the [GitHub Issue Tracker](https://github.com/TUM-utg/PyCAALP/issues).

__Research Coordinator:__ Dr.-Ing. Christoph Hartmann

* __Institution:__ Chair of Metal Forming and Casting, Technical University of Munich
* __Research & Collaboration Inquiries:__ Email [christoph.hartmann@utg.de](mailto:christoph.hartmann@utg.de).
