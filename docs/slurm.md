### Slurm
The framework supports submitting jobs to a cluster managed by Slurm using [submitit](https://github.com/facebookincubator/submitit).
This allows you to access a lot of Slurm functionality through Python directly.
Some things to keep in mind:

- `submitit` uses the currently activated Python environment for everything. Make sure your dependencies are installed and the Python program that submits the jobs runs in the environment you want your jobs to run in as well.

#### VSC-specific notes
- The included examples (`debug`, `genius` and `wice`) are configs for [VSC](https://docs.vscentrum.be/index.html) partitions. The `cpus_per_gpu` counts are not arbitrary! These are the CPUs required to request a *single* GPU. If you want to use more, make sure to keep this in mind: `base_cpus_per_gpu * gpus_per_node = cpus_per_gpu` where `base_cpus_per_gpu` is the number in the examples. So three GPUs on `wice` will be `18 * 3 = 54`.
- The `debug` partition has a max time of 30 min.
- To view all your jobs, run: `squeue -u $USER --clusters=wice,genius`
