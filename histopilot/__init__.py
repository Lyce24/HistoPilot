"""Local-first, self-hosted PFM–MIL workbench with a dependency-free domain.

The control service owns metadata. ML and WSI adapters belong in isolated workers;
importing the top-level package never initializes a compute runtime.
"""

__version__ = "0.1.0.dev0"
