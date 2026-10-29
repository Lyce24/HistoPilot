"""Local-first, self-hosted PFM–MIL workbench with a dependency-free domain.

The control service owns metadata. ML and WSI adapters belong in isolated workers;
importing the top-level package never initializes a compute runtime.
"""

import logging

__version__ = "0.1.0"

# HistoPilot logs under "histopilot.*". A process that routes nothing (a CLI command, a
# test) prints none of it; the service, the runner and workers route their records with
# histopilot/observability/logs.py.
logging.getLogger(__name__).addHandler(logging.NullHandler())
