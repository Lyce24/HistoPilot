"""Records owned by HistoPilot, independent of UI and backend frameworks.

These frozen records describe the intended contracts. They do not yet validate
cross-record references, persist datasets, or enforce clinical/experiment audits.
"""

from .cohort import Cohort
from .dataset import Block, DatasetVersion, Patient, Slide, Specimen
from .experiment import Experiment, Result, Run
from .features import FeatureSet
from .project import Project
from .split import Split, SplitAssignment

__all__ = [
    "Block",
    "Cohort",
    "DatasetVersion",
    "Experiment",
    "FeatureSet",
    "Patient",
    "Project",
    "Result",
    "Run",
    "Slide",
    "Specimen",
    "Split",
    "SplitAssignment",
]
