"""Small tmux process boundary; scientific state belongs to the project folder."""

from histopilot.workers.packing_process import TmuxScriptExecutor


class TmuxExtractionExecutor(TmuxScriptExecutor):
    """Legacy extraction launches; new extractions are Task Center tasks."""

    label = "extraction"
