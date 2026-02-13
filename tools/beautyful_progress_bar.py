import atexit
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

class PBar:

    def __init__(self, total, description, preset=None):
        self.progress = None
        self.task_id = None
        self._closed = False

        match preset:
            case "training":
                self._use_training_preset(total)
            case _:
                # Default case
                self.pbar = Progress()
                self.task_id = self.pbar.add_task("", total=total)
        
        self.set_description(description)

        self.pbar.start()
        atexit.register(self.close)
        

    def update(self, val):
        if not self._closed:
            self.pbar.update(self.task_id, advance=val)
    
    def set_description(self, desc):
        if not self._closed:
            self.pbar.update(self.task_id, description=desc)

    def close(self):
        if not self._closed and self.pbar:
            self.pbar.stop()
            self._closed = True


    def _use_training_preset(self, total):
        colonne = [
            # Description
            TextColumn("{task.description}"),
            # Percentage
            TextColumn("{task.percentage:>3.0f}%"),
            # Progress bar
            BarColumn(
                style="bold #333333", 
                complete_style="bold #2b95fb", 
                finished_style="bold #852eff",
            ),
            # Task count
            TextColumn("{task.completed:,}/{task.total:,}", style="#444444"),
            # Open braket
            TextColumn("["),
            # Time elapsed
            TimeElapsedColumn(),
            # Separator
            TextColumn("<"),
            # Time remaining
            TimeRemainingColumn(),
            # Closed braket
            TextColumn("]"),
        ]
        self.pbar = Progress(*colonne)
        self.task_id = self.pbar.add_task("", total=total)
