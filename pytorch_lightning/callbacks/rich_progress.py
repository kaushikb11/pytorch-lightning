# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.utilities import _RICH_AVAILABLE

if _RICH_AVAILABLE:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text


class MetricsTextColumn(TextColumn):
    """A column containing text."""

    def __init__(self, trainer):
        self._trainer = trainer
        super().__init__("")

    def render(self, task) -> Text:
        _text = ''
        if "red" in f'{task.description}':
            for k, v in self._trainer.progress_bar_dict.items():
                _text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text


class RichProgressBar(ProgressBarBase):

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__()
        self._refresh_rate = refresh_rate
        self._process_position = process_position
        self._enabled = True
        self.main_progress_bar = None
        self.val_progress_bar = None
        self.test_progress_bar = None
        self.console = Console(record=True)
        self.tasks = {}

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = self.__dict__.copy()
        state['main_progress_bar'] = None
        state['val_progress_bar'] = None
        state['test_progress_bar'] = None
        return state

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            MetricsTextColumn(trainer),
            console=self.console,
            transient=True,
        ).__enter__()
        super().on_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float('inf'):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        self.total_batches = total_train_batches + total_val_batches
        self.tasks["train"] = self.progress.add_task(
            f"[red][Epoch {trainer.current_epoch}]",
            total=self.total_batches,
        )
        if total_val_batches > 0:
            self.tasks["val"] = self.progress.add_task(
                f"[green][Epoch {trainer.current_epoch}]",
                total=total_val_batches,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches + self.total_val_batches):
            if getattr(self, "progress", None) is not None:
                self.progress.update(self.tasks["train"], advance=1.)
                self.progress.track(trainer.progress_bar_dict)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches + self.total_val_batches):
            if getattr(self, "progress", None) is not None:
                self.progress.update(self.tasks["train"], advance=1.)
                self.progress.update(self.tasks["val"], advance=1.)

    def on_train_epoch_end(self, trainer, pl_module, *_):
        super().on_train_end(trainer, pl_module)
        self.progress.__exit__(None, None, None)
        epoch_pbar_metrics = self.trainer.logger_connector.cached_results.get_epoch_pbar_metrics()
        table = Table(show_header=True, header_style="bold magenta")
        width = max([len(k) for k in epoch_pbar_metrics.keys()]) + 5
        table.add_column(f"Metrics Epoch {trainer.current_epoch} ", style="dim", width=width)
        table.add_column("Value")
        for k, v in epoch_pbar_metrics.items():
            v = round(v.item(), 4) if isinstance(v, torch.Tensor) else v
            table.add_row(k, str(v))
        self.console.log(table)

    def _should_update(self, current, total):
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    def _update_bar(self, bar):
        """ Updates the bar by the refresh rate without overshooting. """
        if bar.total is not None:
            delta = min(self.refresh_rate, bar.total - bar.n)
        else:
            # infinite / unknown size
            delta = self.refresh_rate
        if delta > 0:
            bar.update(delta)
