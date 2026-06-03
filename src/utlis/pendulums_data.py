import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import detrend


@dataclass
class PendulumsData:
    """
    Data structure for holding pendulum time series data and associated metadata.
    """

    cols: list[str]
    labels: list[str]
    lengths: np.ndarray
    data: pd.DataFrame

    def drop_cols(self, cols_to_drop: list[str]):
        """
        Drop specified columns from the data and update metadata accordingly.
        Args:
            cols_to_drop: List of column names to be dropped from the data.
        """
        self.data = self.data.drop(columns=cols_to_drop)
        for col in cols_to_drop:
            index = self.cols.index(col)
            self.cols.pop(index)
            self.labels.pop(index)
            self.lengths = np.delete(self.lengths, index)

    def drop_time_range(self, t_start: float, t_end: float):
        """
        Drop rows from the data where the time 't' is within the specified range.
        Args:
            t_start: Start of the time range to be dropped (inclusive).
            t_end: End of the time range to be dropped (inclusive).
        """
        mask = (self.data["t"] < t_start) | (self.data["t"] > t_end)
        self.data = self.data[mask].reset_index(drop=True)

    def preprocess_offsets(self):
        """
        Preprocess the data by centering each column around its median value.
        This helps to remove any DC offset and focus on the oscillatory behavior.
        """
        for col in self.cols:
            self.data[col] -= self.data[col].median()

    def detrend(self):
        for col in self.cols:
            self.data[col] = detrend(self.data[col], type="linear")
