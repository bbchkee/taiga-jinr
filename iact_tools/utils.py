
import os
import pandas as pd

COMMON_RANGES = {
    "size": (0, 3000),
    "dist[0]": (0, 6.0),
    "dist1": (0, 6.0),
    "width[0]": (0, 2.5),
    "width": (0, 2.5),
    "length[0]": (0, 2.5),
    "length": (0, 2.5),
}

def calc_seconds(directory, file_extension=".csv"):
    """Estimate run duration in seconds by counting unique values in the first column
    across CSV files (assuming each unique corresponds to a 3-minute chunk).
    """
    total_unique_sum = 0
    for filename in os.listdir(directory):
        if filename.endswith(file_extension):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            if not df.empty:
                total_unique_sum += df.iloc[:, 0].nunique()
    return int(total_unique_sum * 3 * 60)
