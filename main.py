from pathlib import Path
import pandas as pd

try:
    from tc_python import TCPython
except ImportError:
    raise ImportError(
        "tc_python must be installed seperately via ThermoCalc license."
        "Please contact ThermoCalc for licensing information."
    )
from process_data import process_data

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path('results') / timestamp
print(output_dir)
# output_dir.mkdir(parents=True, exist_ok=True)

process_data()

print("Starting ThermoCalc session...")
with TCPython() as session:
    print("Calculating indexes for all alloys...")

