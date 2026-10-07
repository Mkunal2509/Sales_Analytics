import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import subprocess
import io
import os
import glob
import shutil
from datetime import datetime, timedelta

# Auto-run R package setup on first launch in cloud
try:
    if os.path.exists("setup.R"):
        subprocess.run(["Rscript", "setup.R"], check=False)
except Exception:
    pass

st.set_page_config(page_title="Retail Product & Sales Analytics", layout="wide")

# --- AUTO-LOCATE RSCRIPT (WORKS LOCALLY & ON STREAMLIT CLOUD) ---
def find_rscript_path():
    # 1. Check system PATH first (Matches Linux/Streamlit Cloud default: /usr/bin/Rscript)
    system_r = shutil.which("Rscript")
    if system_r:
        return system_r
    
    # 2. Check standard Windows installations if local
    windows_candidates = glob.glob(r"C:\Program Files\R\R-*\bin\x64\Rscript.exe") + \
                         glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
    if windows_candidates:
        return windows_candidates[-1]
    
    # 3. Check macOS/Linux defaults
    for path in ["/usr/bin/Rscript", "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript"]:
        if os.path.exists(path):
            return path
            
    return "Rscript"