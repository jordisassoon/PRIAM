import pandas as pd
import csv
from io import StringIO

def read_csv_auto_delimiter(uploaded_file, encoding_list=None):
    if encoding_list is None:
        encoding_list = ["utf-8", "latin1", "cp1252"]
    
    # Read file content into a string (needed for Sniffer)
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        content = uploaded_file.read()
        if isinstance(content, bytes):
            # Try to decode using first working encoding
            for enc in encoding_list:
                try:
                    content = content.decode(enc)
                    break
                except Exception:
                    continue
    else:
        with open(uploaded_file, "r", encoding=encoding_list[0], errors="ignore") as f:
            content = f.read()
    
    # Try to detect delimiter using csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(content, delimiters=[",", ";", "\t", "|", ":"])
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","  # fallback

    # Try reading with multiple encodings
    for enc in encoding_list:
        try:
            df = pd.read_csv(StringIO(content), delimiter=delimiter, encoding=enc, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    raise ValueError("Could not read CSV automatically. Check encoding or delimiter.")
