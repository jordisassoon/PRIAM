import pandas as pd
import csv
from io import StringIO


def read_csv_auto_delimiter(uploaded_file, encoding_list=None):
    if encoding_list is None:
        encoding_list = ["utf-8", "latin1", "cp1252"]

    # Reset file pointer in case it's already read
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    # Read file content
    try:
        content_bytes = uploaded_file.read()
        if isinstance(content_bytes, str):
            content = content_bytes
        else:
            # Try decoding with available encodings
            for enc in encoding_list:
                try:
                    content = content_bytes.decode(enc)
                    break
                except Exception:
                    continue
            else:
                raise ValueError("Could not decode the uploaded file with any tried encoding.")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    if not content.strip():
        raise ValueError("The uploaded CSV file is empty.")

    # Try detecting delimiter
    try:
        dialect = csv.Sniffer().sniff(content, delimiters=[",", ";", "\t", "|", ":"])
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","  # fallback

    # Read CSV into DataFrame
    for enc in encoding_list:
        try:
            df = pd.read_csv(
                StringIO(content),
                delimiter=delimiter,
                encoding=enc,
                engine="python",
            )
            if df.shape[1] > 0:  # at least one column
                return df
        except pd.errors.EmptyDataError:
            continue
        except Exception:
            continue

    raise ValueError("Could not parse CSV. Check file content, encoding, or delimiter.")
