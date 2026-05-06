"""Runtime bootstrap helpers for hosted deploys (Render, Fly, etc.).

Hosted platforms typically expose secrets only as environment variables,
while the Vertex SDK expects a credentials file path. This module bridges
that gap: if `GOOGLE_APPLICATION_CREDENTIALS_JSON` is set, write it to a
temp file and point `GOOGLE_APPLICATION_CREDENTIALS` at it.

No-op when the JSON env var is missing (local dev with `gcloud auth
application-default login` keeps working unchanged).
"""
from __future__ import annotations

import os
import tempfile


def setup_gcp_credentials_from_env() -> None:
    json_blob = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not json_blob:
        return
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    fd, path = tempfile.mkstemp(prefix="gcp-key-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(json_blob)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
