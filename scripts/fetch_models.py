"""Download the two face models used by the human-vs-machine benchmark.

They come from InsightFace's ``buffalo_sc`` pack (about 15 MB), which is
released for non-commercial research use only:
https://github.com/deepinsight/insightface/tree/master/model_zoo
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
DEST = Path(__file__).resolve().parent.parent / "models" / "buffalo_sc"

if __name__ == "__main__":
    if (DEST / "det_500m.onnx").exists() and (DEST / "w600k_mbf.onnx").exists():
        print(f"models already in {DEST}")
    else:
        print(f"downloading {URL} ...")
        data = urllib.request.urlopen(URL).read()
        DEST.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in ("det_500m.onnx", "w600k_mbf.onnx"):
                member = next(m for m in z.namelist() if m.endswith(name))
                (DEST / name).write_bytes(z.read(member))
        print(f"saved to {DEST}")
