"""Compatibility launcher; AppLauncher owns --device and --headless arguments."""

from pathlib import Path


path = Path(__file__).with_name("probe_xiaohai_reference_pd.py")
source = path.read_text(encoding="utf-8")
source = source.replace('parser.add_argument("--device", default="cuda:0")\n', "")
source = source.replace('parser.add_argument("--headless", action="store_true")\n', "")
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
