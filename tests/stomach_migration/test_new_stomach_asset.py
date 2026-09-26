import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdShade


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "assets" / "stomach" / "new_stomach_v1"
USD_PATH = ASSET_DIR / "new_stomach_visual.usd"
TEXTURE_PATH = ASSET_DIR / "textures" / "newcolor.png"
MANIFEST_PATH = ASSET_DIR / "source_manifest.json"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_stomach_asset_is_project_local_portable_and_unscaled():
    assert USD_PATH.is_file()
    assert TEXTURE_PATH.is_file()
    assert MANIFEST_PATH.is_file()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["asset"]["usd_sha256"] == (
        "d89826fc0309840e567d1be6a23a6a59215e6284b265a288523b0e887d5ff58a"
    )
    assert manifest["asset"]["texture_sha256"] == (
        "615ccc1e13aa29e2a7ea972a5aed8dfc0bae1a33eddc6a14ec5549ac3c12ff2a"
    )
    assert _sha256(USD_PATH) == manifest["asset"]["usd_sha256"]
    assert _sha256(TEXTURE_PATH) == manifest["asset"]["texture_sha256"]
    assert manifest["asset"]["scale"] == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(
        manifest["geometry"]["size_m"],
        [0.208016204, 0.177894592, 0.092954700],
        atol=1.0e-4,
    )
    assert manifest["geometry"]["vertex_count"] == 214_624
    assert manifest["geometry"]["triangle_count"] == 429_029
    assert len(manifest["boundary_components"]) == 6
    assert [item["vertex_count"] for item in manifest["selected_openings"]] == [180, 180]

    stage = Usd.Stage.Open(str(USD_PATH))
    assert stage is not None
    asset_paths = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        for shader_input in UsdShade.Shader(prim).GetInputs():
            value = shader_input.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                asset_paths.append(value.path)
    assert asset_paths
    assert all(not Path(path).is_absolute() for path in asset_paths)
    assert "textures/newcolor.png" in asset_paths

