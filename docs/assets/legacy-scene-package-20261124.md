# Legacy scene asset package

The validated robotarm main scene and its source dependencies are available in
[Release `legacy-assets-20261124`](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/tag/legacy-assets-20261124).
The release assets are kept outside Git because USD and texture binaries are
large or are covered by the repository binary policy.

## Main acceptance package

Download [`legacy-robotarm-stage-20261124.zip`](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/legacy-assets-20261124/legacy-robotarm-stage-20261124.zip), verify SHA-256 `4938a10e84a6769441eeeeb3ed08d6bf7ab940a76e89e6765e0d1bd3547a015a`, and extract it at `/home/multirobo/Desktop`. The archive restores the path authored by [`assets/robotarm_magnetic_training.usda`](../../assets/robotarm_magnetic_training.usda):

`/home/multirobo/Desktop/sim of FF/Stage.usd`

The restored `Stage.usd` SHA-256 is
`0d5afec84d42989a7910899b4ef78369c454e34185c89052ab25b33fac4a6671`, matching the historical acceptance value
`0d5afec84d42989a7910899b4ef78369c454e34185c89052ab25b33fac4a6671`.
The archive contains the 19-layer USD closure, generated geometry/material
layers, ASM and robotarm source meshes, and the original-byte `robotarm.urdf`
(SHA-256 `1d6e12d461207e65ac30b4ddfcbc2d136f5cbeb9f7ac0f9e9c9852511133a613`). `ASSET_PATHS.md` in the archive
records the absolute path mapping and explains exporter paths left in comments.

After extraction, a USD 25.11+ environment can verify the closure with:

```bash
python3 - <<'PYVERIFY'
from pxr import UsdUtils
layers, assets, unresolved = UsdUtils.ComputeAllDependencies('/home/multirobo/Desktop/sim of FF/Stage.usd')
assert len(layers) == 19 and not assets and not unresolved
print('USD dependency closure: 19 layers, 0 external assets, 0 unresolved')
PYVERIFY
```

## Optional old stomach source

To reproduce the legacy stomach wrapper, download
[`legacy-stomach-source-20261124.zip`](https://github.com/xy211699-arch/robotarm-magnetic-lab/releases/download/legacy-assets-20261124/legacy-stomach-source-20261124.zip), verify SHA-256 `8565c67d2a5c126329dcfdea6a14ade70ad9e4ca43e349fe44d278833bc023e9`, and extract it at the repository root. It restores `assets/stomach/stomach_capsule_test_v1_source.usd` (source SHA-256 `072696f4550ec5acdb07fa66f64fbc9e3618df1d977836aee3c001303cad0008`) and its authored texture `assets/stomach/textures/NEW STOMACH.tga` (SHA-256 `6c21d00400044804810b26d99567d83c9eaef3b1be0ffc9a314a82dc270d246a`). The source USD has one layer and one resolved texture dependency; `ASSET_PATHS.md` records the original workstation paths.

The machine-readable archive and dependency details are in
[`asset_release_manifest.json`](../../dependencies/legacy_magnetic_sim/asset_release_manifest.json).
