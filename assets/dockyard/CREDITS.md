# Dockyard scene assets

All downloaded assets are from [Poly Haven](https://polyhaven.com), license **CC0** (public domain). Models converted from glTF (1k) to OBJ + PNG by `scripts/fetch_dockyard_assets.py`.

| asset | type | used as |
|---|---|---|
| [concrete_floor_worn_001](https://polyhaven.com/a/concrete_floor_worn_001) | texture | quay ground |
| [rusty_painted_metal](https://polyhaven.com/a/rusty_painted_metal) | texture | ship hull |
| [metal_plate_02](https://polyhaven.com/a/metal_plate_02) | texture | ship deck / gangway |
| [metal_grate_rusty](https://polyhaven.com/a/metal_grate_rusty) | texture | stair treads / landings |
| [painted_concrete](https://polyhaven.com/a/painted_concrete) | texture | quay wall |
| [rusty_corrugated_iron](https://polyhaven.com/a/rusty_corrugated_iron) | texture | containers / deck house |
| [boulder_01](https://polyhaven.com/a/boulder_01) | model | stone |
| [rock_07](https://polyhaven.com/a/rock_07) | model | stone |
| [rock_09](https://polyhaven.com/a/rock_09) | model | stone |
| [namaqualand_boulder_02](https://polyhaven.com/a/namaqualand_boulder_02) | model | stone |
| [wooden_crate_01](https://polyhaven.com/a/wooden_crate_01) | model | dock clutter |
| [barrel_03](https://polyhaven.com/a/barrel_03) | model | dock clutter |
| [stone_01](https://polyhaven.com/a/stone_01) | model | stone |

Made procedurally in `sim/scenes/dockyard.py` (no free modern steel ship model was found): ship hull (curved bilge, flared steel sides), deck, deck house, deck containers, keel blocks, 8-step stair + scaffold platform, bumpy yard heightfield. Textures on these are the Poly Haven ones above.

## Ship
| asset | source | license | used as |
|---|---|---|---|
| MV Spartan (Clyde puffer, photogrammetry) | [Scottish Maritime Museum on Sketchfab](https://sketchfab.com/3d-models/mv-spartan-e2c3ced464f14e3b864f15871bf6d87d) | CC0-1.0 | the ship (default), converted by `scripts/convert_spartan.py` to `ship_spartan/` |
| Bulk carrier (`cargo-ship2.blend`, Blender 2.63 file, 2013) | provided by the project owner (cargo-ship-3d-model.zip) | **not stated in the download, to be confirmed** | the default ship, converted by `scripts/convert_cargo_ship.py` to `ship_cargo/` |
