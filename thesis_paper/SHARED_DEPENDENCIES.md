# Thesis paper shared dependencies

The thesis paper lives under `thesis_paper/` so it can be handed over separately. It READS the files below by explicit path (never a glob for
"the most recent" file) and imports only the `src/` modules listed under "Shared code". Every file is pinned by SHA256; text files are hashed with
CRLF converted to LF, and the `.npz` files are Git LFS objects (run `git lfs pull` in a fresh clone).

`thesis_paper/scripts/check_shared_dependencies.py` verifies this table. Run it before every thesis-paper commit. If a listed file changes,
regenerate the table with `--write` in the same commit and say why.

| Path | SHA256 |
|---|---|
| `checkpoints/saved_predictions/checkpoints/frozen_hyperparams/S.json` | `243ff66d96837bee02f0426f76bf2a2e358ad3b1c14bd1316c79c5250f2e505b` | <!-- Paper A artifact / source -->
| `checkpoints/saved_predictions/checkpoints/frozen_hyperparams/kappa.json` | `883e3d0b2e62b1569473f397f00bd5ffbd055be60a54c7e4c758c11c6d129d3c` | <!-- Paper A artifact / source -->
| `checkpoints/saved_predictions/checkpoints/frozen_hyperparams/sigma.json` | `b457b6f2f946b0ef32bb3b38f09cd2084d846d6683af6b5b91df01b9672d6f0f` | <!-- Paper A artifact / source -->
| `checkpoints/saved_predictions/checkpoints/frozen_hyperparams/zT.json` | `6b461276523817f5df3b6f9ef9b8da9758c1da95c458aa0da99b8cd7eddf44b5` | <!-- Paper A artifact / source -->
| `reports/ablation_snapfix/20260918T000111/ablation_metrics.json` | `48dd411bbb6e145a78ac27cd2e87b83d0b483b9b1b9cbe7ecf105bcc8c2e9487` | <!-- Paper A artifact / source -->
| `reports/grouping_key_remeasure/20260918T000232/grouping_key_remeasure.json` | `bd15acbd7f3f83334ef11016b82e83cf86543fd9a4ae13ad4cb06cbf3a200c29` | <!-- Paper A artifact / source -->
| `reports/insupport_share/20260924T194948Z/insupport_share.json` | `0429be78aea50542a390da72fd5ecc9cf3ade5402210310770f5ccf264efa453` | <!-- Paper A artifact / source -->
| `reports/regen_snapfix/20260917T150000/ladder_metrics.json` | `d7b95e2a5adfaf82bf5a09d2efab4d30e9d3074ecfb57834743a4fbd14ce3109` | <!-- Paper A artifact / source -->
| `reports/ungrouped_snapfix/20260922T093243/metrics.json` | `a402d186c9c4f736c3de0f2b66c1a37a6c63c78f3d318cc4e13825d892ae24f1` | <!-- Paper A artifact / source -->
| `results/cleaning_funnel/20260914T100914/funnel_counts.json` | `223496940a5f3866ecd9f16158f357f5de28a42f548c2e47d0e2bb9ffd0ee01d` | <!-- Paper A artifact / source -->
| `results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/backtransform_check_results.json` | `a6803d20e0460add55154ed30a58b18d59848ef1113e5cb1e091a17c6f426a5f` | <!-- Paper A artifact / source -->
| `results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/results.json` | `d92b9c81b47458bbb2f4bbcec6c99eae0320580dcc180fa30370335118d87ec5` | <!-- Paper A artifact / source -->
| `results/external_snapfix/20260917T160553/estm_results.json` | `fd75cbb6e5a4289277ac440f3771324d2335396c6f81bee1ba96ff03aa0a9016` | <!-- Paper A artifact / source -->
| `results/noise_floor/20260923T202312/noise_floor_inputs.json` | `1c8db90bc9634351728583390ca942273e47f098f4988d8c345857424b73b142` | <!-- Paper A artifact / source -->
| `results/raw_pull_metadata/extraction_metadata.json` | `d101d6675ceb15190a435da19783b9bbdba616caebb46009275ca1429653c1e3` | <!-- Paper A artifact / source -->
| `results/shap_attribution/20260917T134930/shap_arrays.npz` | `135baaef52ff227bda9aa9d54bb39e7e35def25bc7abab4394836e7604a34552` | <!-- Paper A artifact / source -->
| `results/shap_attribution/20260917T134930/summary.json` | `f39784ddd3056415d9252e2976d48a4f4bb5273a08c585fc71791703fda93ce3` | <!-- Paper A artifact / source -->
| `src/data_cleaning.py` | `e18715a51d81a8768142712793d3e9be948a8208272e845b575bef40e6fbf160` | <!-- shared code -->
| `src/nested_cv.py` | `ca3787ff5e36750aaa6062be3fdf5ec293f6584ca103a64a25322a2283b73254` | <!-- shared code -->
| `thesis_paper/source/Perovskite_Thermoelectric_Manuscript.docx` | `f12d66404e126de1cb7cc32d8843058988f43dc3777ac5669b17c319223fa2d4` | <!-- Paper A artifact / source -->
| `thesis_paper/results/na4/20261003T131903/results.json` | `e940b4e6d464db91c10ef3fc4fd9abbae35bb6a4345f795e6844a835f6ad29c3` | <!-- thesis analysis result -->
| `thesis_paper/results/na4/20261003T131903/run_config.json` | `725f4e38a7ee8372f1d0b2bff2065fb0bf135e91c16146547ccea524a1e82a46` | <!-- thesis analysis result -->
| `thesis_paper/results/na5/20261003T131813/results.json` | `bba7fa6b510a8fbf0068019e81a2417f3f04f9398569d2c00c599f579cd7b35e` | <!-- thesis analysis result -->
| `thesis_paper/results/na5/20261003T131813/run_config.json` | `4d14b74648fe91f09344cb658c2ca8be4b23e4df7175307225d21e0447581384` | <!-- thesis analysis result -->
| `thesis_paper/results/na8/20261003T131938/results.json` | `79367f8c1155ac9d186ef348f8aa5fcb92221691bf8c974a5161d93aa13ff45e` | <!-- thesis analysis result -->
| `thesis_paper/results/na8/20261003T131938/run_config.json` | `dc73c6000a12f4cb3dd686ebbeee6528d6019f6a693bba3084a0a018c474cf6c` | <!-- thesis analysis result -->
| `thesis_paper/results/na9/20261003T132121/results.json` | `ed89b78fbcf4cc9b2fa2b090d7ff88545f6386d0e5d3b4a4171d672b59f6b8bd` | <!-- thesis analysis result -->
| `thesis_paper/results/na9/20261003T132121/run_config.json` | `97cee3876653f4825eff94040dc40e1884ba7631bd264dbc37dec73a0a203565` | <!-- thesis analysis result -->
| `src/backtransform_check.py` | `4317955f32fbcb6c9daa3602f588945f2e236bdb906af36a655f9d3c3412512e` | <!-- shared code -->
| `src/canonicalization.py` | `4d1b427e68a72ad95ffe5588d54b76467e350451e335e97dcb13b326efb9dfb1` | <!-- shared code -->
| `src/direct_vs_derived_zt.py` | `ae418a133adccaa7a65ac4953a13da5dc1eb054b14a0d2bacfa5ca65c92f3551` | <!-- shared code -->
| `src/external_validation.py` | `f38f1ab71d48fae7e1dd3e19418718569546733d690d298656523fece34930c2` | <!-- shared code -->
| `src/featurization.py` | `c2d72008d77d6ea5853fc2e16f2bb09847b88441a8444f0ddd9141577ef34a89` | <!-- shared code -->
| `src/plotting_style.py` | `2a35193e53996aecb47c997c9dd5d98801e9b51ddd93d4bba7565154cf94224a` | <!-- shared code -->
