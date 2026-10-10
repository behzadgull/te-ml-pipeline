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
| `reports/regen_snapfix/20260917T150000/ladder_metrics.json` | `d7b95e2a5adfaf82bf5a09d2efab4d30e9d3074ecfb57834743a4fbd14ce3109` | <!-- Paper A artifact / source -->
| `reports/ungrouped_snapfix/20260922T093243/metrics.json` | `a402d186c9c4f736c3de0f2b66c1a37a6c63c78f3d318cc4e13825d892ae24f1` | <!-- Paper A artifact / source -->
| `results/cleaning_funnel/20260914T100914/funnel_counts.json` | `223496940a5f3866ecd9f16158f357f5de28a42f548c2e47d0e2bb9ffd0ee01d` | <!-- Paper A artifact / source -->
| `results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/backtransform_check_results.json` | `a6803d20e0460add55154ed30a58b18d59848ef1113e5cb1e091a17c6f426a5f` | <!-- Paper A artifact / source -->
| `results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/results.json` | `d92b9c81b47458bbb2f4bbcec6c99eae0320580dcc180fa30370335118d87ec5` | <!-- Paper A artifact / source -->
| `results/noise_floor/20260923T202312/noise_floor_inputs.json` | `1c8db90bc9634351728583390ca942273e47f098f4988d8c345857424b73b142` | <!-- Paper A artifact / source -->
| `results/raw_pull_metadata/extraction_metadata.json` | `d101d6675ceb15190a435da19783b9bbdba616caebb46009275ca1429653c1e3` | <!-- Paper A artifact / source -->
| `results/shap_attribution/20260917T134930/shap_arrays.npz` | `135baaef52ff227bda9aa9d54bb39e7e35def25bc7abab4394836e7604a34552` | <!-- Paper A artifact / source -->
| `results/shap_attribution/20260917T134930/summary.json` | `f39784ddd3056415d9252e2976d48a4f4bb5273a08c585fc71791703fda93ce3` | <!-- Paper A artifact / source -->
| `src/data_cleaning.py` | `e18715a51d81a8768142712793d3e9be948a8208272e845b575bef40e6fbf160` | <!-- shared code -->
| `src/nested_cv.py` | `ca3787ff5e36750aaa6062be3fdf5ec293f6584ca103a64a25322a2283b73254` | <!-- shared code -->
| `thesis_paper/results/external_rescore/20261010T124804/estm_results.json` | `f75d63799e7aeb89a4039e13067720867f4a28680a4e3b4a569c1dcdfebdeee4` | <!-- thesis analysis result -->
| `thesis_paper/results/external_rescore/20261010T124804/insupport_share.json` | `6d51acb9c0ebdfc05a51583cbc74b010f3eacf6992c38d9d3980793704c9130f` | <!-- thesis analysis result -->
| `thesis_paper/source/Perovskite_Thermoelectric_Manuscript.docx` | `f12d66404e126de1cb7cc32d8843058988f43dc3777ac5669b17c319223fa2d4` | <!-- Paper A artifact / source -->
| `thesis_paper/results/na4/20261003T131903/results.json` | `e940b4e6d464db91c10ef3fc4fd9abbae35bb6a4345f795e6844a835f6ad29c3` | <!-- thesis analysis result -->
| `thesis_paper/results/na4/20261003T131903/run_config.json` | `725f4e38a7ee8372f1d0b2bff2065fb0bf135e91c16146547ccea524a1e82a46` | <!-- thesis analysis result -->
| `thesis_paper/results/na5/20261003T131813/results.json` | `bba7fa6b510a8fbf0068019e81a2417f3f04f9398569d2c00c599f579cd7b35e` | <!-- thesis analysis result -->
| `thesis_paper/results/na5/20261003T131813/run_config.json` | `4d14b74648fe91f09344cb658c2ca8be4b23e4df7175307225d21e0447581384` | <!-- thesis analysis result -->
| `thesis_paper/results/na8/20261003T131938/results.json` | `79367f8c1155ac9d186ef348f8aa5fcb92221691bf8c974a5161d93aa13ff45e` | <!-- thesis analysis result -->
| `thesis_paper/results/na8/20261003T131938/run_config.json` | `dc73c6000a12f4cb3dd686ebbeee6528d6019f6a693bba3084a0a018c474cf6c` | <!-- thesis analysis result -->
| `thesis_paper/results/na9/20261003T132121/results.json` | `ed89b78fbcf4cc9b2fa2b090d7ff88545f6386d0e5d3b4a4171d672b59f6b8bd` | <!-- thesis analysis result -->
| `thesis_paper/results/na9/20261003T132121/run_config.json` | `97cee3876653f4825eff94040dc40e1884ba7631bd264dbc37dec73a0a203565` | <!-- thesis analysis result -->
| `thesis_paper/results/na7/20261004T093502/results.json` | `2a9c980ca2d6591ad3caf2e16e1cb2814a56679b27ff34e46bbe9903ab2bfcd1` | <!-- thesis analysis result -->
| `thesis_paper/results/na7/20261004T093502/status.json` | `bf52e4163df1649fb5872347dd1c4112afd8873a42928c365d4598ae3689e837` | <!-- thesis analysis result -->
| `thesis_paper/results/na7/20261004T093502/manifest.json` | `0dbd8948b59c8bdce1508c83a7bdee5a795b006a7077c4d938bbd964b28646fc` | <!-- thesis analysis result -->
| `results/external_snapfix/20260917T160553/tematdb_inventory_snapfix.json` | `83ec3e83b5dc1922cd3a7821c973f997ef1480a73095aa0dc2e025ca40f343cd` | <!-- Paper A artifact / source -->
| `results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json` | `54b47a52fd959bce9f8856e3e87708c7722d5ae7fc040770b4851278fb39a11d` | <!-- Paper A artifact / source -->
| `results/external_snapfix/20260917T160553/estm_results.json` | `fd75cbb6e5a4289277ac440f3771324d2335396c6f81bee1ba96ff03aa0a9016` | <!-- Paper A artifact / source -->
| `thesis_paper/results/na6_metrics/20261004T135638/metrics.json` | `c7a89dd286a0c6964447f096ea5976e2d675df79ac543058c8b13631798d89f6` | <!-- thesis analysis result -->
| `thesis_paper/results/na6_metrics/20261004T135638/run_config.json` | `b1b2cf751723d920f738d957a2dac26740823622885a032b52d7e57b62f9aa8f` | <!-- thesis analysis result -->
| `thesis_paper/results/na6_sign_override/20261004T194811_same_folds/sign_comparison_same_folds.json` | `8ec43a028ed7c4035c748b6e940d29602a8e6020e447aba7347e540a38f6b9df` | <!-- thesis analysis result -->
| `thesis_paper/results/na6_sign_override/20261004T194811_same_folds/run_config.json` | `9c5e67c27f90609922213170a1381fadb6a9b7d344f629d2df88e2c6687f61e5` | <!-- thesis analysis result -->
| `thesis_paper/results/na10_analysis/20261004T190444/analysis.json` | `ffe5f9d085ec11d2ed5d91548451f1410511082f9f8f2cc0caa2e025ac948cc6` | <!-- thesis analysis result -->
| `thesis_paper/results/na10_analysis/20261004T190444/run_config.json` | `95b7cb4ef93535e01e5dea8fe7ee9e45cd80fa80e7b8ac9c34990157ee2fda41` | <!-- thesis analysis result -->
| `thesis_paper/results/na10_analysis/20261005T054659/analysis.json` | `a0b48cf58fc4cf3d6116fee5208d460d4b8ac02e3f392c4e19e5dd4d0f2d1563` | <!-- thesis analysis result -->
| `thesis_paper/results/na10_analysis/20261005T054659/run_config.json` | `590fdb425c5569c0e67c2a2015d6d0158d09d3f54194c15fc16727f1a45a9e0e` | <!-- thesis analysis result -->
| `thesis_paper/results/na11/20261004T135055/counts.json` | `e9c26ddf5788fe673006c17a072ffd04982d90b72ec25ac6894c6815b0851acf` | <!-- thesis analysis result -->
| `thesis_paper/results/na11/20261004T135055/run_config.json` | `39b6c156436b662d72e4c706bb06c268b0639e8deb6e6146f2f718da3b7c5d84` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_sensitivity/20261004T135205/counts.json` | `08d78804c8cb16b3734a731ecbbd1e157d14f95e4865e44f0013163a7f302bd7` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_sensitivity/20261004T135205/run_config.json` | `840a7e7d6c1451abe1658f386e06f6c8717b5d0f8921c0f2b222114586676b92` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_candidate_novelty/20261004T185101/counts.json` | `9e529950a650d73914607455ff99cdc69d77c249ae4a0f93f8098b9deef4e0df` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_candidate_novelty/20261004T185101/run_config.json` | `cb1309823cacb9d70b24871286aef36eca83fc3a8596a5223208038a7deedfd2` | <!-- thesis analysis result -->
| `thesis_paper/results/na2_random_forest_tuning/summary/tuning_summary.json` | `26862ec4aca385dc58b48c6a1a71081d80d5eb7ea63391894a710cbb7311e2fe` | <!-- thesis analysis result -->
| `thesis_paper/results/na2_random_forest_tuning/summary/run_config.json` | `487302182489440337d206c9347ac021c330bdba4abd46d90cde5d1c4a2d5157` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_shortlist_limits/20261005T060020/shortlist_limits.csv` | `b998949e1d25f423e8cd19246ab081e7092f142d2fea00309d33dd73d4a57d37` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_shortlist_limits/20261005T060020/summary.json` | `735cf1d68435bcd78cbce37ee27a025f62443723f8673f31b064a76c3e2ffd77` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_shortlist_limits/20261005T060020/run_config.json` | `81ddbff5dc19ba1c52872ff4c227f22f0ef4396f9c0e09c19bca968ac840c486` | <!-- thesis analysis result -->
| `thesis_paper/docs/shortlist_thermal_limits.csv` | `c8b9927a8f17ed4cd38f5aae53cde43e8259e7810b50f92781275d5fb2e204d1` | <!-- Paper A artifact / source -->
| `thesis_paper/results/na2_comparison/20261009T190259/comparison.json` | `5d7f967285f62b32840420f0c675c4e15987bf69a51c5e27c3521b9b3454b65f` | <!-- thesis analysis result -->
| `thesis_paper/results/na2_comparison/20261009T190259/run_config.json` | `025ed1d6d4874ebb5a10b9260a914939b96e40e754310d330bc08ad000031602` | <!-- thesis analysis result -->
| `thesis_paper/results/na1_comparison/20261006T050237/comparison.json` | `44fd28b348a5c88d65cf2e60af66c3a3c33e7a127d9ec9c06df55d8c2b458f60` | <!-- thesis analysis result -->
| `thesis_paper/results/na1_comparison/20261006T050237/run_config.json` | `17c43a38a39e8a7774834a0a9622882571c5417a86b18c7a378b9630ca2f4752` | <!-- thesis analysis result -->
| `thesis_paper/results/na2_stack_weights/20261010T103738/summary.json` | `37c470426a03cd83150479b5072c2b760ef4246822676e118791ae2d2ddd7405` | <!-- thesis analysis result -->
| `thesis_paper/results/na2_stack_weights/20261010T103738/run_config.json` | `26aa25100c1ad70ae3c951abd61f5629d3f8a3b86b0bc4690105213945ada5c6` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_analysis/20261009T190036/analysis.json` | `db082881be1205df9172672b3ae37365531327ec67bd4b3f6ed432731f792955` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_analysis/20261009T190036/run_config.json` | `9f94ef6bc7a3e213e7a07f06f348f506130ab0462974d9587e8090f6634d945c` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_analysis/20261009T190036/selection_frequency.csv` | `056a11265c3b4ed5a5637946aaf320d2503f53377adf7097c506ebaa6845709f` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_temperature/20261009T194925Z/summary.json` | `ac9c5fe9b4a621b921311ed428ee4eee3abf8b40489bbabf5a278544391322a9` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_temperature/20261009T194925Z/run_config.json` | `d4b3a09865c154ce3efbebb930e72ab6d140cd435ceabacb54b76124a7d3c6dd` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_convergence/20261008T171143/summary.json` | `2ed2a8f1171c86e0138e4a7163c6d7a0f80b1e7b65d9f27859a8557678599dcf` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_convergence/20261008T171143/report.json` | `da3bd5290f6d6a2697710c3d9596500feb08ea32cae74be12740a3b7e29636d9` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_S/20261009T101641/run_configs/session_01.json` | `ffe4627be5429b5ac83d0556cb3fe713c4bc2cd745262ae21c298629d1bfb0e2` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_sigma/20261009T101646/run_configs/session_01.json` | `07c2daa2bcaf0561c54d0e33dafac9b489e7e221c71224b4d20a61e065c16485` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_kappa/20261009T101651/run_configs/session_01.json` | `c5f566b9105404b122d7caa3c7a7383a27e0d29b7a7dc046f56e9738d247f5e8` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_zT/20261009T101656/run_configs/session_01.json` | `9c6ce718ca84b5b95528e0820d511c272f366417f4340359782931a19a43d155` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_platform_record/20261008T174804Z/units/kappa_repeat0_fold0.json` | `da2ab2d6bacea9a9a6b408c0ebf55742dbe29574b1dfb1fb4a049db0e78e00f8` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_platform_record/20261008T174804Z/run_configs/session_01.json` | `1e6ffbf1ddb4920e19474faae67ef098c3f8412caa863504e6a5631d4d3bb33c` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_convergence/20261008T093143/diagnostic.json` | `26f5ebc01b110ed1562f2f1028a466296169013c9a728d5915dc7a205698e7cf` | <!-- thesis analysis result -->
| `thesis_paper/results/na13_convergence/20261008T093143/run_config.json` | `2a2b456b0de6fba3523ff7c386314388dc65b2973055ad5de4e6bb8e27358522` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_ranked/20261005T053426/summary.json` | `c6e5a0cdc996b5035183edc6dbfddeb6ea6e0c71f8f50dc521f0a116407c90dd` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_ranked/20261005T053426/ranked_main_30.csv` | `9390b31acd32d237c8497b59690cd81b2c6e26fe11705328f4fa42925e82a474` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_ranked/20261005T053426/shortlist.csv` | `501586df2768e175c2ea513244c9e5aef4d6482e089e74422fb76b9c3fcc71ff` | <!-- thesis analysis result -->
| `thesis_paper/results/na11_ranked/20261005T053426/run_config.json` | `3d1194f535879cdcbc627a5dacca158c0a1e333d5cf1757970369b3928ace6d3` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_rows_a/20261005T102144/results.json` | `d993b6aec67020445a63e5d1ea5f6ff23bcc0fed7fe67a394d01647606cf6bef` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_rows_a/20261005T102144/status.json` | `5d891efa676a17484c36dbfba0b4c5a05af3d150ecdaaf3536fea2f5ae227df4` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_rows_a/20261005T102144/run_configs/session_01.json` | `381fb080fadcd5db56ec8ea8e622120ae0c2a0650cf0ccbaa45bfbcee42d6d8e` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_rows_b/20261005T102144/results.json` | `9dd28fdca34294f2a4e991ca8441ff98b49baa6bd3223a441040718f281dd85c` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_rows_b/20261005T102144/status.json` | `5d891efa676a17484c36dbfba0b4c5a05af3d150ecdaaf3536fea2f5ae227df4` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_rows_b/20261005T102144/run_configs/session_01.json` | `721923b67847a3e290415b64b2694dd02988a9325aeb1df96f07ba84de4f20a3` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_a/20261004T201639/results.json` | `c3deb7e1122a9a3ab45edd38c07ae86bad5b5c9ab4e4a563e26236d2f20660f1` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_a/20261004T201639/status.json` | `eaf442ca79c074c4cc1f5408480809e417207bd0b6c79d42a8a9246e3ca62cf8` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_a/20261004T201639/run_configs/session_01.json` | `927ff243ccb12a38cfc816b216dbf5471cc17310accf918ac7b17167046db90d` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_b/20261004T201433/results.json` | `e32e6f7f816d7346e56191ce7512dcf48284d31ebc08b74ebcac48a639d1a690` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_b/20261004T201433/status.json` | `eaf442ca79c074c4cc1f5408480809e417207bd0b6c79d42a8a9246e3ca62cf8` | <!-- thesis analysis result -->
| `thesis_paper/results/na3_b/20261004T201433/run_configs/session_01.json` | `f82ac83a60f0f2d6f7638a37a78e395cf95a2401b32bb74d3dac34a72b5bde58` | <!-- thesis analysis result -->
| `src/backtransform_check.py` | `4317955f32fbcb6c9daa3602f588945f2e236bdb906af36a655f9d3c3412512e` | <!-- shared code -->
| `src/canonicalization.py` | `4d1b427e68a72ad95ffe5588d54b76467e350451e335e97dcb13b326efb9dfb1` | <!-- shared code -->
| `src/direct_vs_derived_zt.py` | `ae418a133adccaa7a65ac4953a13da5dc1eb054b14a0d2bacfa5ca65c92f3551` | <!-- shared code -->
| `src/external_validation.py` | `f38f1ab71d48fae7e1dd3e19418718569546733d690d298656523fece34930c2` | <!-- shared code -->
| `src/featurization.py` | `c2d72008d77d6ea5853fc2e16f2bb09847b88441a8444f0ddd9141577ef34a89` | <!-- shared code -->
| `src/plotting_style.py` | `2a35193e53996aecb47c997c9dd5d98801e9b51ddd93d4bba7565154cf94224a` | <!-- shared code -->
