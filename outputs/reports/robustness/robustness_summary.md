# Robustness study

Synthetic scene 640x640, 3 trials per cell, correct = ground-truth RMSE <= 3.0 px. Success counts only correct registrations; `false` = accepted by the gates but wrong.

| factor | level | detector | success | false-accept | rejected | median GT RMSE px | median inliers | s |
|---|---|---|---|---|---|---|---|---|
| rotation_deg | 0 | sift | 100% | 0% | 0% | 0.04 | 687 | 0.06 |
| rotation_deg | 0 | orb | 100% | 0% | 0% | 0.20 | 1410 | 0.08 |
| rotation_deg | 15 | sift | 100% | 0% | 0% | 0.11 | 706 | 0.06 |
| rotation_deg | 15 | orb | 100% | 0% | 0% | 0.27 | 1326 | 0.07 |
| rotation_deg | 45 | sift | 100% | 0% | 0% | 0.31 | 759 | 0.06 |
| rotation_deg | 45 | orb | 100% | 0% | 0% | 0.53 | 1299 | 0.07 |
| rotation_deg | 90 | sift | 100% | 0% | 0% | 0.53 | 695 | 0.05 |
| rotation_deg | 90 | orb | 100% | 0% | 0% | 0.92 | 1421 | 0.07 |
| rotation_deg | 180 | sift | 100% | 0% | 0% | 0.74 | 715 | 0.07 |
| rotation_deg | 180 | orb | 100% | 0% | 0% | 1.04 | 1391 | 0.07 |
| scale (src px per ref px) | 0.3 | sift | 100% | 0% | 0% | 0.81 | 175 | 0.05 |
| scale (src px per ref px) | 0.3 | orb | 100% | 0% | 0% | 2.02 | 96 | 0.02 |
| scale (src px per ref px) | 0.5 | sift | 100% | 0% | 0% | 0.36 | 385 | 0.04 |
| scale (src px per ref px) | 0.5 | orb | 100% | 0% | 0% | 0.71 | 566 | 0.04 |
| scale (src px per ref px) | 1.0 | sift | 100% | 0% | 0% | 0.08 | 875 | 0.06 |
| scale (src px per ref px) | 1.0 | orb | 100% | 0% | 0% | 0.34 | 2195 | 0.08 |
| scale (src px per ref px) | 2.0 | sift | 100% | 0% | 0% | 0.18 | 924 | 0.09 |
| scale (src px per ref px) | 2.0 | orb | 100% | 0% | 0% | 0.32 | 1259 | 0.08 |
| scale (src px per ref px) | 3.0 | sift | 100% | 0% | 0% | 0.24 | 876 | 0.13 |
| scale (src px per ref px) | 3.0 | orb | 100% | 0% | 0% | 0.39 | 604 | 0.09 |
| brightness (gain,bias) | (1.0, 0) | sift | 100% | 0% | 0% | 0.10 | 710 | 0.06 |
| brightness (gain,bias) | (1.0, 0) | orb | 100% | 0% | 0% | 0.37 | 1266 | 0.07 |
| brightness (gain,bias) | (0.6, 0) | sift | 100% | 0% | 0% | 0.10 | 697 | 0.05 |
| brightness (gain,bias) | (0.6, 0) | orb | 100% | 0% | 0% | 0.29 | 1288 | 0.07 |
| brightness (gain,bias) | (1.4, 20) | sift | 100% | 0% | 0% | 0.09 | 746 | 0.06 |
| brightness (gain,bias) | (1.4, 20) | orb | 100% | 0% | 0% | 0.29 | 1212 | 0.08 |
| brightness (gain,bias) | (0.3, -30) | sift | 100% | 0% | 0% | 0.08 | 542 | 0.05 |
| brightness (gain,bias) | (0.3, -30) | orb | 100% | 0% | 0% | 0.29 | 1066 | 0.07 |
| noise sigma | 0 | sift | 100% | 0% | 0% | 0.10 | 710 | 0.06 |
| noise sigma | 0 | orb | 100% | 0% | 0% | 0.37 | 1266 | 0.07 |
| noise sigma | 20 | sift | 100% | 0% | 0% | 0.13 | 207 | 0.06 |
| noise sigma | 20 | orb | 100% | 0% | 0% | 0.30 | 898 | 0.08 |
| noise sigma | 40 | sift | 100% | 0% | 0% | 0.32 | 108 | 0.06 |
| noise sigma | 40 | orb | 100% | 0% | 0% | 0.40 | 590 | 0.08 |
| noise sigma | 70 | sift | 100% | 0% | 0% | 0.91 | 53 | 0.05 |
| noise sigma | 70 | orb | 100% | 0% | 0% | 1.44 | 312 | 0.08 |
| noise sigma | 100 | sift | 100% | 0% | 0% | 1.30 | 22 | 0.05 |
| noise sigma | 100 | orb | 100% | 0% | 0% | 1.38 | 174 | 0.08 |
| occlusion fraction | 0.0 | sift | 100% | 0% | 0% | 0.10 | 710 | 0.06 |
| occlusion fraction | 0.0 | orb | 100% | 0% | 0% | 0.37 | 1266 | 0.07 |
| occlusion fraction | 0.3 | sift | 100% | 0% | 0% | 0.08 | 285 | 0.05 |
| occlusion fraction | 0.3 | orb | 100% | 0% | 0% | 0.25 | 629 | 0.05 |
| occlusion fraction | 0.5 | sift | 100% | 0% | 0% | 0.13 | 183 | 0.05 |
| occlusion fraction | 0.5 | orb | 100% | 0% | 0% | 0.35 | 251 | 0.04 |
| occlusion fraction | 0.7 | sift | 100% | 0% | 0% | 0.17 | 85 | 0.05 |
| occlusion fraction | 0.7 | orb | 100% | 0% | 0% | 0.62 | 116 | 0.03 |
| occlusion fraction | 0.85 | sift | 33% | 0% | 67% | 0.18 | 0 | 0.05 |
| occlusion fraction | 0.85 | orb | 0% | 0% | 100% | nan | 0 | 0.03 |
| low texture (blur px, contrast) | (0, 1.0) | sift | 100% | 0% | 0% | 0.10 | 710 | 0.06 |
| low texture (blur px, contrast) | (0, 1.0) | orb | 100% | 0% | 0% | 0.37 | 1266 | 0.07 |
| low texture (blur px, contrast) | (4, 0.4) | sift | 100% | 0% | 0% | 0.09 | 265 | 0.04 |
| low texture (blur px, contrast) | (4, 0.4) | orb | 100% | 0% | 0% | 0.32 | 785 | 0.03 |
| low texture (blur px, contrast) | (8, 0.25) | sift | 100% | 0% | 0% | 0.13 | 92 | 0.03 |
| low texture (blur px, contrast) | (8, 0.25) | orb | 100% | 0% | 0% | 0.41 | 270 | 0.01 |
| low texture (blur px, contrast) | (12, 0.15) | sift | 100% | 0% | 0% | 0.19 | 70 | 0.03 |
| low texture (blur px, contrast) | (12, 0.15) | orb | 100% | 0% | 0% | 0.35 | 304 | 0.02 |
| low texture (blur px, contrast) | (16, 0.08) | sift | 100% | 0% | 0% | 0.14 | 106 | 0.03 |
| low texture (blur px, contrast) | (16, 0.08) | orb | 100% | 0% | 0% | 0.40 | 401 | 0.02 |
| source resolution (x ref) | 1.0 | sift | 100% | 0% | 0% | 0.08 | 875 | 0.05 |
| source resolution (x ref) | 1.0 | orb | 100% | 0% | 0% | 0.34 | 2195 | 0.08 |
| source resolution (x ref) | 0.35 | sift | 100% | 0% | 0% | 0.65 | 230 | 0.04 |
| source resolution (x ref) | 0.35 | orb | 100% | 0% | 0% | 1.58 | 187 | 0.03 |
| source resolution (x ref) | 0.2 | sift | 100% | 0% | 0% | 1.50 | 67 | 0.05 |
| source resolution (x ref) | 0.2 | orb | 0% | 0% | 100% | nan | 0 | 0.02 |
| source resolution (x ref) | 0.12 | sift | 33% | 0% | 67% | 2.23 | 0 | 0.06 |
| source resolution (x ref) | 0.12 | orb | 0% | 0% | 100% | nan | 0 | 0.00 |
| source resolution (x ref) | 0.08 | sift | 0% | 0% | 100% | nan | 0 | 0.03 |
| source resolution (x ref) | 0.08 | orb | 0% | 0% | 100% | nan | 0 | 0.00 |
| perspective strength | 0.0 | sift | 100% | 0% | 0% | 0.10 | 710 | 0.10 |
| perspective strength | 0.0 | orb | 100% | 0% | 0% | 0.37 | 1266 | 0.09 |
| perspective strength | 0.0002 | sift | 100% | 0% | 0% | 0.09 | 688 | 0.08 |
| perspective strength | 0.0002 | orb | 100% | 0% | 0% | 0.29 | 1432 | 0.09 |
| perspective strength | 0.0004 | sift | 100% | 0% | 0% | 0.09 | 663 | 0.06 |
| perspective strength | 0.0004 | orb | 100% | 0% | 0% | 0.27 | 1524 | 0.07 |
| perspective strength | 0.0006 | sift | 100% | 0% | 0% | 0.08 | 645 | 0.06 |
| perspective strength | 0.0006 | orb | 100% | 0% | 0% | 0.36 | 1478 | 0.07 |
| perspective strength | 0.0008 | sift | 100% | 0% | 0% | 0.08 | 607 | 0.06 |
| perspective strength | 0.0008 | orb | 100% | 0% | 0% | 0.30 | 1409 | 0.07 |
| scene change (area replaced) | 0.0 | sift | 100% | 0% | 0% | 0.10 | 710 | 0.06 |
| scene change (area replaced) | 0.0 | orb | 100% | 0% | 0% | 0.37 | 1266 | 0.07 |
| scene change (area replaced) | 0.3 | sift | 100% | 0% | 0% | 0.08 | 391 | 0.08 |
| scene change (area replaced) | 0.3 | orb | 100% | 0% | 0% | 0.36 | 730 | 0.08 |
| scene change (area replaced) | 0.6 | sift | 100% | 0% | 0% | 0.14 | 79 | 0.07 |
| scene change (area replaced) | 0.6 | orb | 100% | 0% | 0% | 0.72 | 50 | 0.08 |
| scene change (area replaced) | 0.8 | sift | 0% | 0% | 100% | nan | 0 | 0.10 |
| scene change (area replaced) | 0.8 | orb | 0% | 0% | 100% | nan | 0 | 0.12 |
| scene change (area replaced) | 0.9 | sift | 0% | 0% | 100% | nan | 0 | 0.10 |
| scene change (area replaced) | 0.9 | orb | 0% | 0% | 100% | nan | 0 | 0.12 |
