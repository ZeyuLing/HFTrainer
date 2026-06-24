# T2M HumanML3D Leaderboard

Web leaderboard: [T2M HumanML3D](t2m_humanml3d.html).

Default protocol: HumanML3D official test split with the selected ground-truth
caption for each clip. `MS` is the MotionStreamer-272 evaluator and `MC` is
the MotionCLIP evaluator. MC metrics use raw MotionCLIP projection embeddings
without L2 normalization. The `Ref` column records the motion reference used for
FID-style metrics, because raw SMPL references and HML263 round-trip references
should not be mixed.

## Semantic Metrics

| Method | Version | Ref | n | MS R1 ↑ | MS R2 ↑ | MS R3 ↑ | MS FID ↓ | MS MM ↓ | MS Div ↑ | MC R1 ↑ | MC R2 ↑ | MC R3 ↑ | MC FID ↓ | MC MM ↓ | MC Div ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT | 0 beta | raw/refk | 4042 | 0.7703 | 0.9030 | 0.9442 | 0.0000 | 14.8785 | 27.7705 | 0.7855 | 0.9039 | 0.9429 | -0.0000 | 41.5461 | 23.3569 |
| HYMotion | 1.0B | raw/refk | 4042 | 0.6438 | 0.7852 | 0.8438 | 14.6944 | 16.7121 | 27.4362 | 0.6105 | 0.7520 | 0.8144 | 132.6432 | 42.0210 | 23.0992 |
| HYMotion | 0.46B | raw/refk | 4042 | 0.6528 | 0.7932 | 0.8500 | 10.5127 | 16.6585 | 27.6548 | 0.6021 | 0.7434 | 0.8084 | 122.6162 | 41.9071 | 23.3367 |
| PRISM | 1.0 | raw/refk | 4042 | 0.7104 | 0.8317 | 0.8805 | 22.2574 | 16.2994 | 27.2828 | 0.5915 | 0.7187 | 0.7764 | 541.4167 | 41.5765 | 22.0501 |
| MotionStreamer | official | raw/refk | 4042 | 0.6303 | 0.7865 | 0.8498 | 12.2110 | 16.5810 | 27.4637 | 0.3960 | 0.5460 | 0.6324 | 46.0066 | 44.9556 | 23.0713 |
| T2M-GPT | official | HML263 round-trip | 4042 | 0.5516 | 0.7056 | 0.7788 | 25.4913 | 19.0912 | 25.5949 | 0.3978 | 0.5505 | 0.6400 | 378.1520 | 42.4744 | 23.0833 |
| MDM | official | HML263 round-trip | 4042 | 0.5208 | 0.6937 | 0.7701 | 35.5169 | 19.4246 | 25.3383 | 0.3520 | 0.5087 | 0.6085 | 429.1679 | 42.9340 | 22.8710 |
| MoMask | official | HML263 round-trip | 4042 | 0.6404 | 0.7974 | 0.8609 | 21.0729 | 18.1216 | 25.9789 | 0.4661 | 0.6336 | 0.7230 | 372.0083 | 41.7063 | 23.2418 |
| MoGenTS | official | HML263 round-trip | 4042 | 0.5910 | 0.7523 | 0.8138 | 109.8191 | 18.6038 | 25.3317 | 0.3575 | 0.5095 | 0.5995 | 350.6024 | 42.6554 | 23.4597 |
| MotionGPT3 | official | HML263 round-trip | 4042 | 0.6761 | 0.8274 | 0.8814 | 20.9324 | 17.5665 | 25.7047 | 0.4909 | 0.6559 | 0.7450 | 378.6097 | 41.5269 | 23.0807 |
| KIMODO | SMPL-X RP | raw/refk | 4042 | 0.3646 | 0.4998 | 0.5818 | 117.0279 | 21.4102 | 25.3629 | 0.2943 | 0.4305 | 0.5215 | 290.6178 | 44.9244 | 23.1677 |

## Physical Metrics

| Method | Version | n | Slide ↓ | Float ↓ | Jitter ↓ | Dynamic ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GT | 0 beta | 4042 | 0.0020 | 0.0264 | 0.0055 | 0.0232 |
| HYMotion | 1.0B | 4042 | 2.8324 | 11.2276 | 6.5152 | 25.7241 |
| HYMotion | 0.46B | 4042 | 3.2626 | 13.5222 | 7.0928 | 26.9823 |
| PRISM | 1.0 | 4042 | 4.8457 | 21.4024 | 7.4677 | 28.4771 |
| MotionStreamer | official | 4042 | 5.0011 | 17.0663 | 11.0975 | 23.6778 |
| T2M-GPT | official | 4042 | 3.7598 | 11.1799 | 4.9000 | 19.3651 |
| MDM | official | 4042 | 2.5318 | 4.9987 | 4.7034 | 22.1361 |
| MoMask | official | 4042 | 3.7319 | 9.5021 | 5.7226 | 23.4879 |
| MoGenTS | official | 4042 | 4.1508 | 12.3141 | 5.1289 | 23.0138 |
| MotionGPT3 | official | 4042 | 3.7913 | 9.6361 | 4.7523 | 23.1896 |
| KIMODO | SMPL-X RP | 4042 | 3.5414 | 2.7151 | 6.3208 | 29.9568 |

## Conversion Calibration

| Calibration | n | MS R1 ↑ | MS R3 ↑ | MS FID ↓ | MS MM ↓ | MS Div ↑ | MC R1 ↑ | MC R3 ↑ | MC FID ↓ | MC MM ↓ | MC Div ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT SMPL -> HML263 -> SMPL vs raw GT | 4042 | 0.7173 | 0.9219 | 67.4322 | 16.7867 | 26.6252 | 0.6649 | 0.8781 | 279.5049 | 40.5633 | 23.5585 |
