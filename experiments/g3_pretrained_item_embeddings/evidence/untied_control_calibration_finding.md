# Untied-control calibration finding

All ten frozen native-50M jobs are contract-matched, horizon-complete, selection-resolved, and their saved metrics reproduce from the bound ranking evidence.

The best search cells at horizons 15, 25, and 40 fit `embedding_lr = 0.00027392030479946993 × horizon^1.9009722675001388` and `deep_lr = 0.7402079000956551 × horizon^-1.0659935019477855`. The corresponding fitted horizon-25 rates are `0.12447135415265811` and `0.023941907610393703`.

Horizon transfer is accepted under the lead-approved conservative interpretation of the previously ambiguous check. At horizon 25, the held-out Recall@100 is `0.07897170887245977`, versus `0.07837992363273662` for the best search control: an absolute difference of `0.0005917852397231554`, within the approved native-50M operational band of `0.015216678374059487`. This validates performance-region equivalence only; it does not validate LR-space distance.

Machine evidence: `untied_control_calibration.json`, logical SHA-256 `015c94a182bc0df4179092e098e69b9b12c4fc62474ff4a2f15ad5d3e693e896`.
