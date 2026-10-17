# Transfer and disk budget

Measured: OCC weights 3,441,185,608 bytes and tokenizer JSON 11,425,566; full snapshot unknown. E5 main file 1,112,201,288; full local exports about 3.9GB. Cross-Encoder 470,592,698. These listed model weights total 5,023,979,594 bytes before other files. Repo DB/sidecar 14,065,664/3,926,936; external 185,470,976/37,180,228; June backups 147,177,472 each. Notes ~3.1MB; Parquet 1,075,513 bytes. Historical run about 1,084,832 KiB (~1.11GB) including compact telemetry.

No throughput, full image size, full generator snapshot, future full-logit output, peak RAM/VRAM or total arm size is measured. Required free space formula: image + model cache + received DB + writable DB clone + rebuilt sidecar + run output + temporary archive. Assume source and destination coexist. Measure one-query archive before selecting capacity; do not derive download time without a network assumption.
