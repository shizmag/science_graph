# Author-side staging plan

Future prepare-assets command should verify source commit; extract/hash exact 375 manifest; decide historical versus new DB; export SQLite consistently; include note/source hashes and data rights; pin all models/tokenizers/templates; create package/image lock and digest; rebuild target USEARCH after restore; generate manifest with bytes/SHA256/revision/license/acquisition/auth; remove secrets; package and upload to controlled store; test interrupted transfer resume and checksum; preserve canonical source archive; preflight GPU/disk/output.

OCI registry is a candidate for image; versioned object store for DB/data/model bundle. No provider selected. Evaluate object limits, multipart/resumability, ACL, egress, retention and licensing. No package/upload performed.
