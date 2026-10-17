from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ablation_cuda import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="ablation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-release", help="freeze and package the collaborator input bundle")
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--artifact-store", default=os.environ.get("ABLA_ARTIFACT_STORE"),
                         help="author-configured local directory or private S3 URI")
    prepare.add_argument("--authorize-remote-upload", action="store_true",
                         help="explicitly authorize upload to the configured private S3 URI")

    verify = subparsers.add_parser("verify-results", help="verify and import a returned CUDA ablation archive")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--expected-sha256")
    verify.add_argument("--input-manifest", type=Path)
    verify.add_argument("--questions", type=Path)
    verify.add_argument("--destination", type=Path)

    args = parser.parse_args()
    if args.command == "prepare-release":
        from ablation_cuda.release import prepare_release
        from ablation_cuda.runtime.assets import sha256_file

        bundle, archive, staging = prepare_release(args.output_dir, args.artifact_store,
                                                  args.authorize_remote_upload)
        print(json.dumps({
            "release_version": __version__, "bundle_directory": str(bundle),
            "archive": str(archive), "archive_sha256": sha256_file(archive),
            "staging": staging,
        }, ensure_ascii=False, indent=2))
    else:
        from ablation_cuda.verify import verify_result_archive

        result = verify_result_archive(
            args.archive, expected_sha256=args.expected_sha256,
            input_manifest_path=args.input_manifest, questions_path=args.questions,
            destination=args.destination,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
