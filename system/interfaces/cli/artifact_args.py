# -*- coding: utf-8 -*-
import os

def _register_artifact_args(parser):
    parser.add_argument(
        "--artifact-registry",
        type=str,
        default="artifacts/checkpoints/adapter_artifacts.json",
        help="Adapter artifact registry path",
    )
    parser.add_argument(
        "--artifact-auto-prune",
        action="store_true",
        help="Prune adapter artifact registry at startup",
    )
    parser.add_argument(
        "--artifact-auto-maintain",
        action="store_true",
        help="Run artifact maintenance (audit + optional restore/prune) at startup",
    )
    parser.add_argument(
        "--artifact-maintain-now",
        action="store_true",
        help="Run artifact maintenance now and exit",
    )
    parser.add_argument(
        "--artifact-maintain-fix",
        action="store_true",
        help="Enable fix mode for artifact maintenance audit step",
    )
    parser.add_argument(
        "--artifact-maintain-prune",
        action="store_true",
        help="Enable prune step during artifact maintenance",
    )
    parser.add_argument(
        "--artifact-maintain-auto-restore",
        action="store_true",
        help="Enable auto-restore step during artifact maintenance when no valid adapter exists",
    )
    parser.add_argument(
        "--artifact-maintain-interval-s",
        type=float,
        default=0.0,
        help="Periodic maintenance interval in seconds (0 disables)",
    )
    parser.add_argument(
        "--artifact-maintain-min-free-gb",
        type=float,
        default=0.0,
        help="Trigger maintenance when free disk is below this GB threshold (0 disables)",
    )
    parser.add_argument(
        "--artifact-maintain-disk-path",
        type=str,
        default="checkpoints",
        help="Disk path used to evaluate free-space trigger for maintenance",
    )
    parser.add_argument(
        "--artifact-maintain-cooldown-s",
        type=float,
        default=120.0,
        help="Minimum seconds between periodic maintenance runs",
    )
    parser.add_argument(
        "--artifact-audit-now",
        action="store_true",
        help="Audit adapter artifact registry and exit",
    )
    parser.add_argument(
        "--artifact-audit-fix",
        action="store_true",
        help="Apply audit fixes to artifact registry",
    )
    parser.add_argument(
        "--artifact-audit-drop-missing",
        action="store_true",
        help="Drop missing artifact rows during audit fix",
    )
    parser.add_argument(
        "--artifact-audit-drop-invalid",
        action="store_true",
        help="Drop invalid artifact rows during audit fix",
    )
    parser.add_argument(
        "--artifact-audit-drop-missing-archives",
        action="store_true",
        help="Drop missing archive rows during audit fix",
    )
    parser.add_argument(
        "--artifact-audit-verify-hash",
        action="store_true",
        help="Verify archive SHA256 during audit",
    )
    parser.add_argument(
        "--artifact-audit-verify-signature",
        action="store_true",
        help="Verify archive signature during audit",
    )
    parser.add_argument(
        "--artifact-audit-require-signature",
        action="store_true",
        help="Treat missing archive signature as audit issue",
    )
    parser.add_argument(
        "--artifact-audit-sign-missing",
        action="store_true",
        help="Sign unsigned archives during audit fix when signing key is available",
    )
    parser.add_argument(
        "--artifact-prune-now",
        action="store_true",
        help="Prune adapter artifact registry and exit",
    )
    parser.add_argument(
        "--artifact-keep-latest",
        type=int,
        default=12,
        help="Keep latest N adapter artifacts",
    )
    parser.add_argument(
        "--artifact-keep-promoted",
        type=int,
        default=4,
        help="Keep latest N promoted adapter artifacts",
    )
    parser.add_argument(
        "--artifact-max-total-gb",
        type=float,
        default=0.0,
        help="Optional max total artifact size in GB (0 disables size cap)",
    )
    parser.add_argument(
        "--artifact-auto-restore-missing",
        action="store_true",
        help="Auto-restore latest archived adapter when no valid adapter is found",
    )
    parser.add_argument(
        "--artifact-auto-restore-output-dir",
        type=str,
        default="artifacts/checkpoints/restored_adapters",
        help="Restore directory for auto-restored adapters",
    )
    parser.add_argument(
        "--artifact-auto-restore-activate",
        action="store_true",
        help="Mark auto-restored adapter as active",
    )
    parser.add_argument(
        "--artifact-prune-delete-files",
        action="store_true",
        help="Delete artifact files when pruning",
    )
    parser.add_argument(
        "--artifact-prune-remove-invalid",
        action="store_true",
        help="Drop invalid adapter artifacts (missing adapter files)",
    )
    parser.add_argument(
        "--artifact-prune-archive",
        action="store_true",
        help="Archive artifacts before deleting during prune",
    )
    parser.add_argument(
        "--artifact-prune-archive-dir",
        type=str,
        default="artifacts/checkpoints/adapter_archive",
        help="Archive directory for pruned artifacts",
    )
    parser.add_argument(
        "--artifact-archive-signing-key-id",
        type=str,
        default="",
        help="Signing key id recorded in archive metadata",
    )
    parser.add_argument(
        "--artifact-signing-keys-json",
        type=str,
        default="",
        help="JSON map for signature verification key rotation (id->key)",
    )
    parser.add_argument(
        "--artifact-restore-now",
        action="store_true",
        help="Restore one archived adapter artifact and exit",
    )
    parser.add_argument(
        "--artifact-restore-archive-path",
        type=str,
        default="",
        help="Archive zip path to restore from (highest priority)",
    )
    parser.add_argument(
        "--artifact-restore-artifact-id",
        type=str,
        default="",
        help="Artifact id to restore from archive registry (uses latest archive for id)",
    )
    parser.add_argument(
        "--artifact-restore-preferred-base-model",
        type=str,
        default="",
        help="Preferred base model when selecting archive automatically",
    )
    parser.add_argument(
        "--artifact-restore-strict-base-model",
        action="store_true",
        help="Require base model match when selecting archive automatically",
    )
    parser.add_argument(
        "--artifact-restore-output-dir",
        type=str,
        default="artifacts/checkpoints/restored_adapters",
        help="Destination directory for restored artifacts",
    )
    parser.add_argument(
        "--artifact-restore-activate",
        action="store_true",
        help="Mark restored artifact as active in artifact registry",
    )
    parser.add_argument(
        "--artifact-restore-overwrite",
        action="store_true",
        help="Allow overwrite when restore target folder already exists",
    )
    parser.add_argument(
        "--artifact-restore-no-hash-verify",
        action="store_true",
        help="Disable archive SHA256 verification during restore",
    )
    parser.add_argument(
        "--artifact-restore-no-signature-verify",
        action="store_true",
        help="Disable archive HMAC signature verification during restore",
    )
    parser.add_argument(
        "--artifact-restore-require-signature",
        action="store_true",
        help="Require archive signature when restoring (strict security mode)",
    )
    parser.add_argument(
        "--adapter-strict-base-model",
        action="store_true",
        help="Only load adapters that match current transformer base model",
    )

