#!/usr/bin/env python3
"""
Forensic Export Script for HEAN Project
Creates a complete snapshot with reports, logs, manifests, and metadata.
"""

import os
import sys
import shutil
import subprocess
import json
import hashlib
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Optional
import re

# Configuration
PROJECT_ROOT = Path(__file__).parent.absolute()
EXPORT_ROOT = PROJECT_ROOT / "EXPORT_BUNDLE"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP_NAME = f"HEAN_FULL_EXPORT_{TIMESTAMP}.zip"

# Exclusion patterns
EXCLUDE_PATTERNS = {
    # Python
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "*.so",
    "*.egg-info",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    ".tox",
    # Virtual environments
    "venv",
    "env",
    "ENV",
    ".venv",
    # Node.js
    "node_modules",
    ".next",
    ".turbo",
    # Build artifacts
    "build",
    "dist",
    "*.egg",
    # IDE
    ".vscode",
    ".idea",
    "*.swp",
    "*.swo",
    "*~",
    # OS
    ".DS_Store",
    "Thumbs.db",
    # Project specific
    "*.db",
    "*.sqlite",
    # Logs (will be copied separately)
    "*.log",
    "logs",
    # Export bundle itself
    "EXPORT_BUNDLE",
    # Existing zip
    "HEAN_FULL_EXPORT_*.zip",
}

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    "env",
    "ENV",
    ".venv",
    "node_modules",
    ".next",
    ".turbo",
    "build",
    "dist",
    ".vscode",
    ".idea",
    "htmlcov",
    ".tox",
    "logs",
    "EXPORT_BUNDLE",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
}


def should_exclude(path: Path, root: Path) -> bool:
    """Check if path should be excluded."""
    rel_path = path.relative_to(root)

    # Check exact matches
    if path.name in EXCLUDE_FILES:
        return True

    # Check directory names
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return True

    # Check patterns
    path_str = str(rel_path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str or path.name == pattern.replace("*", ""):
            return True

    # Check if it's a log file (will be copied separately)
    if path.suffix == ".log" and "EXPORT_BUNDLE" not in path_str:
        return False  # Don't exclude, we'll copy them separately

    return False


def mask_env_value(value: str) -> str:
    """Mask sensitive environment variable values."""
    if not value or len(value.strip()) == 0:
        return ""

    value = value.strip()
    # If it looks like a key/token, mask it
    if len(value) > 8:
        return f"{value[:4]}...{value[-4:]}"
    else:
        return "***MASKED***"


def process_env_file(env_path: Path) -> tuple[str, str]:
    """Process .env file: create masked version and example."""
    if not env_path.exists():
        return "", ""

    masked_lines = []
    example_lines = []

    with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                masked_lines.append(line)
                example_lines.append(line)
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Masked version
                masked_value = mask_env_value(value)
                masked_lines.append(f"{key}={masked_value}")

                # Example version (no value)
                example_lines.append(f"{key}=")
            else:
                masked_lines.append(line)
                example_lines.append(line)

    masked_content = "\n".join(masked_lines)
    example_content = "\n".join(example_lines)

    return masked_content, example_content


def copy_project_snapshot():
    """Copy entire project to snapshot directory with exclusions."""
    print("📦 Copying project snapshot...")

    snapshot_dir = EXPORT_ROOT / "project_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    excluded_paths = []
    copied_count = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)

        # Skip EXPORT_BUNDLE itself
        if "EXPORT_BUNDLE" in root_path.parts:
            continue

        # Filter directories
        dirs[:] = [d for d in dirs if not should_exclude(root_path / d, PROJECT_ROOT)]

        # Create relative path
        rel_root = root_path.relative_to(PROJECT_ROOT)
        dest_root = snapshot_dir / rel_root

        # Process files
        for file in files:
            file_path = root_path / file

            if should_exclude(file_path, PROJECT_ROOT):
                excluded_paths.append(str(file_path.relative_to(PROJECT_ROOT)))
                continue

            # Copy file
            dest_file = dest_root / file
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(file_path, dest_file)
                copied_count += 1
            except Exception as e:
                print(f"⚠️  Warning: Could not copy {file_path}: {e}")
                excluded_paths.append(str(file_path.relative_to(PROJECT_ROOT)))

    # Handle .env files specially
    env_files = [".env", ".env.bak", ".env.local"]
    for env_file in env_files:
        env_path = PROJECT_ROOT / env_file
        if env_path.exists():
            # Copy masked version to reports
            masked_content, example_content = process_env_file(env_path)

            reports_dir = EXPORT_ROOT / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Save masked version
            masked_path = reports_dir / f"{env_file}.masked"
            with open(masked_path, "w", encoding="utf-8") as f:
                f.write(masked_content)

            # Save example version
            example_path = snapshot_dir / f"{env_file}.export.example"
            with open(example_path, "w", encoding="utf-8") as f:
                f.write(example_content)

            excluded_paths.append(str(env_path.relative_to(PROJECT_ROOT)))

    # Save excluded paths report
    excluded_file = EXPORT_ROOT / "manifests" / "excluded_paths.txt"
    excluded_file.parent.mkdir(parents=True, exist_ok=True)
    with open(excluded_file, "w", encoding="utf-8") as f:
        f.write("EXCLUDED PATHS FROM SNAPSHOT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total excluded: {len(excluded_paths)}\n\n")
        for path in sorted(excluded_paths):
            f.write(f"{path}\n")

    print(f"✅ Copied {copied_count} files")
    print(f"⚠️  Excluded {len(excluded_paths)} paths")

    return excluded_paths


def collect_logs():
    """Collect existing logs from project."""
    print("📋 Collecting logs...")

    logs_dir = EXPORT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Find all log files
    log_files = list(PROJECT_ROOT.glob("*.log"))
    log_dirs = [PROJECT_ROOT / "logs"]

    copied_logs = []

    # Copy log files from root
    for log_file in log_files:
        if log_file.exists():
            dest = logs_dir / log_file.name
            try:
                shutil.copy2(log_file, dest)
                copied_logs.append(str(log_file.relative_to(PROJECT_ROOT)))
            except Exception as e:
                print(f"⚠️  Warning: Could not copy log {log_file}: {e}")

    # Copy logs directory if it exists and has files
    for log_dir in log_dirs:
        if log_dir.exists() and log_dir.is_dir():
            for log_file in log_dir.rglob("*"):
                if log_file.is_file():
                    rel_path = log_file.relative_to(PROJECT_ROOT)
                    dest = logs_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(log_file, dest)
                        copied_logs.append(str(rel_path))
                    except Exception as e:
                        print(f"⚠️  Warning: Could not copy log {log_file}: {e}")

    print(f"✅ Collected {len(copied_logs)} log files")
    return copied_logs


def run_command(
    cmd: List[str], cwd: Optional[Path] = None, timeout: int = 30
) -> tuple[str, str, int]:
    """Run command and return stdout, stderr, exit_code."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 124
    except FileNotFoundError:
        return "", "Command not found", 127
    except Exception as e:
        return "", str(e), 1


def generate_reports():
    """Generate all required reports."""
    print("📊 Generating reports...")

    reports_dir = EXPORT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. repo_tree.txt
    print("  → repo_tree.txt")
    stdout, stderr, code = run_command(
        ["tree", "-L", "5", "-I", "__pycache__|*.pyc|.git"], timeout=10
    )
    if code == 0:
        with open(reports_dir / "repo_tree.txt", "w", encoding="utf-8") as f:
            f.write(stdout)
    else:
        # Fallback: use find
        stdout, _, _ = run_command(
            [
                "find",
                ".",
                "-type",
                "d",
                "-not",
                "-path",
                "*/.git/*",
                "-not",
                "-path",
                "*/__pycache__/*",
            ],
            timeout=10,
        )
        with open(reports_dir / "repo_tree.txt", "w", encoding="utf-8") as f:
            f.write("Directory structure:\n" + stdout)

    # 2. file_inventory.csv
    print("  → file_inventory.csv")
    inventory = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if "EXPORT_BUNDLE" in root or ".git" in root:
            continue
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            try:
                stat = file_path.stat()
                inventory.append(
                    {
                        "path": str(file_path.relative_to(PROJECT_ROOT)),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": file_path.suffix or "no_ext",
                    }
                )
            except:
                pass

    with open(reports_dir / "file_inventory.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "modified", "type"])
        writer.writeheader()
        writer.writerows(inventory)

    # 3. git_status.txt
    print("  → git_status.txt")
    stdout, stderr, code = run_command(["git", "status"], timeout=10)
    with open(reports_dir / "git_status.txt", "w", encoding="utf-8") as f:
        if code == 0:
            f.write(stdout)
            if stderr:
                f.write("\n\nSTDERR:\n" + stderr)
        else:
            f.write(
                f"SKIPPED: Git not available or not a git repository\nExit code: {code}\n{stderr}"
            )

    # 4. git_log.txt
    print("  → git_log.txt")
    stdout, stderr, code = run_command(
        ["git", "log", "-n", "200", "--oneline", "--decorate"], timeout=10
    )
    with open(reports_dir / "git_log.txt", "w", encoding="utf-8") as f:
        if code == 0:
            f.write(stdout)
        else:
            f.write(f"SKIPPED: Git not available\nExit code: {code}\n{stderr}")

    # 5. git_diff.patch
    print("  → git_diff.patch")
    stdout, stderr, code = run_command(["git", "diff"], timeout=10)
    with open(reports_dir / "git_diff.patch", "w", encoding="utf-8") as f:
        if code == 0:
            if stdout.strip():
                f.write(stdout)
            else:
                f.write("No uncommitted changes.\n")
        else:
            f.write(f"SKIPPED: Git not available\nExit code: {code}\n{stderr}")

    # 6. python_env.txt
    print("  → python_env.txt")
    python_info = []
    stdout, _, code = run_command(["python3", "--version"], timeout=5)
    python_info.append(f"Python version: {stdout.strip() if code == 0 else 'SKIPPED'}")

    # Try pip freeze
    stdout, stderr, code = run_command(["pip", "freeze"], timeout=10)
    if code == 0:
        python_info.append("\n=== pip freeze ===\n" + stdout)
    else:
        # Try uv pip freeze
        stdout, stderr, code = run_command(["uv", "pip", "freeze"], timeout=10)
        if code == 0:
            python_info.append("\n=== uv pip freeze ===\n" + stdout)
        else:
            # Try poetry show
            stdout, stderr, code = run_command(["poetry", "show"], timeout=10)
            if code == 0:
                python_info.append("\n=== poetry show ===\n" + stdout)
            else:
                python_info.append("\n=== Dependencies ===\nSKIPPED: pip/uv/poetry not available")

    with open(reports_dir / "python_env.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(python_info))

    # 7. node_env.txt
    print("  → node_env.txt")
    node_info = []
    stdout, _, code = run_command(["node", "-v"], timeout=5)
    node_info.append(f"Node version: {stdout.strip() if code == 0 else 'SKIPPED'}")

    stdout, _, code = run_command(["npm", "-v"], timeout=5)
    node_info.append(f"npm version: {stdout.strip() if code == 0 else 'SKIPPED'}")

    # Try npm list
    stdout, stderr, code = run_command(["npm", "list", "--depth=0"], timeout=10)
    if code == 0:
        node_info.append("\n=== npm list ===\n" + stdout)
    else:
        node_info.append("\n=== Dependencies ===\nSKIPPED: npm not available or no package.json")

    with open(reports_dir / "node_env.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(node_info))

    # 8. make_targets.txt
    print("  → make_targets.txt")
    stdout, stderr, code = run_command(["make", "-n"], timeout=10)
    if code != 0:
        stdout, stderr, code = run_command(["make", "help"], timeout=10)

    with open(reports_dir / "make_targets.txt", "w", encoding="utf-8") as f:
        if code == 0:
            f.write(stdout)
        else:
            f.write(f"SKIPPED: Make not available or no Makefile\nExit code: {code}\n{stderr}")

    # 9. tests_last_run.txt
    print("  → tests_last_run.txt")
    stdout, stderr, code = run_command(["pytest", "--version"], timeout=5)
    if code == 0:
        # Run tests in dry-run mode or just show what would run
        stdout, stderr, code = run_command(["pytest", "--collect-only", "-q"], timeout=30)
        with open(reports_dir / "tests_last_run.txt", "w", encoding="utf-8") as f:
            f.write("=== Test Collection (dry-run) ===\n")
            f.write(stdout)
            if stderr:
                f.write("\n\nSTDERR:\n" + stderr)
    else:
        with open(reports_dir / "tests_last_run.txt", "w", encoding="utf-8") as f:
            f.write("SKIPPED: pytest not available\n")

    # 10. lint_last_run.txt
    print("  → lint_last_run.txt")
    # Try ruff
    stdout, stderr, code = run_command(["ruff", "--version"], timeout=5)
    if code == 0:
        stdout, stderr, code = run_command(
            ["ruff", "check", "--output-format=text", "src/"], timeout=30
        )
        with open(reports_dir / "lint_last_run.txt", "w", encoding="utf-8") as f:
            f.write("=== Ruff Check ===\n")
            f.write(stdout)
            if stderr:
                f.write("\n\nSTDERR:\n" + stderr)
    else:
        with open(reports_dir / "lint_last_run.txt", "w", encoding="utf-8") as f:
            f.write("SKIPPED: ruff not available\n")

    # 11. docker_info.txt
    print("  → docker_info.txt")
    docker_info = []

    stdout, stderr, code = run_command(["docker", "--version"], timeout=5)
    docker_info.append(f"Docker version: {stdout.strip() if code == 0 else 'SKIPPED'}")

    stdout, stderr, code = run_command(["docker", "compose", "version"], timeout=5)
    docker_info.append(f"Docker Compose version: {stdout.strip() if code == 0 else 'SKIPPED'}")

    # Find compose files
    compose_files = list(PROJECT_ROOT.glob("docker-compose*.yml")) + list(
        PROJECT_ROOT.glob("docker-compose*.yaml")
    )
    docker_info.append(f"\n=== Compose Files ===\n")
    for cf in compose_files:
        docker_info.append(str(cf.relative_to(PROJECT_ROOT)))

    with open(reports_dir / "docker_info.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(docker_info))

    # 12. runtime_smoke.txt
    print("  → runtime_smoke.txt")
    # Try to run a safe smoke test
    stdout, stderr, code = run_command(
        ["python3", "-c", "import hean; print('HEAN module import: OK')"], timeout=10
    )
    with open(reports_dir / "runtime_smoke.txt", "w", encoding="utf-8") as f:
        f.write("=== Smoke Test ===\n")
        if code == 0:
            f.write(stdout)
            f.write("\n✅ Module import successful\n")
        else:
            f.write(f"⚠️  Module import failed\nExit code: {code}\n")
            f.write(f"STDOUT: {stdout}\n")
            f.write(f"STDERR: {stderr}\n")

    print("✅ All reports generated")


def generate_manifests():
    """Generate manifest files."""
    print("📝 Generating manifests...")

    manifests_dir = EXPORT_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # 1. sha256_manifest.txt
    print("  → sha256_manifest.txt")
    snapshot_dir = EXPORT_ROOT / "project_snapshot"
    reports_dir = EXPORT_ROOT / "reports"

    hashes = []
    for directory in [snapshot_dir, reports_dir]:
        if directory.exists():
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        with open(file_path, "rb") as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()
                        rel_path = file_path.relative_to(EXPORT_ROOT)
                        hashes.append(f"{file_hash}  {rel_path}")
                    except Exception as e:
                        hashes.append(f"ERROR  {file_path.relative_to(EXPORT_ROOT)}: {e}")

    with open(manifests_dir / "sha256_manifest.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(hashes)))

    # 2. export_meta.json
    print("  → export_meta.json")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "user": os.getenv("USER", "unknown"),
        "os": sys.platform,
        "python_version": sys.version,
    }

    # Git info
    stdout, _, code = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=5)
    meta["git_branch"] = stdout.strip() if code == 0 else "unknown"

    stdout, _, code = run_command(["git", "rev-parse", "HEAD"], timeout=5)
    meta["git_commit"] = stdout.strip() if code == 0 else "unknown"

    with open(manifests_dir / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Manifests generated")


def generate_system_info():
    """Generate system information."""
    print("💻 Generating system info...")

    system_dir = EXPORT_ROOT / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    info = []
    info.append(f"OS: {sys.platform}")
    info.append(f"Python: {sys.version}")
    info.append(f"User: {os.getenv('USER', 'unknown')}")
    info.append(f"Home: {os.getenv('HOME', 'unknown')}")
    info.append(f"PWD: {PROJECT_ROOT}")

    # System commands
    for cmd, label in [
        (["uname", "-a"], "uname"),
        (["sw_vers"], "macOS version"),
        (["df", "-h"], "disk usage"),
    ]:
        stdout, stderr, code = run_command(cmd, timeout=5)
        if code == 0:
            info.append(f"\n=== {label} ===\n{stdout}")

    with open(system_dir / "system_info.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(info))

    print("✅ System info generated")


def create_zip():
    """Create final ZIP archive."""
    print("📦 Creating ZIP archive...")

    zip_path = PROJECT_ROOT / ZIP_NAME

    # Remove existing zip if any
    if zip_path.exists():
        zip_path.unlink()

    # Create zip
    shutil.make_archive(
        str(zip_path.with_suffix("")), "zip", root_dir=PROJECT_ROOT, base_dir="EXPORT_BUNDLE"
    )

    # Calculate checksum
    with open(zip_path, "rb") as f:
        zip_hash = hashlib.sha256(f.read()).hexdigest()

    # Get size
    zip_size = zip_path.stat().st_size
    zip_size_mb = zip_size / (1024 * 1024)

    print(f"✅ ZIP created: {zip_path}")
    print(f"   Size: {zip_size_mb:.2f} MB ({zip_size:,} bytes)")
    print(f"   SHA256: {zip_hash}")

    return zip_path, zip_hash, zip_size


def main():
    """Main execution."""
    print("=" * 60)
    print("HEAN Forensic Export")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Export root: {EXPORT_ROOT}")
    print(f"Timestamp: {TIMESTAMP}")
    print()

    try:
        # Create structure
        print("📁 Creating export structure...")
        for subdir in ["project_snapshot", "reports", "logs", "manifests", "system"]:
            (EXPORT_ROOT / subdir).mkdir(parents=True, exist_ok=True)

        # Copy project snapshot
        excluded = copy_project_snapshot()

        # Collect logs
        logs = collect_logs()

        # Generate reports
        generate_reports()

        # Generate manifests
        generate_manifests()

        # Generate system info
        generate_system_info()

        # Create ZIP
        zip_path, zip_hash, zip_size = create_zip()

        # Final summary
        print()
        print("=" * 60)
        print("✅ EXPORT COMPLETE")
        print("=" * 60)
        print(f"📦 ZIP Archive: {zip_path}")
        print(f"   Size: {zip_size / (1024 * 1024):.2f} MB")
        print(f"   SHA256: {zip_hash}")
        print()
        print("📂 Contents:")
        print("   - EXPORT_BUNDLE/project_snapshot/  (full project copy)")
        print("   - EXPORT_BUNDLE/reports/           (all reports)")
        print("   - EXPORT_BUNDLE/logs/              (collected logs)")
        print("   - EXPORT_BUNDLE/manifests/         (checksums, metadata)")
        print("   - EXPORT_BUNDLE/system/            (system information)")
        print()
        print("🔒 Security:")
        print("   - .env files masked and moved to reports/")
        print("   - .env.export.example created in snapshot")
        print()
        print(f"⚠️  Excluded {len(excluded)} paths (see manifests/excluded_paths.txt)")

    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
