"""Surgically remove workspace members + sources from prime-rl's pyproject.toml
that reference paths inside private submodules we don't have access to.

prime-rl's .gitmodules has four submodules:
  - deps/verifiers           (public — we clone)
  - deps/renderers           (public — we clone)
  - deps/research-environments (PRIVATE — we skip)
  - configs/private          (PRIVATE — we skip)

`uv sync` validates every member listed in `tool.uv.workspace.members` exists
on disk. We strip the missing ones, plus the `tool.uv.sources` entries that
no longer have a corresponding member, plus the `envs` optional-dep block
(which references the env packages under deps/research-environments).

Usage:
    python patch_prime_rl_pyproject.py /root/prime-rl/pyproject.toml
"""
import sys
import tomllib
from pathlib import Path

try:
    import tomli_w
except ImportError:
    print("ERROR: tomli_w not installed. Install with: pip install tomli_w", file=sys.stderr)
    sys.exit(1)


def main(pyproject_path: str) -> None:
    p = Path(pyproject_path)
    root = p.parent
    data = tomllib.loads(p.read_text())

    # 1. Filter workspace.members to only paths that exist on disk.
    uv = data.get("tool", {}).get("uv", {})
    workspace = uv.get("workspace", {})
    members = list(workspace.get("members", []))
    existing_members = [m for m in members if (root / m).exists()]
    missing = [m for m in members if not (root / m).exists()]
    workspace["members"] = existing_members

    # 2. Collect package names of existing members so we can prune sources.
    existing_pkg_names = set()
    for m in existing_members:
        sub_pyproj = root / m / "pyproject.toml"
        if sub_pyproj.exists():
            sub = tomllib.loads(sub_pyproj.read_text())
            name = sub.get("project", {}).get("name")
            if name:
                existing_pkg_names.add(name)
                # uv normalizes hyphens/underscores; track both forms.
                existing_pkg_names.add(name.replace("-", "_"))
                existing_pkg_names.add(name.replace("_", "-"))

    # 3. Filter sources: drop workspace=True entries whose package isn't a
    # surviving member. Leave non-workspace sources (git, index pins) alone.
    sources = uv.get("sources", {})
    new_sources = {}
    dropped_sources = []
    for k, v in sources.items():
        if isinstance(v, dict) and v.get("workspace") is True:
            if k in existing_pkg_names:
                new_sources[k] = v
            else:
                dropped_sources.append(k)
        else:
            new_sources[k] = v
    uv["sources"] = new_sources

    # 4. Drop project.optional-dependencies.envs (references the env packages
    # that lived in deps/research-environments).
    opt = data.get("project", {}).get("optional-dependencies", {})
    dropped_envs = "envs" in opt
    if dropped_envs:
        del opt["envs"]

    # 5. Also drop project.dependencies entries for packages we no longer
    # have a workspace member for (they'd otherwise be resolved from PyPI but
    # may not exist there).
    deps = data.get("project", {}).get("dependencies", [])
    new_deps = []
    dropped_deps = []
    for d in deps:
        # Strip version specs to get bare package name
        bare = d.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split(" ")[0].strip()
        # If this dep had a workspace=True source that we dropped, also drop the dep.
        if bare in dropped_sources:
            dropped_deps.append(d)
        else:
            new_deps.append(d)
    if "dependencies" in data.get("project", {}):
        data["project"]["dependencies"] = new_deps

    # Write back
    p.write_bytes(tomli_w.dumps(data).encode())

    # Report what we did
    print(f"Patched {p}")
    print(f"  workspace.members: removed {len(missing)} missing path(s):")
    for m in missing:
        print(f"    - {m}")
    print(f"  tool.uv.sources: dropped {len(dropped_sources)} workspace entries:")
    for s in dropped_sources:
        print(f"    - {s}")
    print(f"  project.dependencies: dropped {len(dropped_deps)} bare deps:")
    for d in dropped_deps:
        print(f"    - {d}")
    if dropped_envs:
        print(f"  project.optional-dependencies.envs: dropped")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/pyproject.toml>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
