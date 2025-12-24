import tomllib
from pathlib import Path


def caret_to_pip(name: str, spec: str) -> str:
    """
    Convert Poetry caret constraints to pip-compatible ranges.
    Examples:
      ^1.18.2   -> >=1.18.2,<2.0.0
      ^0.12.11  -> >=0.12.11,<0.13.0
      ^0.0.5    -> >=0.0.5,<0.0.6
    """
    if not spec.startswith("^"):
        return f"{name}{spec}"

    version = spec[1:]
    parts = version.split(".")

    while len(parts) < 3:
        parts.append("0")

    major, minor, patch = map(int, parts[:3])

    if major > 0:
        upper = f"{major + 1}.0.0"
    elif minor > 0:
        upper = f"0.{minor + 1}.0"
    else:
        upper = f"0.0.{patch + 1}"

    return f"{name}>={version},<{upper}"


def normalize_dep(name: str, spec) -> str:
    """
    Normalize a Poetry dependency entry into pip-compatible syntax.
    """
    if isinstance(spec, dict):
        spec = spec.get("version", "")
    spec = spec.strip()

    if not spec:
        return name

    return caret_to_pip(name, spec)


def main() -> None:
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    dev_deps = (
        data
        .get("tool", {})
        .get("poetry", {})
        .get("group", {})
        .get("dev", {})
        .get("dependencies", {})
    )

    output = Path("requirements-dev.txt")

    with output.open("w") as f:
        for name, spec in dev_deps.items():
            line = normalize_dep(name, spec)
            f.write(f"{line}\n")

    print(f"Generated {output} ({len(dev_deps)} dependencies)")


if __name__ == "__main__":
    main()
