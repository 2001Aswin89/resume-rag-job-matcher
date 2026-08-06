from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT.parent / "dataset_generation"

RESUME_SOURCE = DATASET_DIR / "resumes.txt"
JD_SOURCE = DATASET_DIR / "job_descriptions.txt"

RESUME_OUTPUT = PROJECT_ROOT / "data" / "resumes"
JD_OUTPUT = PROJECT_ROOT / "data" / "job_descriptions"


SEPARATOR_PATTERN = re.compile(
    r"FILE:\s*(?P<filename>[^\n]+)\n=+\n",
    re.MULTILINE,
)


def split_file(source_path: Path, output_dir: Path):
    if not source_path.exists():
        raise FileNotFoundError(f"Could not find: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    content = source_path.read_text(encoding="utf-8")

    matches = list(SEPARATOR_PATTERN.finditer(content))

    if not matches:
        raise ValueError(f"No FILE sections found in {source_path.name}")

    created = []

    for index, match in enumerate(matches):
        filename = match.group("filename").strip()

        start = match.end()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        document = content[start:end].strip()

        output_file = output_dir / filename

        output_file.write_text(document + "\n", encoding="utf-8")

        created.append(output_file)

    return created


def main():
    print("\nSplitting resume dataset...\n")

    resumes = split_file(RESUME_SOURCE, RESUME_OUTPUT)

    print(f"Created {len(resumes)} resume files.")

    print("\nSplitting job descriptions...\n")

    jds = split_file(JD_SOURCE, JD_OUTPUT)

    print(f"Created {len(jds)} job description files.")

    print("\nDone.")

    print(f"Resume folder : {RESUME_OUTPUT}")

    print(f"JD folder     : {JD_OUTPUT}")


if __name__ == "__main__":
    main()