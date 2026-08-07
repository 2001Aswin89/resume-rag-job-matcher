from pathlib import Path
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

PROJECT_ROOT = Path(__file__).resolve().parent
RESUME_DIR = PROJECT_ROOT / "data" / "resumes"

styles = getSampleStyleSheet()
style = styles["BodyText"]


def txt_to_pdf(txt_path: Path, pdf_path: Path):
    doc = SimpleDocTemplate(
        str(pdf_path),
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    story = []

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line:
                story.append(Paragraph(line.replace(" ", "&nbsp;"), style))
            else:
                story.append(Paragraph("<br/>", style))

    doc.build(story)


def main():
    converted = []

    for i in range(25, 31):
        txt_file = RESUME_DIR / f"resume_{i:02}.txt"
        pdf_file = RESUME_DIR / f"resume_{i:02}.pdf"

        if not txt_file.exists():
            print(f"Skipping {txt_file.name} (not found)")
            continue

        txt_to_pdf(txt_file, pdf_file)

        if pdf_file.exists():
            txt_file.unlink()
            converted.append(pdf_file.name)

    print("\nConverted files:")
    for file in converted:
        print(f"  ✓ {file}")

    txt_count = len(list(RESUME_DIR.glob("*.txt")))
    pdf_count = len(list(RESUME_DIR.glob("*.pdf")))

    print("\nDataset summary")
    print(f"TXT resumes : {txt_count}")
    print(f"PDF resumes : {pdf_count}")


if __name__ == "__main__":
    main()