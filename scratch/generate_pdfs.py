import os
from pathlib import Path
from fpdf import FPDF

# Paths
DAY07_DATA_DIR = Path("c:/AIThucChien/DAY07-2A202601861-BuiTienPhat/data/k3_university")
DAY08_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DAY08_LEGAL_DIR.mkdir(parents=True, exist_ok=True)

# Vietnamese font from Windows system
SYSTEM_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

def convert_md_to_pdf(md_path: Path, pdf_path: Path):
    print(f"Converting {md_path.name} to {pdf_path.name}...")
    
    # Read md content
    lines = md_path.read_text(encoding="utf-8").splitlines()
    
    # Filter out frontmatter (metadata block)
    content_lines = []
    in_frontmatter = False
    frontmatter_count = 0
    
    for line in lines:
        if line.strip() == "---":
            frontmatter_count += 1
            if frontmatter_count == 1:
                in_frontmatter = True
                continue
            elif frontmatter_count == 2:
                in_frontmatter = False
                continue
        if not in_frontmatter:
            content_lines.append(line)
            
    content_text = "\n".join(content_lines).strip()
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Add Vietnamese font support
    if os.path.exists(SYSTEM_FONT_PATH):
        pdf.add_font("Arial_VN", "", SYSTEM_FONT_PATH)
        pdf.set_font("Arial_VN", size=11)
    else:
        # Fallback to standard Helvetica if font doesn't exist (might have accent issues)
        pdf.set_font("Helvetica", size=11)
        
    # Write text content line by line to handle wrapping
    pdf.multi_cell(0, 7, content_text)
    
    # Save PDF
    pdf.output(str(pdf_path))
    print(f"✓ Saved PDF: {pdf_path}")

def main():
    files_to_convert = [
        "tuition-fees.md",
        "scholarships-policy.md",
        "course-registration.md",
        "library-services.md",
        "re-examination-rules.md"
    ]
    
    for filename in files_to_convert:
        md_file = DAY07_DATA_DIR / filename
        if md_file.exists():
            pdf_filename = filename.replace(".md", ".pdf")
            pdf_file = DAY08_LEGAL_DIR / pdf_filename
            convert_md_to_pdf(md_file, pdf_file)
        else:
            print(f"⚠ Warning: {filename} not found in Day 7 directory.")

if __name__ == "__main__":
    main()
