"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai RMIT Vietnam — rmit.edu.vn):
    - https://www.rmit.edu.vn/study-at-rmit/tuition-fees
    - https://www.rmit.edu.vn/study-at-rmit/scholarships/...
    - https://www.rmit.edu.vn/students/my-studies/fees-and-payments

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DAY07_DATA_DIR = Path("c:/AIThucChien/DAY07-2A202601861-BuiTienPhat/data/k3_university")
SYSTEM_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


def convert_md_to_pdf(md_path: Path, pdf_path: Path):
    from fpdf import FPDF
    print(f"Converting {md_path.name} to {pdf_path.name}...")
    
    # Read md content
    lines = md_path.read_text(encoding="utf-8").splitlines()
    
    # Simple conversion of markdown headers and lists to plain text
    content_lines = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            content_lines.append("")
            continue
        # Remove markdown heading symbols (#)
        if cleaned.startswith("#"):
            cleaned = cleaned.lstrip("#").strip()
        # Remove bold formatting
        cleaned = cleaned.replace("**", "").replace("__", "")
        content_lines.append(cleaned)
        
    content_text = "\n".join(content_lines)
    
    pdf = FPDF()
    pdf.add_page()
    
    # Register and use Windows system font Arial to support Vietnamese character glyphs
    if os.path.exists(SYSTEM_FONT_PATH):
        pdf.add_font("Arial_VN", "", SYSTEM_FONT_PATH)
        pdf.set_font("Arial_VN", size=11)
    else:
        # Fallback to standard Helvetica if font doesn't exist
        pdf.set_font("Helvetica", size=11)
        
    # Write text content line by line to handle wrapping
    pdf.multi_cell(0, 7, content_text)
    
    # Save PDF
    pdf.output(str(pdf_path))
    print(f"[OK] Saved PDF: {pdf_path}")


def main():
    setup_directory()
    
    files_to_convert = [
        "tuition-fees.md",
        "scholarships-policy.md",
        "course-registration.md",
        "library-services.md",
        "re-examination-rules.md"
    ]
    
    success_count = 0
    for filename in files_to_convert:
        md_file = DAY07_DATA_DIR / filename
        if md_file.exists():
            pdf_filename = filename.replace(".md", ".pdf")
            pdf_file = DATA_DIR / pdf_filename
            try:
                convert_md_to_pdf(md_file, pdf_file)
                success_count += 1
            except Exception as e:
                print(f"[ERROR] Loi khi chuyen doi {filename}: {e}")
        else:
            print(f"[WARNING] {filename} khong tim thay tai thu muc Day 7.")
            
    print(f"[OK] Hoan thanh tao {success_count}/{len(files_to_convert)} tai lieu PDF NTTU tai {DATA_DIR}")


if __name__ == "__main__":
    main()
