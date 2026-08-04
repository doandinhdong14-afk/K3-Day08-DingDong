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

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có và kiểm tra các file PDF."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in ('.pdf', '.docx')]
    print(f"[OK] Thu muc san sang: {DATA_DIR} ({len(files)} van ban)")
    for f in files:
        print(f"  - {f.name}")


def verify_and_collect_docs() -> list[Path]:
    """Kiểm tra và trả về danh sách các văn bản pháp lý đã thu thập."""
    setup_directory()
    files = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in ('.pdf', '.docx')]
    if len(files) < 3:
        print(f"[CANH BAO] Can toi thieu 3 van ban, hien co {len(files)} file.")
    else:
        print(f"[HOAN THANH Task 1] Da thu thap du {len(files)} van ban quy dinh PDF/DOCX.")
    return files


if __name__ == "__main__":
    verify_and_collect_docs()
