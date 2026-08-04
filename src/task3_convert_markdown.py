"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> tuple[int, list[str]]:
    """
    Convert PDF/DOCX files trong data/landing/legal/ sang markdown.

    Returns:
        (số file convert thành công, danh sách tên file bị bỏ qua)

    Lưu ý: MarkItDown chỉ trích được text layer của PDF. PDF scan (ảnh chụp văn bản)
    sẽ trả về chuỗi rỗng — không phải lỗi code mà là bản chất file đầu vào. Muốn đưa
    những file đó vào corpus thì phải OCR trước (vd `ocrmypdf input.pdf output.pdf`)
    rồi chạy lại task này.
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted, skipped = 0, []

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                result = md.convert(str(filepath))
                content = result.text_content.strip()
                output_path = output_dir / f"{filepath.stem}.md"
                if content:
                    output_path.write_text(content, encoding="utf-8")
                    print(f"  [OK] Saved: {output_path.name} ({len(content)} chars)")
                    converted += 1
                else:
                    if output_path.exists():
                        output_path.unlink()
                    print(f"  [WARNING] Khong trich duoc text (PDF scan?) - bo qua: {filepath.name}")
                    skipped.append(filepath.name)
            except Exception as e:
                print(f"  [ERROR] Failed to convert {filepath.name}: {e}")
                skipped.append(filepath.name)

    return converted, skipped


def convert_news_articles() -> int:
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"

                header = f"# {data.get('title', 'Unknown')}\n\n"
                header += f"**Source:** {data.get('url', 'N/A')}\n"
                header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

                content = header + data.get("content_markdown", "")
                output_path.write_text(content, encoding="utf-8")
                print(f"  [OK] Saved: {output_path.name}")
                converted += 1
            except Exception as e:
                print(f"  [ERROR] Failed to convert {filepath.name}: {e}")

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_ok, legal_skipped = convert_legal_docs()

    print("\n--- News Articles ---")
    news_ok = convert_news_articles()

    print("\n" + "=" * 50)
    print(f"[TONG KET] legal: {legal_ok} converted, {len(legal_skipped)} bo qua | news: {news_ok} converted")
    if legal_skipped:
        print("\n[CANH BAO] Cac file sau KHONG co trong corpus (khong trich duoc text):")
        for name in legal_skipped:
            print(f"  - {name}")
        print("  -> Nhieu kha nang la PDF scan. Chay OCR truoc neu can dua vao corpus:")
        print("     pip install ocrmypdf && ocrmypdf --language vie input.pdf output.pdf")
    print(f"\n[OK] Done! Output tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
