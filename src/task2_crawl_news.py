"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.

⚠️ Nguyên tắc quan trọng: KHÔNG ghi dữ liệu bịa (mock/placeholder) vào corpus khi crawl
thất bại. Corpus RAG chỉ được chứa nội dung thật — nếu lẫn văn bản tự chế, chatbot sẽ
trích dẫn một "quy định" không tồn tại và toàn bộ điểm faithfulness của eval mất ý nghĩa.
Crawl lỗi thì giữ nguyên file cũ (nếu có) và báo lỗi ra màn hình.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài viết cần crawl — trang công khai RMIT Vietnam.
# Thứ tự ở đây quyết định tên file (article_01.json ... article_08.json) và phải
# khớp với corpus đã index + golden_dataset.json. Đổi thứ tự = đổi ánh xạ file.
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/students/my-studies/fees-and-payments",
    "https://www.rmit.edu.vn/students/my-studies/enrolment",
    "https://www.rmit.edu.vn/students/my-studies/rights-and-responsibilities",
    "https://www.rmit.edu.vn/students/my-studies/graduation",
    "https://www.rmit.edu.vn/students/my-studies/international-students",
    "https://www.rmit.edu.vn/students/support",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/what-is-consent-and-why-does-it-matter",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/your-voice-matters-help-shape-your-rmit-experience-with-ses",
]


async def crawl_article(url: str) -> dict | None:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
        hoặc None nếu crawl thất bại (bị chặn, timeout, trang rỗng).
    """
    from crawl4ai import AsyncWebCrawler

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success and result.markdown:
                title = url.rstrip("/").split("/")[-1]
                if result.metadata:
                    title = result.metadata.get("title") or title
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": str(result.markdown),
                }
            print("  [WARNING] Trang tra ve rong hoac crawl khong thanh cong.")
    except Exception as e:
        err_msg = str(e).encode("ascii", errors="ignore").decode("ascii")
        print(f"  [ERROR] Crawl that bai: {err_msg}")

    return None


async def crawl_all():
    """
    Crawl toàn bộ bài viết trong ARTICLE_URLS.

    Bài nào crawl lỗi thì GIỮ NGUYÊN file JSON cũ (nếu đã có từ lần chạy trước)
    thay vì ghi đè bằng dữ liệu rỗng/giả.
    """
    setup_directory()

    ok, kept, failed = 0, 0, 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = DATA_DIR / f"article_{i:02d}.json"

        if article:
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  [OK] Saved: {filepath.name} ({len(article['content_markdown'])} chars)")
            ok += 1
        elif filepath.exists():
            print(f"  [KEEP] Giu nguyen ban crawl truoc do: {filepath.name}")
            kept += 1
        else:
            print(f"  [SKIP] Khong co du lieu cho {filepath.name} - can crawl lai.")
            failed += 1

    print(f"\n[TONG KET] Crawl moi: {ok} | Giu ban cu: {kept} | Thieu: {failed}")
    if ok + kept < 5:
        print("[CANH BAO] Task 2 yeu cau toi thieu 5 bai viet trong data/landing/news/.")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARNING] Hay dien ARTICLE_URLS truoc khi chay!")
        print("Goi y: tim trang thong bao/su kien tren trang chinh thuc cua truong dai hoc")
    else:
        asyncio.run(crawl_all())
