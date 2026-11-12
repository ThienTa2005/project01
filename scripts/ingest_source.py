"""Nạp tài liệu web/PDF BỀN VỮNG vào corpus/graph.

Dùng local/CI để thẩm định rồi commit knowledge/extra_sources.json.
Không expose chức năng persist này cho người dùng website công khai.

python -m scripts.ingest_source --url https://www.who.int/news-room/fact-sheets/detail/diabetes
python -m scripts.ingest_source --pdf /path/to/document.pdf
python -m scripts.build_kb
"""
import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from knowledge.ingest import ingest_pdf, ingest_url
from knowledge.engine import ROOT as KNOWLEDGE_ROOT, MAX_EXTRA_TEXT


def main():
    ap = argparse.ArgumentParser(description="Nhập tài liệu KB từ URL cho phép hoặc PDF")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--url')
    src.add_argument('--pdf', type=Path)
    args = ap.parse_args()
    if args.pdf:
        doc = ingest_pdf(args.pdf.read_bytes(), args.pdf.name)
    else:
        doc = ingest_url(args.url)
    f = KNOWLEDGE_ROOT / 'extra_sources.json'
    rows = json.loads(f.read_text(encoding='utf-8')) if f.exists() else []
    if len(rows) >= 15:
        raise SystemExit('Đã đạt tối đa 15 nguồn bổ sung để hạn chế dung lượng deploy.')
    if any(d.get('url') == doc['url'] and doc['url'] for d in rows):
        raise SystemExit('URL đã có trong extra_sources.json; không nhập trùng.')
    doc_id = f"EXTRA-{len(rows)+1:03}"
    rows.append({"id": doc_id, "title": doc['title'], "publisher": "Nguồn bổ sung (chưa kiểm duyệt)",
                 "url": doc['url'], "kind": doc['kind'], "summary": doc['text'][:350],
                 "body": doc['text'][:MAX_EXTRA_TEXT], "curation": "Tự trích entity mention; cần người kiểm duyệt nếu muốn tạo quan hệ y khoa.",
                 "mentions": doc['mentions']})
    f.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Đã thêm', doc_id, 'với', len(doc['mentions']), 'thực thể mention. Chạy python -m scripts.build_kb để xuất lại.')


if __name__ == '__main__':
    main()
