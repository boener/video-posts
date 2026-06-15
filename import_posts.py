"""
import_posts.py — merge a JSON export from the site CMS into a pipeline SQLite database.

Usage:
    python import_posts.py export.json
    python import_posts.py export.json --db kombativ
    python import_posts.py export.json --db /path/to/custom.db

The JSON file should be an array of objects produced by export-posts.sql.

For existing posts the content fields (title, paragraphs, meta, url, etc.) are
updated but pipeline-tracking fields (status, error_message, youtube_video_id,
processed_at) are left untouched.

New posts are inserted with status='pending'.
"""

import argparse
import json
import sqlite3
import sys

DB_OPTIONS = {
    'karate':   '/mnt/storage/vertical-posts/data/blog_posts.db',
    'kombativ': '/mnt/storage/vertical-posts/data/blog_posts_kombativ.db',
}

CONTENT_FIELDS = [
    'post_title',
    'opening_paragraph',
    'post_content',
    'meta_title',
    'meta_description',
    'meta_keywords',
    'post_date',
    'post_image',
    'url',
]


def resolve_db(db_arg):
    if db_arg in DB_OPTIONS:
        return DB_OPTIONS[db_arg]
    return db_arg


def ensure_url_column(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(blog_posts)")}
    if 'url' not in cols:
        conn.execute("ALTER TABLE blog_posts ADD COLUMN url TEXT")
        print("Added 'url' column to blog_posts.")


def import_posts(path, db_path):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    # Handle phpMyAdmin JSON export format, which wraps data as:
    # [{type:header}, {type:database}, {type:table, data:[...]}]
    if isinstance(raw, list) and any(isinstance(r, dict) and r.get('type') == 'table' for r in raw):
        posts = []
        for item in raw:
            if isinstance(item, dict) and item.get('type') == 'table':
                posts.extend(item.get('data', []))
    elif isinstance(raw, list):
        posts = raw
    else:
        sys.exit("JSON must be an array of post objects or a phpMyAdmin export.")

    conn = sqlite3.connect(db_path)
    ensure_url_column(conn)

    inserted = updated = skipped = 0

    for post in posts:
        post_id = str(post.get('post_id', '')).strip()
        if not post_id:
            print(f"  SKIP: missing post_id in row {post}")
            skipped += 1
            continue

        existing = conn.execute(
            "SELECT post_id FROM blog_posts WHERE post_id = ?", (post_id,)
        ).fetchone()

        if existing:
            set_clause = ', '.join(f"{f} = ?" for f in CONTENT_FIELDS)
            values = [post.get(f) for f in CONTENT_FIELDS] + [post_id]
            conn.execute(
                f"UPDATE blog_posts SET {set_clause} WHERE post_id = ?",
                values,
            )
            updated += 1
        else:
            all_fields = ['post_id'] + CONTENT_FIELDS + ['status']
            placeholders = ', '.join('?' for _ in all_fields)
            values = [post_id] + [post.get(f) for f in CONTENT_FIELDS] + ['pending']
            conn.execute(
                f"INSERT INTO blog_posts ({', '.join(all_fields)}) VALUES ({placeholders})",
                values,
            )
            inserted += 1

    conn.commit()
    conn.close()
    print(f"Done: {inserted} inserted, {updated} updated, {skipped} skipped.")
    print(f"Database: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Import CMS export JSON into pipeline SQLite DB.")
    parser.add_argument('json_file', help="Path to the exported JSON file")
    parser.add_argument(
        '--db',
        default='karate',
        help="Target database: 'karate' (default), 'kombativ', or a file path",
    )
    args = parser.parse_args()

    db_path = resolve_db(args.db)
    print(f"Importing {args.json_file} → {db_path}")
    import_posts(args.json_file, db_path)


if __name__ == '__main__':
    main()
