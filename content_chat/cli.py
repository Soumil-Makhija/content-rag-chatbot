"""Command-line entry point. All data output is JSON; errors exit nonzero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .models import ModelProtocolError, ModelUnavailable, Ollama
from .service import Service
from .store import Store

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


def read_document(path: str) -> str:
    file = Path(path)
    if file.suffix.lower() not in {'.txt', '.md'}:
        raise ValueError('Supported files are UTF-8 .txt and .md documents; PDF/OCR is not supported.')
    with file.open('rb') as stream:
        raw = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError('Document exceeds the 2 MiB limit.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeError as exc:
        raise ValueError('Document must use UTF-8 text encoding.') from exc
    if '\x00' in text:
        raise ValueError('Document contains NUL characters and is not supported text.')
    return text


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Local document chatbot with persistent vectors and cited answers.')
    p.add_argument('--db', default='content-chat.sqlite3', help='SQLite file (default: content-chat.sqlite3)')
    p.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    p.add_argument('--embedding-model', default='nomic-embed-text')
    p.add_argument('--generation-model', default='qwen2.5:1.5b')
    p.add_argument('--timeout', type=float, default=120)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Check local Ollama and installed model names (does not run inference)')
    imp = sub.add_parser('import', help='Embed and import/replace a .md or .txt file')
    imp.add_argument('collection'); imp.add_argument('file'); imp.add_argument('--title'); imp.add_argument('--id')
    docs = sub.add_parser('list', help='List documents without model access'); docs.add_argument('collection')
    for name in ['show', 'delete', 'source']:
        cmd = sub.add_parser(name)
        cmd.add_argument('collection'); cmd.add_argument('id')
    for name in ['ask', 'search']:
        cmd = sub.add_parser(name)
        cmd.add_argument('collection'); cmd.add_argument('question'); cmd.add_argument('--limit', type=int, default=5)
    serve = sub.add_parser('serve', help='Run single-user loopback web UI/API')
    serve.add_argument('--port', type=int, default=8765)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        model = Ollama(args.ollama_url, args.embedding_model, args.generation_model, args.timeout)
        if args.command == 'doctor':
            result = model.health()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result['ready'] else 2
        if args.command == 'serve':
            from .server import serve
            serve(args.db, model, args.port)
            return 0
        with Store(args.db) as store:
            service = Service(store, model)
            if args.command == 'import':
                result = service.import_document(args.collection, args.title or Path(args.file).name,
                                                 read_document(args.file), document_id=args.id)
            elif args.command == 'list':
                result = store.list_documents(args.collection)
            elif args.command == 'show':
                result = store.get_document(args.collection, args.id)
                if result is None:
                    raise ValueError('Document not found in this collection.')
            elif args.command == 'delete':
                result = {'deleted': store.delete(args.collection, args.id)}
            elif args.command == 'source':
                result = store.source(args.collection, args.id)
                if result is None:
                    raise ValueError('Source not found; it may have been replaced or deleted.')
            elif args.command == 'search':
                result = service.search(args.collection, args.question, args.limit)
            else:
                result = service.ask(args.collection, args.question, args.limit)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, sqlite3.Error, ModelUnavailable, ModelProtocolError) as exc:
        print(json.dumps({'error': str(exc), 'type': type(exc).__name__}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
