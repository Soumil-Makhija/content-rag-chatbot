"""CLI integration regressions; fixture inference is not real-model validation."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from content_chat.cli import main, read_document
from content_chat.models import ModelUnavailable
from content_chat.store import Store
from tests.test_service import FixtureModel


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / 'index.sqlite3'
        self.model = FixtureModel()

    def invoke(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with patch('content_chat.cli.Ollama', return_value=self.model), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(['--db', str(self.database), *args])
        return status, json.loads(out.getvalue() or err.getvalue())

    def test_cli_import_restart_ask_source_replace_delete(self):
        document = self.root / 'support.md'
        document.write_text('Refunds are available for 30 days.', encoding='utf-8')
        status, metadata = self.invoke('import', 'support', str(document), '--id', 'billing')
        self.assertEqual(status, 0)
        status, answer = self.invoke('ask', 'support', 'What is the policy?')
        self.assertEqual(status, 0)
        source_id = answer['citations'][0]['id']
        self.assertEqual(self.invoke('source', 'support', source_id)[1]['text'], document.read_text())
        document.write_text('Refunds are available for 45 days.', encoding='utf-8')
        status, metadata = self.invoke('import', 'support', str(document), '--id', 'billing')
        self.assertEqual(metadata['version'], 2)
        self.assertEqual(self.invoke('source', 'support', source_id)[0], 2)
        self.assertTrue(self.invoke('delete', 'support', 'billing')[1]['deleted'])
        self.assertEqual(self.invoke('list', 'support'), (0, []))

    def test_subprocess_reads_persisted_database_without_models(self):
        with Store(self.database) as store:
            store.put('support', 'Saved FAQ', 'A persisted fact.', [[1, 0]], 'fixture', 'saved')
        command = [sys.executable, '-m', 'content_chat', '--db', str(self.database), 'show', 'support', 'saved']
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['text'], 'A persisted fact.')

    def test_import_failure_is_json_and_retains_existing_data(self):
        document = self.root / 'faq.txt'; document.write_text('Original.')
        self.assertEqual(self.invoke('import', 'support', str(document), '--id', 'faq')[0], 0)
        document.write_text('Replacement.')
        self.model.embedding_error = ModelUnavailable('Fixture unavailable')
        status, error = self.invoke('import', 'support', str(document), '--id', 'faq')
        self.assertEqual(status, 2); self.assertEqual(error['type'], 'ModelUnavailable')
        self.assertEqual(self.invoke('show', 'support', 'faq')[1]['text'], 'Original.')

    def test_file_types_encoding_binary_and_size_are_checked(self):
        cases = [('bad.pdf', b'%PDF'), ('bad.txt', b'\xff\xfe'), ('nul.md', b'text\0text'), ('large.txt', b'x' * (2 * 1024 * 1024 + 1))]
        for name, raw in cases:
            path = self.root / name; path.write_bytes(raw)
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_document(str(path))
        path = self.root / 'bom.txt'; path.write_bytes(b'\xef\xbb\xbfUnicode caf\xc3\xa9')
        self.assertEqual(read_document(str(path)), 'Unicode café')

    def test_empty_ask_and_search_do_not_require_models(self):
        self.model.embedding_error = ModelUnavailable('Fixture unavailable')
        status, answer = self.invoke('ask', 'empty', 'Any sources?')
        self.assertEqual(status, 0); self.assertFalse(answer['supported'])
        self.assertEqual(self.invoke('search', 'empty', 'Any sources?'), (0, []))

    def test_doctor_reports_missing_models_without_creating_database(self):
        self.model.health = lambda: {'ready': False, 'missing_models': ['fixture']}
        self.assertEqual(self.invoke('doctor')[0], 2)
        self.assertFalse(self.database.exists())
