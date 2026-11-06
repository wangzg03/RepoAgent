"""CPU-only incremental Python symbol index, literal references and ranked windows."""
import ast
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import ClassVar

from corecoder_mini.runtime import check_cancelled
from corecoder_mini.task_state import atomic_json
from corecoder_mini.tools.base import Tool

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache', '.aws', '.codex'}
SUFFIXES = {'.py', '.js', '.jsx', '.ts', '.tsx', '.go', '.rs', '.java', '.c', '.h', '.cpp', '.cs', '.md', '.toml'}


class Symbols(ast.NodeVisitor):
    def __init__(self):
        self.definitions, self.calls, self.imports, self.scope = [], [], [], []

    def definition(self, node):
        name = '.'.join(self.scope + [node.name])
        self.definitions.append({'name': node.name, 'qualified_name': name, 'start': node.lineno,
                                 'end': node.end_lineno, 'kind': 'class' if isinstance(node, ast.ClassDef) else 'function'})
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = definition
    visit_AsyncFunctionDef = definition
    visit_ClassDef = definition

    def visit_Call(self, node):
        target = ast.unparse(node.func)
        if len(target) < 200:
            self.calls.append({'caller': '.'.join(self.scope) or '<module>', 'callee_text': target,
                               'line': node.lineno, 'resolution': 'syntactic_not_runtime_binding'})
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports.extend({'module': alias.name, 'alias': alias.asname, 'line': node.lineno} for alias in node.names)

    def visit_ImportFrom(self, node):
        self.imports.extend({'module': '.' * node.level + (node.module or ''), 'name': alias.name,
                             'alias': alias.asname, 'line': node.lineno} for alias in node.names)


class CodeIndex:
    def __init__(self, workspace, cache, max_files=5000):
        self.root, self.cache = Path(workspace).resolve(), Path(cache)
        self.max_files, self.lock = max_files, threading.RLock()
        self.records = {}
        if self.cache.exists():
            try:
                data = json.loads(self.cache.read_text(encoding='utf-8'))
                if data.get('root') == str(self.root):
                    self.records = data['files']
            except (OSError, ValueError, KeyError):
                pass

    def refresh(self):
        current, changed, candidates = {}, 0, []
        for parent, dirs, files in os.walk(self.root, followlinks=False):
            check_cancelled()
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.env')
                             and not (Path(parent) / d).is_symlink())
            for name in sorted(files):
                path = Path(parent) / name
                if name.startswith('.env') or path.is_symlink() or path.suffix.lower() not in SUFFIXES:
                    continue
                if path.resolve().is_relative_to(self.root):
                    candidates.append(path)
                if len(candidates) > self.max_files:
                    break
            if len(candidates) > self.max_files:
                break
        candidates.sort()
        for path in candidates[:self.max_files]:
            check_cancelled()
            try:
                if path.stat().st_size > 2**20:
                    continue
                raw = path.read_bytes()
                text = raw.decode('utf-8-sig')
            except (OSError, UnicodeError):
                continue
            name, digest = path.relative_to(self.root).as_posix(), hashlib.sha256(raw).hexdigest()
            old = self.records.get(name)
            if old and old['sha256'] == digest:
                current[name] = old
                continue
            changed += 1
            visitor, error = Symbols(), None
            if path.suffix == '.py':
                try:
                    visitor.visit(ast.parse(text))
                except (SyntaxError, RecursionError):
                    error = 'syntax_error; literal search remains available'
            current[name] = {'sha256': digest, 'definitions': visitor.definitions,
                             'calls': visitor.calls, 'imports': visitor.imports, 'parse_error': error}
        removed = len(set(self.records) - set(current))
        self.records = current
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.cache, {'schema': 1, 'root': str(self.root), 'files': current})
        return {'files': len(current), 'reparsed': changed, 'removed': removed,
                'incomplete': len(candidates) > self.max_files}

    def search(self, query, mode='auto', limit=8, path_prefix=''):
        if mode not in {'auto', 'path', 'definition', 'references', 'calls'} or not query.strip() or not 1 <= limit <= 30:
            raise ValueError('invalid_code_search')
        if path_prefix and (path_prefix.startswith('/') or '..' in path_prefix.split('/') or '\\' in path_prefix):
            raise ValueError('relative_path_prefix_required')
        with self.lock:
            stats = self.refresh()
            hits = []
            query = query.strip()
            terms = list(dict.fromkeys(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', query)))[:16]
            for path, record in sorted(self.records.items()):
                if path_prefix and not (path == path_prefix or path.startswith(path_prefix.rstrip('/') + '/')):
                    continue
                check_cancelled()
                try:
                    raw = (self.root / path).read_bytes()
                    if hashlib.sha256(raw).hexdigest() != record['sha256']:
                        continue
                    lines = raw.decode('utf-8-sig').splitlines()
                except (OSError, UnicodeError):
                    continue
                definitions = {d['start']: d for d in record['definitions']}
                candidates = []
                path_score = 12 * int(query.casefold() in path.casefold()) + sum(t.casefold() in path.casefold() for t in terms)
                if mode in {'auto', 'path'} and path_score:
                    candidates.append((path_score, 1, 12, 'path', '', {'path': path_score}))
                if mode in {'auto', 'definition', 'references'}:
                    for number, line in enumerate(lines, 1):
                        definition = definitions.get(number)
                        literal = int(query in line)
                        matched = sum(bool(re.search(r'\b' + re.escape(t) + r'\b', line)) for t in terms)
                        if not literal and not matched:
                            continue
                        if mode == 'definition' and not definition or mode == 'references' and definition:
                            continue
                        exact = 15 if definition and query in {definition['name'], definition['qualified_name']} else 0
                        evidence = {'path': path_score, 'literal': 5 * literal, 'terms': matched, 'exact_definition': exact}
                        candidates.append((sum(evidence.values()), number, definition['end'] if definition else number + 2,
                                           'definition' if definition else 'text_reference',
                                           definition['qualified_name'] if definition else '', evidence))
                if mode == 'calls':
                    for call in record['calls']:
                        if query in {call['caller'], call['callee_text']} or query in call['callee_text']:
                            candidates.append((20 + path_score, call['line'], call['line'] + 2, 'syntactic_call',
                                               call['caller'], {'call_match': 20, 'path': path_score}))
                for score, start, end, kind, symbol, evidence in sorted(candidates, key=lambda x: (-x[0], x[1]))[:3]:
                    start, end = max(1, start - 2), min(len(lines), end, start + 35)
                    hits.append({'path': path, 'start': start, 'end': end, 'symbol': symbol, 'relation': kind,
                                 'score': score, 'score_components': evidence, 'sha256': record['sha256'],
                                 'text': '\n'.join(f'{n}: {lines[n-1]}' for n in range(start, end + 1))[:6000]})
            unique = {}
            for hit in sorted(hits, key=lambda h: (-h['score'], h['path'], h['start'])):
                unique.setdefault((hit['path'], hit['start'], hit['relation']), hit)
            return {'hits': list(unique.values())[:limit], 'index': stats,
                    'strategy': 'path-ast-definition-literal-reference',
                    'call_analysis': 'syntactic_candidates; dynamic dispatch is unresolved'}


class CodeSearchTool(Tool):
    name = 'code_search'
    read_only = concurrency_safe = True
    description = ('Find paths, Python AST definitions, literal references or syntactic call sites. '
                   'Returns ranked source windows, score components and current SHA256. '
                   'Use read_file for the full relevant function; calls are not a resolved runtime call graph.')
    parameters: ClassVar[dict] = {'type': 'object', 'properties': {
        'query': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
        'mode': {'type': 'string', 'enum': ['auto', 'path', 'definition', 'references', 'calls']},
        'limit': {'type': 'integer', 'minimum': 1, 'maximum': 30},
        'path_prefix': {'type': 'string', 'maxLength': 512}}, 'required': ['query'], 'additionalProperties': False}

    def __init__(self, index):
        self.index = index

    def execute(self, query, mode='auto', limit=8, path_prefix=''):
        return json.dumps(self.index.search(query, mode, limit, path_prefix), ensure_ascii=False)
