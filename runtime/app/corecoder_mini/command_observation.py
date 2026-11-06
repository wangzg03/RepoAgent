"""Conservative observation classification. This never authorizes or executes a shell."""

import ast
import hashlib
import json
import re
import shlex
from pathlib import Path


def _literal_heredoc(command):
    match = re.search(r"<<(?P<tabs>-?)\s*(?P<quote>['\"]?)(?P<marker>[A-Za-z_]\w*)"
                      r"(?P=quote)[ \t]*\n(?P<body>.*?)\n(?P=marker)[ \t]*(?:\n|$)", command, re.DOTALL)
    if not match:
        return command
    head = command[:match.start()].rstrip()
    # Only translate a literal Python stdin program, never a file-writing heredoc.
    if not match['quote'] or not re.search(r"(?:^|\s)(?:\S*/)?python(?:3(?:\.\d+)?)?\s+-?$", head):
        return command
    code = match['body']
    if match['tabs']:
        code = '\n'.join(line.lstrip('\t') for line in code.splitlines())
    return re.sub(r"\s+-?$", "", head) + " -c " + shlex.quote(code) + "\n" + command[match.end():]


def shell_steps(command, workspace=None, *, inspection_pipeline=False):
    """Observe plain commands and /dev/null redirects, never authorize execution.

    Pipelines are parsed only for a separate read-only inspection classifier.
    Checks deliberately reject them: a pipe's exit status can mask a failed test.
    """
    lexer = shlex.shlex(_literal_heredoc(command), posix=True, punctuation_chars=';&|<>\n')
    lexer.whitespace = ' \t\r'
    lexer.whitespace_split = True
    lexer.commenters = '#'
    try:
        tokens = []
        for token in lexer:
            # shlex groups adjacent shell punctuation, including a trailing ;\n.
            if token and set(token) <= set(';&|<>\n'):
                tokens.extend(re.findall(r'&&|\|\||>>|<<|[;&|<>\n]', token))
            else:
                tokens.append(token)
    except ValueError:
        return []
    cleaned, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if token in {'>', '>>'} and tokens[index + 1:index + 2] == ['/dev/null']:
            # A numeric fd belongs to the redirect, not the command arguments.
            if cleaned and cleaned[-1] in {'1', '2'}:
                cleaned.pop()
            index += 2
            continue
        cleaned.append(token)
        index += 1
    tokens = cleaned
    forbidden = {'||', '&', '>', '>>', '<', '<<'} | (set() if inspection_pipeline else {'|'})
    if any(t in forbidden for t in tokens):
        return []
    chunks, chunk = [], []
    for token in tokens + [';']:
        if token in {';', '&&', '|'} or token and set(token) == {'\n'}:
            if chunk:
                chunks.append((chunk, token))
                chunk = []
        elif token in {'(', ')', 'if', 'then', 'fi', 'for', 'do', 'done', 'while'}:
            return []
        else:
            chunk.append(token)
    cwd = Path(workspace).resolve() if workspace else None
    rows = []
    previous = None
    for args, connector in chunks:
        piped = previous == '|' or connector == '|'
        previous = connector
        while args and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', args[0], re.DOTALL):
            args = args[1:]
        if not args:
            continue
        if args[0] == 'export':
            if piped:
                return []
            continue
        if args[0] == 'cd':
            if piped or len(args) != 2 or not cwd or any(c in args[1] for c in '$`~'):
                return []
            path = args[1]
            target = Path(workspace) / path.removeprefix('/workspace').lstrip('/') if path == '/workspace' or path.startswith('/workspace/') else cwd / path
            try:
                target = target.resolve()
            except (OSError, ValueError):
                return []
            if not target.is_relative_to(Path(workspace).resolve()):
                return []
            cwd = target
            continue
        if args[0] == 'env':
            args = args[1:]
            while args and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', args[0], re.DOTALL):
                args = args[1:]
            if not args or args[0].startswith('-'):
                return []
        rows.append((args, cwd))
    return rows


def _inspection_rows(command, workspace=None):
    """Recognize source-reading pipelines without mistaking arbitrary xargs for reads."""
    if '$(' in command or '`' in command:
        return []
    rows = shell_steps(command, workspace, inspection_pipeline=True)
    inspections = {'cat', 'head', 'tail', 'wc', 'ls', 'find', 'rg', 'grep', 'pwd'}
    result = []
    for args, cwd in rows:
        program = args[0].rsplit('/', 1)[-1]
        if program == 'xargs':
            child = 1
            while child < len(args) and args[child] in {'-0', '-r', '--null', '--no-run-if-empty'}:
                child += 1
            if child == len(args) or args[child].rsplit('/', 1)[-1] not in {'grep', 'rg'}:
                return []
            args, program = args[child:], args[child].rsplit('/', 1)[-1]
        if program == 'git' and len(args) > 1 and args[1] in {'log', 'status', 'show', 'diff', 'grep', 'rev-parse'}:
            result.append((args, cwd))
            continue
        if program == 'find' and any(a in {'-exec', '-execdir', '-ok', '-okdir', '-delete'}
                                     or a.startswith(('-fprint', '-fls')) for a in args[1:]):
            return []
        if program == 'rg' and any(a == '--pre' or a.startswith('--pre=') for a in args[1:]):
            return []
        if program == 'sed' and '|' not in command and not any(a.startswith(('-i', '--in-place')) for a in args[1:]):
            result.append((args, cwd))
            continue
        if program not in inspections:
            return []
        result.append((args, cwd))
    return result


def _python_tree(args, cwd, workspace):
    if any(a in {'-O', '-OO'} for a in args[1:]):
        return None
    code = None
    if '-c' in args:
        index = args.index('-c') + 1
        code = args[index] if index < len(args) else None
    elif cwd and len(args) > 1 and args[1].endswith('.py'):
        try:
            path = (cwd / args[1]).resolve()
            if path.is_relative_to(Path(workspace).resolve()) and path.is_file() and path.stat().st_size <= 128000:
                code = path.read_bytes()
        except (OSError, ValueError):
            return None
    try:
        return ast.parse(code) if code is not None else None
    except (SyntaxError, UnicodeError):
        return None


def check_command(command, workspace=None):
    """A real executable check candidate, never a claim that the task is correct."""
    rows = shell_steps(command, workspace)
    if not rows:
        return False
    args, cwd = rows[-1]
    program = args[0].rsplit('/', 1)[-1]
    if program in {'pytest', 'py.test'} or args[0] in {'bin/test', './bin/test'}:
        return True
    if program not in {'python', 'python3'} or any(a in {'-O', '-OO'} for a in args[1:]):
        return False
    if '-m' in args:
        index = args.index('-m') + 1
        return index < len(args) and args[index] in {'pytest', 'unittest'}
    if len(args) > 1 and args[1] in {'bin/test', './bin/test'}:
        return True
    tree = _python_tree(args, cwd, workspace)
    return tree is not None and any(
        isinstance(node, ast.Assert) or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id in {'sympy', 'pytest', 'unittest'}
        and node.func.attr in {'test', 'main'} for node in ast.walk(tree))


def observation_kind(command, workspace=None):
    """Inspection failures are diagnostic data; actual check failures remain blocking."""
    if check_command(command, workspace):
        return 'check'
    rows = _inspection_rows(command, workspace)
    if not rows:
        return 'execution'
    programs = [args[0].rsplit('/', 1)[-1] for args, _ in rows]
    return 'search' if any(p in {'rg', 'grep'} for p in programs) else 'inspection'


def search_result_kind(command, returncode, workspace=None):
    """Only a terminal grep/rg owns the documented no-match exit code 1."""
    rows = shell_steps(command, workspace, inspection_pipeline=True)
    terminal = rows[-1][0][0].rsplit('/', 1)[-1] if rows else None
    return 'search_no_match' if returncode == 1 and terminal in {'grep', 'rg'} else 'search_error'


def command_identity(command, workspace):
    """Do not count spacing, comments or a redundant cd as new information."""
    rows = shell_steps(command, workspace)
    normalized = []
    for args, cwd in rows:
        args = list(args)
        if args[0].rsplit('/', 1)[-1] in {'python', 'python3'} and '-c' in args:
            index = args.index('-c') + 1
            if index < len(args):
                try:
                    args[index] = ast.dump(ast.parse(args[index]))
                except SyntaxError:
                    pass
        normalized.append([args, str(cwd)])
    value = json.dumps(normalized, sort_keys=True) if rows else command
    return hashlib.sha256(value.encode()).hexdigest()


def unavailable_git_history(command, result, workspace):
    """An observed git query failure can precede a probe in an && batch."""
    if check_command(command, workspace) or not re.search(
            r'^fatal: not a git repository\b', result.get('output', ''), re.MULTILINE | re.IGNORECASE):
        return False
    return any(args[0].rsplit('/', 1)[-1] == 'git' and len(args) > 1
               and args[1] in {'log', 'status', 'show', 'diff', 'grep', 'rev-parse'}
               for args, _ in shell_steps(command, workspace))


def command_targets(command, workspace):
    """Only names of existing tracked-looking files, never inferred source contents."""
    rows = shell_steps(command, workspace) or _inspection_rows(command, workspace)
    paths = set()
    for args, cwd in rows:
        if cwd is None:
            continue
        for arg in args[1:]:
            if arg.startswith('-') or '\n' in arg or len(arg) > 240:
                continue
            try:
                path = (cwd / arg).resolve()
                if path.is_relative_to(Path(workspace).resolve()) and path.is_file():
                    paths.add(path.relative_to(Path(workspace).resolve()).as_posix())
            except (OSError, ValueError):
                continue
    return sorted(paths)[:8]
