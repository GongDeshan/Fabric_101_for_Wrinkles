import ast
def assign(text, replacements):
    """Replace top-level assignments only, preserving the rest byte-for-byte."""
    lines = text.splitlines(keepends=True)
    edits, found = [], set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                edits.append((node.lineno-1, node.end_lineno, f'{name} = {replacements[name]}\n'))
                found.add(name)
    if set(replacements) - found:
        raise ValueError(f'Missing assignments: {set(replacements)-found}')
    for first, last, replacement in reversed(edits):
        lines[first:last] = [replacement]
    return ''.join(lines)
