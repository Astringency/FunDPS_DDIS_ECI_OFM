"""Read FM4PDE metrics from the active LaTeX main tables, including wrapped rows."""
import math
import re

PDES = ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')
SPLITS = ('id', 'smooth', 'rough')


def _without_comments(raw):
    lines = []
    for line in raw.splitlines():
        for match in re.finditer('%', line):
            preceding = line[:match.start()]
            slashes = len(preceding) - len(preceding.rstrip('\\'))
            if slashes % 2 == 0:
                line = preceding
                break
        lines.append(line)
    return '\n'.join(lines)


def _braced(text, opening):
    depth, index = 1, opening + 1
    while index < len(text):
        if text[index] == '\\':
            index += 2
            continue
        depth += (text[index] == '{') - (text[index] == '}')
        if depth == 0:
            return text[opening + 1:index], index + 1
        index += 1
    raise ValueError('Unclosed LaTeX brace')


def _rows(text, offset):
    """Split on top-level & and \\, preserving offsets into the source file."""
    cells, start, index, depth = [], 0, 0, 0

    def cell(end):
        value = text[start:end]
        return value.strip(), offset + start + len(value) - len(value.lstrip())

    while index < len(text):
        if text.startswith('\\\\', index) and depth == 0:
            cells.append(cell(index))
            yield cells
            cells = []
            index += 2
            spacing = re.match(r'\*?(?:\[[^\]]*\])?', text[index:])
            index += spacing.end()
            start = index
        elif text[index] == '\\':
            index += 2  # Includes escaped &, { and }, and nested shortstack breaks.
        elif text[index] == '&' and depth == 0:
            cells.append(cell(index))
            index += 1
            start = index
        else:
            depth += (text[index] == '{') - (text[index] == '}')
            index += 1
    if depth != 0:
        raise ValueError('Unbalanced braces inside manuscript table')


def _plain(cell):
    value = re.sub(r'\\[A-Za-z@]+\*?(?:\[[^\]]*\])?', '', cell)
    return ' '.join(re.sub(r'[{}$]', '', value).split())


def _mean_std(cell):
    parts = re.split(r'\\pm(?![A-Za-z])|±', cell)
    if len(parts) != 2:
        raise ValueError(f'Expected mean \\pm standard deviation, got {cell!r}')
    values = []
    for part in parts:
        # Ranking markers/footnotes are not part of a numeric estimate.
        part = re.sub(r'\^(?:\{[^{}]*\}|\\[A-Za-z]+|[^\s{}])', '', part)
        numbers = re.findall(r'[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', part)
        if len(numbers) != 1:
            raise ValueError(f'Ambiguous or missing numeric estimate: {cell!r}')
        value = float(numbers[0].replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid relative-error estimate: {cell!r}')
        values.append(value)
    return values


def parse_paper(path):
    clean = _without_comments(path.read_text())

    def table(label):
        candidates = [m for m in re.finditer(r'\\begin\{(table\*?)\}.*?\\end\{\1\}', clean, re.S)
            if re.search(r'\\label\{\s*' + re.escape(label) + r'\s*\}', m.group())]
        if len(candidates) != 1:
            raise ValueError(f'{path}: expected one active table with label {label}, found {len(candidates)}')
        match = candidates[0]
        block = match.group()
        caption = re.search(r'\\caption(?:\[[^\]]*\])?\s*\{', block)
        if caption is None:
            raise ValueError(f'{path}: {label}: missing caption with test sample count')
        caption_text, _ = _braced(block, caption.end() - 1)
        counts = re.findall(r'\b(\d[\d,]*)\s+test\s+(?:inputs|cases|samples)\b', _plain(caption_text), re.I)
        if len(counts) != 1 or int(counts[0].replace(',', '')) <= 0:
            raise ValueError(f'{path}: {label}: cannot identify sample count in caption {caption_text!r}')
        count = int(counts[0].replace(',', ''))
        begin = re.search(r'\\begin\{tabular\}\s*\{', block)
        if begin is None:
            raise ValueError(f'{path}: {label}: missing tabular environment')
        _, start = _braced(block, begin.end() - 1)
        end = block.find('\\end{tabular}', start)
        if end < 0:
            raise ValueError(f'{path}: {label}: missing end of tabular')
        return list(_rows(block[start:end], match.start() + start)), count

    def header(rows, required, label):
        for index, cells in enumerate(rows):
            names = [_plain(c[0]) for c in cells]
            if all(names.count(name) == 1 for name in required):
                return index, {name: names.index(name) for name in required}
        raise ValueError(f'{path}: {label}: cannot uniquely identify columns {required}')

    def row(pde, task, split, cell, count, label):
        value, offset = cell
        line = clean.count('\n', 0, offset) + 1
        try:
            mean, sd = _mean_std(value)
        except ValueError as error:
            raise ValueError(f'{path}:{line}: {label}, FM4PDE {pde}/{task}/{split}: {error}') from error
        return dict(method='FM-FM', pde=pde, task=task, split=split, n=count,
            expected_n=count, n_finite=None, n_failed=None, mean_percent=mean,
            finite_mean_percent=mean, finite_std_percent=sd, status='paper_reference',
            source='manuscript_main_table', source_line=line, source_table=label)

    results = []
    for task in ('forward', 'inverse'):
        label = 'tab:sparse-' + task + '-results'
        rows, count = table(label)
        first, columns = header(rows, ['PDE', 'Distribution', 'FM4PDE'], label)
        pde = None
        for cells in rows[first + 1:]:
            if len(cells) <= columns['Distribution']:
                continue
            split = _plain(cells[columns['Distribution']][0]).lower()
            if split not in SPLITS:
                continue
            if len(cells) <= max(columns.values()):
                raise ValueError(f'{path}: {label}: incomplete data row for {split}')
            name = _plain(cells[columns['PDE']][0])
            if name:
                matches = [value for token, value in [('Poisson', 'poisson'), ('Helmholtz', 'helmholtz'),
                    ('Darcy', 'darcy'), ('Navier--Stokes', 'nsnonbounded')] if token in name]
                if len(matches) != 1:
                    raise ValueError(f'{path}: {label}: unknown PDE row {name!r}')
                pde = matches[0]
            if pde is None:
                raise ValueError(f'{path}: {label}: distribution precedes PDE name')
            results.append(row(pde, task, split, cells[columns['FM4PDE']], count, label))

    label = 'tab:burgers-results'
    rows, count = table(label)
    first, columns = header(rows, ['Observations', 'Method', 'ID', 'Smooth', 'Rough'], label)
    observations = None
    for cells in rows[first + 1:]:
        if len(cells) <= max(columns['Observations'], columns['Method']):
            continue
        name = _plain(cells[columns['Observations']][0])
        if name:
            observations = 'Random' if 'Random' in name else 'Structured' if 'Structured' in name else None
        method = _plain(cells[columns['Method']][0])
        if observations != 'Random' or not re.fullmatch(r'FM4PDE\s*\(100\s+steps\)', method):
            continue
        if len(cells) <= max(columns.values()):
            raise ValueError(f'{path}: {label}: incomplete FM4PDE Random row')
        for split in SPLITS:
            results.append(row('burger', 'both', split, cells[columns[split.title() if split != 'id' else 'ID']], count, label))
    expected = {(p, t, s) for p in PDES for t in (('both',) if p == 'burger' else ('forward', 'inverse')) for s in SPLITS}
    if len(results) != 27 or {(r['pde'], r['task'], r['split']) for r in results} != expected:
        raise ValueError(f'{path}: manuscript does not contain exactly the expected 27 settings')
    return results
