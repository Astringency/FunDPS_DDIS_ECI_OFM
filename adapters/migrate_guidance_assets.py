"""Export only required guidance inputs/results and verify a relocated copy."""
import argparse
import hashlib
import json
from pathlib import Path

OLD = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
REL = Path('diagnostics/fm_ofm_strength_refine_20260922')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def export(runtime):
    files = {REL / 'plan.json', REL / 'migration_197_193.json', Path('data/shared_prior_assets/manifest.json')}
    assets = read(OLD / 'data/shared_prior_assets/manifest.json')
    pdes = {g.split('/')[0] for g in runtime['groups']}
    for group in runtime['groups']:
        for split in ('id', 'smooth', 'rough'):
            ref = Path('evaluation_v2/ofm/fm4pde') / group / split / 'shard_000/run.json'
            files.add(ref)
            files.add(Path(read(OLD / ref)['checkpoint']).relative_to(OLD))
        folder = OLD / REL / group
        for run in folder.glob('*/*/*/run.json'):
            assert (run.parent / 'verified_summary.json').exists(), run
        files.update(p.relative_to(OLD) for p in folder.rglob('*') if p.is_file() and p.suffix != '.lock')
    for pde in pdes:
        folder = OLD / REL / 'validation' / pde
        assert (folder / 'ready.json').exists()
        files.update(p.relative_to(OLD) for p in folder.rglob('*') if p.is_file() and p.suffix != '.lock')
        compact = Path('data/compact') / pde
        files.add(compact / 'manifest.json')
        for split, record in assets['pdes'][pde]['splits'].items():
            files.add((OLD / 'data/shared_prior_assets' / record['file']).relative_to(OLD))
            files.add(compact / f'{split}.npy')
            files.add(compact / f'{split}_ids.npy')
    inventory = {str(p): dict(bytes=(OLD / p).stat().st_size, sha256=sha(OLD / p)) for p in sorted(files)}
    record = REL / ('migration_inventory_' + runtime['host'] + '.json')
    write(OLD / record, dict(source_root=str(OLD), runtime=runtime, files=inventory))
    files.add(record)
    listing = OLD / REL / ('migration_files_' + runtime['host'] + '.txt')
    listing.write_text(''.join(str(p) + '\n' for p in sorted(files)))
    print(json.dumps(dict(listing=str(listing), files=len(files), bytes=sum(r['bytes'] for r in inventory.values()))))


def localize(runtime):
    root = Path(runtime['root'])
    record = read(root / REL / ('migration_inventory_' + runtime['host'] + '.json'))
    audit_path = root / REL / 'migration_verified.json'
    if audit_path.exists():
        raise RuntimeError('Migration already verified; do not rewrite active manifests')
    for name, r in record['files'].items():
        p = root / name
        assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], p
    manifests = []
    for pde in {g.split('/')[0] for g in runtime['groups']}:
        folder = root / REL / 'validation' / pde
        p = folder / 'assets/manifest.json'
        manifest = read(p)
        original_sha = sha(p)
        backup = p.with_name('manifest.server216.json')
        assert not backup.exists()
        backup.write_bytes(p.read_bytes())
        for split, e in manifest['pdes'][pde]['splits'].items():
            assert sha(folder / 'assets' / f'{split}.pt') == e['sha256']
            e['file'] = f'{split}.pt'
        write(p, manifest)
        ready = read(folder / 'ready.json')
        ready.update(original_manifest_sha256=original_sha, manifest_sha256=sha(p))
        write(folder / 'ready.json', ready)
        manifests.append(dict(path=str(p), original_sha256=original_sha, relocated_sha256=sha(p)))
    write(audit_path, dict(files_verified=len(record['files']), runtime=runtime, manifests=manifests,
        retained_trials='Completed trials are byte-identical copies with original server216 provenance; only new trials use relocated manifests.'))
    print(json.dumps(dict(files_verified=len(record['files']), host=runtime['host'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--mode', choices=['export', 'localize'], required=True)
    a = p.parse_args()
    (export if a.mode == 'export' else localize)(read(a.runtime))
