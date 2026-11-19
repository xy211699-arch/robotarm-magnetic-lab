#!/usr/bin/env python3
"""Archive audited evidence with zstd; verify every member without extracting.
Requires the audit_delivery output and the zstd command. No source mutation.
"""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', default='gastric-candidates-20261119-v1')
    p.add_argument('--volume-mib', type=int, default=1500)
    args = p.parse_args()
    source, audit, repo, out = [x.resolve() for x in (args.source,args.audit,args.repo,args.out)]
    if source == out or source in out.parents or audit == out or audit in out.parents:
        p.error('package output must be outside source and audit directories')
    out.mkdir(parents=True, exist_ok=True)
    archive = out/(args.name+'.tar.zst')
    if archive.exists():
        p.error('archive exists; use a new version or output directory')
    entries, expected = [], {}
    import csv
    with (audit/'all_file_index.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            arc = 'capture_source/'+row['path']
            entries.append((source/row['path'],arc))
            expected[arc] = {'bytes':int(row['bytes']), 'sha256':row['sha256']}
    for folder, prefix in ((audit,'audit'),(repo/'scripts/gastric_capture','repository_context/scripts/gastric_capture')):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                arc = prefix+'/'+path.relative_to(folder).as_posix()
                entries.append((path,arc))
                expected[arc] = {'bytes':path.stat().st_size,'sha256':digest(path)}
    with archive.open('xb') as output:
        compressor = subprocess.Popen(['zstd','-T2','-6','--long=27','-q','-c'],stdin=subprocess.PIPE,stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for index,(path,arc) in enumerate(entries,1):
                    tar.add(path,arcname=arc,recursive=False)
                    if index % 500 == 0:
                        print('Archived',index,'/',len(entries),flush=True)
            compressor.stdin.close()
            if compressor.wait() != 0:
                raise RuntimeError('zstd compression failed')
        finally:
            if compressor.poll() is None:
                compressor.terminate()
                compressor.wait()
    # Verify compressed integrity AND each uncompressed member against audited SHA-256.
    subprocess.run(['zstd','-t','-q',str(archive)],check=True)
    decoder = subprocess.Popen(['zstd','-d','-q','-c',str(archive)],stdout=subprocess.PIPE)
    seen, problems = set(), []
    with tarfile.open(fileobj=decoder.stdout,mode='r|') as tar:
        for member in tar:
            if not member.isfile():
                problems.append({'path':member.name,'error':'unexpected non-file member'})
                continue
            info = expected.get(member.name)
            h = hashlib.sha256()
            f = tar.extractfile(member)
            for block in iter(lambda:f.read(4*1024*1024),b''):
                h.update(block)
            if member.name in seen or not info or info['bytes'] != member.size or info['sha256'] != h.hexdigest():
                problems.append({'path':member.name,'error':'member identity mismatch or duplicate'})
            seen.add(member.name)
    decoder.stdout.close()
    if decoder.wait() != 0:
        raise RuntimeError('archive decompression failed')
    problems.extend({'path':name,'error':'missing member'} for name in expected.keys()-seen)
    if problems:
        (out/'archive_verification_failure.json').write_text(json.dumps(problems,indent=2)+'\n')
        raise RuntimeError('archive member verification failed')
    whole = {'name':archive.name,'bytes':archive.stat().st_size,'sha256':digest(archive)}
    limit = args.volume_mib*1024*1024
    if limit <= 0 or limit >= 2*1024**3:
        raise ValueError('volume size must be positive and strictly below 2 GiB')
    parts = []
    if whole['bytes'] > limit:
        with archive.open('rb') as f:
            part_index=1
            while True:
                block=f.read(limit)
                if not block:
                    break
                part=out/(archive.name+'.part-%03d'%part_index)
                with part.open('xb') as output:
                    output.write(block)
                parts.append({'name':part.name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()})
                part_index+=1
        rebuilt=hashlib.sha256()
        for item in parts:
            with (out/item['name']).open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):
                    rebuilt.update(block)
        if rebuilt.hexdigest() != whole['sha256']:
            raise RuntimeError('volume concatenation hash mismatch')
    else:
        parts=[whole]
    manifest={'schema':'gastric_package_v1','full_archive':whole,'download_assets':parts,
              'member_count':len(seen),'source_file_count':sum(name.startswith('capture_source/') for name in seen),
              'member_bytes':sum(item['bytes'] for item in expected.values()),
              'verification':'zstd test plus every extracted member SHA-256 and byte count matched; no extraction to disk',
              'member_manifest_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
              'restore':'If split: concatenate .part-001, .part-002, ... in numeric order to the full_archive name. Verify full SHA-256, then zstd -dc ARCHIVE | tar -xf -.'}
    (out/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'archive_member_index.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'SHA256SUMS').open('w') as f:
        for item in parts:
            f.write(item['sha256']+'  '+item['name']+'\n')
        if len(parts)>1:
            f.write(whole['sha256']+'  '+whole['name']+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
