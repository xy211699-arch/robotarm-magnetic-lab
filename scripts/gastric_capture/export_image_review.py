"""Run INSIDE Isaac Sim. Read-only image/file review export; no capture/Stage calls.
Place this file in the project directory and run it with runpy.run_path.
Creates contact sheets for inspection only; originals stay unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import builtins, csv, hashlib, json, math, threading, traceback, uuid, zipfile
ROOT = Path('/home/multirobo/gastric-single-frame-20261117')
BATCHES = ['collect_small_normal_20261117T184508_a9cdb1', 'collect_small_normal_20261118T074044_e8a930', 'collect_small_normal_20261118T074325_89ad4b', 'collect_small_normal_20261118T075051_0b9c0c', 'collect_small_normal_20261118T081019_916b31', 'collect_small_raised_20261117T181836_4a1c39', 'collect_small_ulcerated_20261117T174135_e86167']

def export_review():
    import numpy as np
    from PIL import Image, ImageDraw
    folder = ROOT/'image_reviews'/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True,exist_ok=False)
    checks=[]; review=[]; assets={}; hash_cache={}
    def digest(path):
        path=Path(path);stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in hash_cache:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            hash_cache[key]=h.hexdigest()
        return hash_cache[key]
    for identity in BATCHES:
        batch=ROOT/'orbit_v6'/identity
        rows=[json.loads(line) for line in (batch/'classification_manifest.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        name=rows[0]['proposed_class'];prefix={'normal':'N','raised':'R','ulcerated':'U'}[name]
        for page in range(0,len(rows),20):
            canvas=Image.new('RGB',(1000,928),'#202020');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),'REVIEW ONLY / '+identity,fill='white')
            for tile,row in enumerate(rows[page:page+20]):
                number=page+tile+1;rid=prefix+'_'+identity[-6:]+'_%03d'%number
                image=batch/row['image'];frame=Path(row['sample_path']).parent
                x,y=(tile%5)*200,28+(tile//5)*225
                item={'review_id':rid,'sample_id':row['sample_id'],'batch_id':identity,'issues':[],'file_checks':{}}
                try:
                    sample=json.loads((frame/'sample.json').read_text(encoding='utf-8-sig'))
                    sums=json.loads((frame/'SHA256SUMS.json').read_text(encoding='utf-8-sig'))
                    for filename in ('image.png','depth_m.npy','instance_ids.npy','capture_script.py','scene_snapshot.usdc','sample.json','config.json','replay_config.json'):
                        p=frame/filename;expected=sums.get(filename,{}).get('sha256')
                        match=p.is_file() and bool(expected) and digest(p)==expected
                        item['file_checks'][filename]=bool(match)
                        if not match:item['issues'].append('missing/hash mismatch: '+filename)
                    item['alias_png_hash_matches']=image.is_file() and digest(image)==row['image_sha256']
                    if not item['alias_png_hash_matches']:item['issues'].append('selected PNG hash mismatch')
                    with Image.open(image) as im:
                        item['image_size']=list(im.size);im.verify()
                    if item['image_size']!=sample['image_resolution']:item['issues'].append('image resolution mismatch')
                    with Image.open(image) as im:
                        raw=im.convert('RGB');thumb=raw.copy();thumb.thumbnail((196,196))
                        canvas.paste(thumb,(x+(200-thumb.width)//2,y))
                        tiny=list(raw.convert('L').resize((17,16)).getdata());value=0
                        for yy in range(16):
                            for xx in range(16):value=(value<<1)|int(tiny[yy*17+xx+1]>tiny[yy*17+xx])
                        item['actual_dhash256']='%064x'%value
                        if value!=int(row['photo_dhash256'],16):item['issues'].append('image fingerprint mismatch')
                    depth=np.load(frame/'depth_m.npy',mmap_mode='r',allow_pickle=False)
                    ids=np.load(frame/'instance_ids.npy',mmap_mode='r',allow_pickle=False)
                    width,height=sample['image_resolution'];cx,cy=width//2,height//2
                    item['depth_shape']=list(depth.shape);item['instance_shape']=list(ids.shape)
                    if list(depth.shape[:2])!=[height,width] or list(ids.shape[:2])!=[height,width]:item['issues'].append('buffer resolution mismatch')
                    center_depth=float(np.asarray(depth[cy,cx]).reshape(-1)[0])*1000
                    item['depth_center_mm']=center_depth if math.isfinite(center_depth) else None
                    item['instance_center_id']=int(np.asarray(ids[cy,cx]).reshape(-1)[0])
                    if item['instance_center_id']!=sample['renderer_occlusion_check']['instance_id']:item['issues'].append('actual center instance disagrees')
                    tolerance=sample['renderer_occlusion_check']['tolerance_m']*1000
                    if not math.isfinite(center_depth) or abs(center_depth-row['distance_mm'])>tolerance:item['issues'].append('actual center depth disagrees')
                    for dependency in sample.get('file_dependencies',[]):
                        key=(dependency.get('path'),dependency.get('sha256'))
                        if key not in assets:
                            p=Path(key[0]);actual=digest(p) if p.is_file() else None
                            assets[key]={'path':key[0],'expected_sha256':key[1],'actual_sha256':actual,'matches':actual==key[1]}
                except Exception as e:
                    item['issues'].append(str(e));draw.text((x+4,y+20),'READ ERROR',fill='#ff7777')
                draw.text((x+4,y+199),rid+' / %.2f mm'%row['distance_mm'],fill='white')
                checks.append(item)
                review.append({'review_id':rid,'sample_id':row['sample_id'],'class_name':name,'image_path':str(image),'decision':'','final_class':'','note':''})
            canvas.save(folder/(identity+'_page%02d.jpg'%(page//20+1)),quality=90)
            print('[REVIEW]',len(checks),'/ 200 images checked and placed on review sheets...')
    summary={'scope':'actual saved file hashes, image decode and saved center depth; human semantic review and GPU replay still pending',
             'images':len(checks),'images_with_file_issues':sum(bool(c['issues']) for c in checks),'checks':checks,'external_assets':list(assets.values()),'raw_images_modified':False}
    summary['external_assets_with_issues']=sum(not a['matches'] for a in assets.values())
    (folder/'image_integrity_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (folder/'label_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(review[0]));w.writeheader();w.writerows(review)
    archive=folder/'image_review_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p!=archive:z.write(p,p.name)
    print('[REVIEW] DONE:',archive)
    print('[REVIEW] Image/file issues:',summary['images_with_file_issues'],'/ 200. Review labels manually.')
    print('[REVIEW] External assets with issues:',summary['external_assets_with_issues'])

def main():
    previous=getattr(builtins,'_gastric_image_review_thread',None)
    if previous is not None and previous.is_alive():print('[REVIEW] Export already running.');return
    def work():
        try:export_review()
        except Exception:print('[REVIEW] ERROR:\n'+traceback.format_exc())
    worker=threading.Thread(target=work,name='gastric-image-review',daemon=True)
    setattr(builtins,'_gastric_image_review_thread',worker);worker.start()
    print('[REVIEW] Background export started. Raw images stay unchanged; no capture started.')

if __name__ in ('__main__','<run_path>'):main()
