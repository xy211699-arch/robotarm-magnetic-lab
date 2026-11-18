# Run INSIDE Isaac Sim on a user-confirmed normal scout frame. Collects 20, not 100 from one spot.
from pathlib import Path
import ast, builtins, runpy
HOME_DIR = Path('/home/multirobo/gastric-single-frame-20261117')
PATCH = r'''
def normal_completed(geometry_hash):
    result = []
    seen = set()
    for file in sorted(ORBIT_HOME.glob('collect_small_normal_*/collection_summary.json')):
        summary = json.loads(file.read_text(encoding='utf-8'))
        config = json.loads((file.parent/'collection_config.json').read_text(encoding='utf-8'))
        rows = summary.get('images',[])
        identity = config['region']['region_id']
        if identity in seen or config.get('batch_id')!=file.parent.name or not summary.get('quota_met') or len(rows)!=20 or config['region']['geometry_sha256']!=geometry_hash:
            continue
        if not all(r.get('proposed_class_id')==0 and not r.get('abnormal',True) and
                   (file.parent/r['image']).is_file() and hashlib.sha256((file.parent/r['image']).read_bytes()).hexdigest()==r['image_sha256'] for r in rows):
            continue
        result.append((file.parent,config,summary))
        seen.add(identity)
    return result

def normal_framing(candidate,cloud,units,resolution,fov_deg):
    x,y = candidate['target_pixel_xy']
    candidate['framing_ok'] = bool(.15*resolution[0]<=x<=.85*resolution[0] and .15*resolution[1]<=y<=.85*resolution[1])
    candidate['surface_framing'] = {'method':'normal seed center plus whole-view known-lesion exclusion','semantic_review':'pending'}
    return candidate

class GastricNormalViews(GastricSmallViews):
    async def load_latest(self):
        path = Path(NORMAL_SOURCE)
        self.set_sample(json.loads(path.read_text(encoding='utf-8')),path.parent)
        self.x.set_value(NORMAL_CIRCLE[0]); self.y.set_value(NORMAL_CIRCLE[1]); self.radius.set_value(NORMAL_CIRCLE[2])
        self.message('NORMAL ready. This patch: 20 images; 5 different patches: 100 total.')

    async def save_region(self,name='normal'):
        if name!='normal': raise ValueError('This window is for normal patches only')
        await self.get_query()
        point = (await self.surface_pixel([self.x.as_int,self.y.as_int]))[0]
        for _,config,_ in normal_completed(self.query_hash):
            if np.linalg.norm(point-np.asarray(config['region']['center_world_m'])) < .020:
                raise ValueError('This patch is too close to an already collected patch. Choose another wall area (>20 mm).')
        await super().save_region('normal')
        self.message('Saved NORMAL patch. Ready for 20 small-angle photos.')

    def check_candidate(self,region,candidate):
        reason = super().check_candidate(region,candidate)
        if reason: return reason
        eye = np.asarray(candidate['eye_m'])
        for name in ('raised','ulcerated'):
            lesion = self.targets.get(name)
            if not lesion or lesion['geometry_sha256']!=self.query_hash:
                raise ValueError('Both lesion regions must be calibrated on the same geometry first')
            points = np.asarray([lesion['center_world_m']]+lesion['landmark_points_world_m'])
            xy,z = orbit_project(points,candidate['matrix'],self.sample['meters_per_unit'],[801,801],90)
            for point,pixel,depth in zip(points,xy,z):
                if .0002<depth<.3 and np.isfinite(pixel).all() and (pixel>=-12).all() and (pixel<=812).all():
                    hits = self.query.hits(eye,point-eye)
                    if len(hits) and abs(hits[0]-np.linalg.norm(point-eye))<=.001:
                        return 'known_'+name+'_region_visible_in_normal_view'
        return None

    def target_render_check(self,record,region):
        result = super().target_render_check(record,region)
        if result['passed'] and self.prior_normal_hashes:
            digest = orbit_photo_fingerprint(record['image_path'])
            if min((digest^h).bit_count() for h in self.prior_normal_hashes)<4:
                result.update(passed=False,reason='near_duplicate_of_previous_normal_patch')
        return result

    async def collect_normal_patch(self):
        self.stop_requested=False
        await self.get_query()
        completed = normal_completed(self.query_hash)
        if len(completed)>=5:
            self.message('NORMAL TOTAL: 100 / 100 already saved. No new batch started.'); return
        self.prior_normal_hashes = [int(r['photo_dhash256'],16) for _,_,s in completed for r in s['images']]
        await self.save_region('normal')
        await super().collect('normal',20)
        completed = normal_completed(self.query_hash)
        orbit_atomic_json(ORBIT_HOME/'normal_dataset_progress.json',
            {'target_count':100,'completed_patches':len(completed),'saved_images':20*len(completed),
             'labels_pending_review':True,'training_split_exported':False,'batches':[str(p) for p,_,_ in completed]})
        print('[NORMAL] TOTAL:',20*len(completed),'/ 100; patches:',len(completed),'/ 5; view review pending')

orbit_photo_framing = normal_framing
GastricOrbitPanel = GastricNormalViews
_normal_panel = gastric_orbit_setup()
async def _normal_start():
    if _normal_panel.task is not None: await _normal_panel.task
    if _normal_panel.sample is not None and not _normal_panel.stop_requested:
        _normal_panel.start(_normal_panel.collect_normal_patch)
asyncio.ensure_future(_normal_start())
'''

def build_normal_source(base,source_path,circle):
    if 'class GastricSmallViews' not in base or base.count('_small_panel = gastric_orbit_setup()')!=1:
        raise ValueError('Complete small-angle V9 engine required')
    text = base.split('_small_panel = gastric_orbit_setup()')[0]
    changes = [
      ("if name not in ('raised','ulcerated') or count != 50:","if name not in ('raised','ulcerated','normal') or count != (20 if name=='normal' else 50):"),
      ('This entry collects exactly 50 images for one confirmed lesion class','Normal patches require 20 images; lesion classes require 50'),
      ("cloud,_,cloud_info=await asyncio.to_thread(self.load_region_cloud,region)\n        self.stop_requested=False", "cloud,_,cloud_info=await asyncio.to_thread(self.load_region_cloud,region)\n        if self.stop_requested: return"),
      ("'abnormal':True", "'abnormal':name!='normal'"),
      ("target 50 distinct usable photos...", "target '+str(count)+' distinct usable photos..."),
      ("SMALL VIEW %s: %d / 50 saved; set %d; pose %d...", "SMALL VIEW %s: %d saved; set %d; pose %d..."),
      ("'COLLECT %s: %d / 50 distinct usable photos saved.'%(name,len(rows))", "'COLLECT %s: %d / %d photos saved.'%(name,len(rows),count)"),
      ("'%s: %s %d / 50 photos saved; pending view review. %s'%(", "'%s: %s %d / %d photos saved; pending view review. %s'%("),
      ("label,name,len(rows),root)", "label,name,len(rows),count,root)"),
      ('Gastric | Small Views: 50 photos','Gastric | Normal Wall: 20 per patch'),
    ]
    for old,new in changes:
        if old not in text: raise ValueError('Engine layout differs: '+old[:60])
        text=text.replace(old,new)
    left=text.index("                        ui.Button('3 Save RAISED")
    right=text.index("                        ui.Label('X/Y/R only",left)
    text=text[:left]+"                        ui.Button('Collect NORMAL: 20 photos', height=36, clicked_fn=lambda: self.start(self.collect_normal_patch))\n"+text[right:]
    left=text.index('                with ui.HStack(height=30, spacing=5):')
    right=text.index('        self.start(self.load_latest)',left)
    text=text[:left]+"                ui.Label('NORMAL=0; 20 photos per distinct patch; 100 total. Review images before final split.', height=26)\n"+text[right:]
    result=text+'\nNORMAL_SOURCE = '+repr(str(source_path))+'\nNORMAL_CIRCLE = '+repr(list(circle))+'\n'+PATCH
    ast.parse(result)
    return result

def main():
    panel=getattr(builtins,'_gastric_orbit_calibration_v6',None)
    if panel is None or panel.sample is None: raise RuntimeError('Open a confirmed normal scout frame first')
    if panel.task is not None and not panel.task.done():
        print('[NORMAL] Another operation is running; wait for it to finish.'); return
    source=panel.sample_dir/'sample.json'
    base=(HOME_DIR/'gastric_small_views_v9.py').read_text(encoding='utf-8-sig')
    output=HOME_DIR/'gastric_normal_views_v10.py'
    output.write_text(build_normal_source(base,source,[panel.x.as_int,panel.y.as_int,min(panel.radius.as_int,90)]),encoding='utf-8')
    runpy.run_path(str(output))

if __name__ in ('__main__','<run_path>'): main()

