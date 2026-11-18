# Run this SHORT file inside Isaac Sim. Reuses the complete saved V8 engine.
# Creates a self-contained small-view engine beside it, then starts 50 ulcer photos.
from pathlib import Path
import ast
import builtins
import runpy

HOME_DIR = Path('/home/multirobo/gastric-single-frame-20261117')
PATCH = r'''
def small_view_proposals(region, phase, index, units, cloud, rng):
    source = json.loads(Path(region['source_sample_path']).read_text(encoding='utf-8'))
    target = np.asarray(region['center_world_m'], float)
    eye0 = np.asarray(source['camera_position_world_m'], float)
    ward = unit(eye0-target)
    right = np.asarray(source['camera_world_matrix_usd_row_vector'], float)[0,:3]
    tangent = unit(right-ward*float(right@ward))
    bitangent = unit(np.cross(ward,tangent))
    alpha = math.radians((phase+index*137.507764)%360)
    radial = math.cos(alpha)*tangent+math.sin(alpha)*bitangent
    angles = [[5.5,7.5,9.5],[7.5,9.5,5.5],[9.5,5.5,7.5]][index%3]
    ranges = [[45,50,55],[50,55,45],[55,45,50]][index%3]+[40,58,35,60,30]
    number = 0
    for angle in angles:
        for distance in ranges:
            number += 1
            theta = math.radians(angle)
            eye = target+distance*.001*(math.cos(theta)*ward+math.sin(theta)*radial)
            pixel = (np.array([400.,400.])+rng.uniform(-6,6,2)).tolist()
            matrix = orbit_camera_matrix(eye,target,units,[801,801],90,pixel)
            axis_angle = math.degrees(math.acos(float(np.clip((-matrix[2,:3])@(-ward),-1,1))))
            if not 5 <= axis_angle <= 10:
                continue
            candidate = {'eye_m':eye.tolist(),'matrix':matrix.tolist(),
                'proposal_index':number,'target_pixel_xy':pixel,
                'camera_to_marked_center_mm':distance,'view_offset_deg':angle,
                'camera_view_change_deg':axis_angle,'local_phase_deg':math.degrees(alpha)%360,
                'reference_sample_path':region['source_sample_path'],
                'rotation_mode':'local 5-10 degrees from registered reference, not full orbit'}
            yield orbit_photo_framing(candidate,cloud,units,[801,801],90)

class GastricSmallViews(GastricOrbitPanelV8):
    async def pilot(self,name):
        await self.collect(name,50)

orbit_v7_proposals = small_view_proposals
GastricOrbitPanel = GastricSmallViews
_small_panel = gastric_orbit_setup()
async def _small_start():
    loading = _small_panel.task
    if loading is not None:
        await loading
    if _small_panel.sample is None or _small_panel.stop_requested:
        print('[SMALL] Load was stopped or failed; collection not started.')
        return
    _small_panel.start(lambda: _small_panel.collect('ulcerated',50))
asyncio.ensure_future(_small_start())
'''


def build_small_source(base):
    if 'class GastricOrbitPanelV8' not in base or not base.rstrip().endswith('gastric_orbit_setup()'):
        raise ValueError('V8 file is incomplete. Restore the complete V8 file first.')
    text = base.rstrip().removesuffix('gastric_orbit_setup()')
    changes = [
        ('v8-image-count-practical-framing','v9-local-five-ten-degree-photos'),
        ('Gastric | Orbit Collection v8','Gastric | Small Views: 50 photos'),
        ("'collect_v8_'+name", "'collect_small_'+name"),
        ("'minimum_hash_hamming':8", "'minimum_hash_hamming':4"),
        ('if nearest < 8:', 'if nearest < 4:'),
        ("'minimum_clearance_mm':5,'marked_center_image_margin_fraction':.15,",
         "'minimum_clearance_mm':5,'marked_center_image_margin_fraction':.15, 'view_change_range_deg':[5,10], 'angular_reference':'registered seed sightline',"),
        ("'capture_group_id':region['region_id']+'_sector_%02d'%(azimuth//30)",
         "'capture_group_id':region['region_id']+'_small_reference'"),
        ("'grouping_note':'provisional view-sector group; dedup/group audit before split'",
         "'grouping_note':'all nearby views of this reference stay in one group; audit before split'"),
        ("'split':None,'azimuth_deg':azimuth,",
         "'split':None,'view_offset_deg':candidate['view_offset_deg'],'camera_view_change_deg':candidate['camera_view_change_deg'],"),
        ("attempt={'proposal_index':proposals,'azimuth_deg':azimuth,'sweep':sweep,",
         "attempt={'proposal_index':proposals,'proposal_phase_deg':azimuth,'sweep':sweep,"),
        ('COLLECT %s: %d / 50 saved; azimuth %d; testing pose %d...',
         'SMALL VIEW %s: %d / 50 saved; set %d; pose %d...'),
        ('4 Pilot RAISED: 12 directions','4 Collect RAISED: 50 small views'),
        ('4 Pilot ULCERATED: 12 directions','4 Collect ULCERATED: 50 small views'),
    ]
    for old,new in changes:
        if old not in text:
            raise ValueError('V8 layout differs; missing expected code: '+old[:70])
        text = text.replace(old,new)
    result = '# Local small-view collection: generated from the saved V8 engine.\n'+text+'\n'+PATCH
    ast.parse(result)
    return result


def main():
    old = getattr(builtins,'_gastric_orbit_calibration_v6',None)
    task = getattr(old,'task',None)
    if task is not None and not task.done():
        print('[SMALL] An operation is running. Wait or Stop after current frame; no duplicate started.')
        return
    base = (HOME_DIR/'gastric_orbit_v8.py').read_text(encoding='utf-8-sig')
    output = HOME_DIR/'gastric_small_views_v9.py'
    output.write_text(build_small_source(base),encoding='utf-8')
    print('[SMALL] Starting 50 ulcer photos: local 5-10 degree views, true range 30-60 mm.')
    runpy.run_path(str(output))


if __name__ in ('__main__','<run_path>'):
    main()

