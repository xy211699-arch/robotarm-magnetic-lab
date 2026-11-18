"""Run INSIDE Isaac Sim. Moves toward the cross, saves ONE scout, no batch.

Requires an existing gastric capture panel. Uses its validated capture engine.
Farther than 40 mm: aims for 35 mm. Nearer: reduces by 5 mm, floor 31 mm.
The saved scout records its actual measured distance and replay pose/config.
"""
import asyncio
import builtins
from pathlib import Path


def closer_distance_mm(current_mm):
    if not 0 < current_mm < float('inf'):
        raise ValueError('Current geometric distance is invalid')
    return max(31.0, min(35.0, current_mm-5.0))


def main():
    panel = getattr(builtins, '_gastric_orbit_calibration_v6', None)
    if panel is None or panel.sample is None:
        print('[CLOSER] Restore the capture window first.')
        return

    async def move():
        # Resolve helpers from the existing panel so the active V8/V9/V10
        # source, units, geometry checks and output structure stay consistent.
        g = panel.recenter.__func__.__globals__
        g['omni'].timeline.get_timeline_interface().pause()
        await g['omni'].kit.app.get_app().next_update_async()
        await panel.get_query()
        eye, axis = g['orbit_pixel_ray'](panel.sample,
                                       [panel.x.as_int, panel.y.as_int])
        hits = await asyncio.to_thread(panel.query.hits, eye, axis)
        if not len(hits):
            panel.message('CLOSER: cross ray has no stomach hit. Choose another point.')
            return
        current_mm = float(hits[0])*1000
        if current_mm <= 31.001:
            panel.message('CLOSER: already %.2f mm; keeping the 30-60 mm range.' % current_mm)
            return
        wanted_mm = closer_distance_mm(current_mm)
        target = eye + axis*float(hits[0])
        new_eye = target-axis*(wanted_mm*.001)
        matrix = g['orbit_camera_matrix'](
            new_eye, target, panel.sample['meters_per_unit'],
            [801,801], panel.sample['projection_parameters']['horizontal_fov_deg'], [400,400])
        record = await panel.capture(matrix, g['ORBIT_HOME']/'scouts', True)
        if not record.get('valid'):
            panel.message('CLOSER rejected; previous preview retained: '+str(record.get('invalid_reasons')))
            return
        panel.set_sample(record, Path(record['image_path']).parent)
        panel.window.visible = True
        panel.message('CLOSER saved: %.2f -> %.2f mm. ONE scout; no batch started.' % (
                      current_mm, record['distance_mm']))

    panel.start(move)


if __name__ in ('__main__', '<run_path>'):
    main()

