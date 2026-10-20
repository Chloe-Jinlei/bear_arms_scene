"""Decode and compare every recording from the same Genesis execution."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,subprocess
from fractions import Fraction
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve()
r=json.loads((run/'report.json').read_text());meta=r['recordings'];times=meta['simulation_times_s'];w,h=meta['left_video_resolution']
assert times and all(b>a for a,b in zip(times,times[1:]))
def inspect(name):
 p=run/name
 probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-show_entries','format=duration,size','-of','json',str(p)],capture_output=True,text=True,check=True)
 data=json.loads(probe.stdout);v=data['streams'][0]
 decode=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True,text=True)
 dims=[2*w,h+76] if name=='wrapping-process.mp4' else [w,h]
 checks={'full_decode':decode.returncode==0 and not decode.stderr.strip(),'frame_count_matches_recorded_steps':int(v['nb_read_frames'])==len(times),'dimensions_match':[v['width'],v['height']]==dims,'fps_matches':float(Fraction(v['r_frame_rate']))==meta['fps'],'duration_matches':abs(float(data['format']['duration'])-len(times)/meta['fps'])<1e-5}
 return {'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'stream':v,'format':data['format'],'ffmpeg_full_decode_exit_code':decode.returncode,'decode_stderr':decode.stderr,'checks':checks}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:videos=list(pool.map(inspect,meta['files']))
synced=meta['same_simulation_steps_for_all_videos'] and len(videos)==3 and len(set(int(v['stream']['nb_read_frames']) for v in videos))==1
passed=synced and all(all(v['checks'].values()) for v in videos)
result={'status':'passed' if passed else 'failed','videos_synchronized':synced,'shared_simulation_frame_count':len(times),'simulation_time_range_s':[times[0],times[-1]],'fps':meta['fps'],'videos':videos}
(run/'video-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert passed
