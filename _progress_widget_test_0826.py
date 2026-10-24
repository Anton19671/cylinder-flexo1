import sys,time,json,tkinter as tk
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from FINAL_GUI import GreenStepProgressbar
r=tk.Tk(); r.geometry('500x100+10+10')
p=GreenStepProgressbar(r,length=420,maximum=10,value=0,height=18)
p.pack(padx=20,pady=30,fill='x')
r.update_idletasks(); r.update()
w=p.winfo_width();h=p.winfo_height()
p.start(60)
coords=[]
for _ in range(8):
    r.update_idletasks();r.update();time.sleep(0.07)
    items=p.find_withtag('progress_fill')
    coords.append([list(map(float,p.coords(i))) for i in items])
res={'width':w,'height':h,'animating':p._animating,'frames_with_green':sum(bool(x) for x in coords),'frames':coords}
print(json.dumps(res,indent=2))
assert w>100 and res['animating'] and res['frames_with_green']>=5,res
p.stop();r.destroy()
