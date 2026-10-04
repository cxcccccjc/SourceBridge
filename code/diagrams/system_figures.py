"""SourceBridge system figures: editable native geometry and text, no raster assets.

Scientific inputs: manuscript_en/{overview,method}.tex. Original native city and
method illustrations are supplemented by bundled licensed Lucide SVG role symbols
(assets/lucide includes source SVGs and ISC/MIT notices). The canvas,
main module frames, stage/title anchors and external flow routes are retained.
Public reference objects remain distinct from source identities.
"""
from pathlib import Path
import os,sys,json,tempfile
HERE=Path(__file__).resolve().parent;BASE=HERE
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from sourcebridge_palettes import get_palette,text_color
PALETTE,C=get_palette()
QA_OUT=Path(os.environ.get('SOURCEBRIDGE_CONCEPT_QA_OUT',str(BASE/'quality')));QA_OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'sourcebridge-mplcache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sourcebridge_icons import draw_icon
from sourcebridge_symbols import draw_symbol
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle,FancyBboxPatch,Circle,Ellipse,Polygon,FancyArrowPatch
for n in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    f=Path(os.environ.get('SOURCEBRIDGE_FONT_DIR','C:/Windows/Fonts'))/n
    if f.exists():font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
    'pdf.fonttype':42,'svg.fonttype':'none','text.color':C['ink'],'savefig.facecolor':'white'})
OUT=Path(os.environ.get('SOURCEBRIDGE_CONCEPT_OUT',str(BASE/'figures')));OUT.mkdir(parents=True,exist_ok=True)
META=[]
def co(c):return C.get(c,c)
def canvas(w,h):
    # Preserve logical coordinates and font sizes while tightening vertical space.
    physical_height=3.40 if h>3 else 2.72
    f=plt.figure(figsize=(w,physical_height));a=f.add_axes([0,0,1,1]);a.set(xlim=(0,w),ylim=(0,h));a.axis('off');return f,a
def text(a,x,y,s,size=9,c='ink',ha='left',va='center',bold=False,**kw):
    return a.text(x,y,s,fontsize=size,color=text_color(C,c),ha=ha,va=va,fontweight='bold' if bold else 'normal',linespacing=1.12,**kw)
def icon_text(a,cx,y,s,kind,size=8.6,c='ink',icon_c='blue',icon_w=.12,gap=.06,bold=False,icon_right=False):
    """Center the complete symbol-plus-label group, not just its text."""
    t=text(a,cx,y,s,size,c,ha='left',bold=bold)
    tw=t.get_window_extent(a.figure.canvas.get_renderer()).width/a.figure.dpi
    total=icon_w+gap+tw;left=cx-total/2
    tx=left if icon_right else left+icon_w+gap
    ix=left+tw+gap if icon_right else left
    t.set_position((tx,y))
    if kind=='coins':
        draw_icon(a,'coins',ix,y-icon_w/2,icon_w,icon_w,color=C[icon_c],accent=C[icon_c],face=C['p'+icon_c],lw=.45,z=5)
    else:
        draw_symbol(a,kind,ix,y-icon_w/2,icon_w,color=C[icon_c],lw=.65,badge=C['p'+icon_c])
    if not hasattr(a,'_centered_units'):a._centered_units=[]
    a._centered_units.append({'label':s,'kind':kind,'center':[cx,y],'group_width_inches':total,'alignment':'icon and text centered together; both vertically centered'})
    return t

def rect(a,x,y,w,h,fc='white',ec='line',lw=.65,r=0,z=1,**kw):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=co(fc),edgecolor=co(ec),lw=lw,zorder=z,**kw) if r else Rectangle((x,y),w,h,facecolor=co(fc),edgecolor=co(ec),lw=lw,zorder=z,**kw)
    a.add_patch(p);return p
def circle(a,x,y,r,fc='white',ec='line',lw=.7,z=3):
    p=Circle((x,y),r,facecolor=co(fc),edgecolor=co(ec),lw=lw,zorder=z);a.add_patch(p);return p
def poly(a,xy,fc='white',ec='line',lw=.65,z=2):
    p=Polygon(xy,closed=True,facecolor=co(fc),edgecolor=co(ec),lw=lw,zorder=z);a.add_patch(p);return p
def line(a,xy,c='line',lw=.8,ls='-',z=3):
    x,y=zip(*xy);a.plot(x,y,color=co(c),lw=lw,linestyle=ls,solid_capstyle='round',solid_joinstyle='round',zorder=z)
def arrow(a,xy,c='blue',lw=.9,ls='-',size=7,z=4):
    if len(xy)>2:line(a,xy[:-1],c,lw,ls,z)
    p=FancyArrowPatch(xy[-2],xy[-1],arrowstyle='-|>',mutation_scale=size,lw=lw,color=co(c),linestyle=ls,shrinkA=0,shrinkB=0,zorder=z);a.add_patch(p)
def interval(a,l,u,y,c='blue',lw=1.6,center=False,tick=.035):
    line(a,[(l,y),(u,y)],c,lw);line(a,[(l,y-tick),(l,y+tick)],c,.8);line(a,[(u,y-tick),(u,y+tick)],c,.8)
    if center:circle(a,(l+u)/2,y,.023,c,'white',.5)
def pin(a,x,y,s,c='blue'):
    # Tip is (x,y); pin indexes an object/location, never a source identity.
    poly(a,[(x,y),(x-.105*s,y+.18*s),(x+.105*s,y+.18*s)],c,c,.4)
    circle(a,x,y+.20*s,.13*s,c,c,.5);circle(a,x,y+.20*s,.052*s,'white','white',.5)
def document(a,x,y,w,h,c='blue',fc='white',rows=3,fold=True):
    if fold:
        t=min(w,h)*.18
        poly(a,[(x,y),(x+w,y),(x+w,y+h-t),(x+w-t,y+h),(x,y+h)],fc,c,.75)
        poly(a,[(x+w-t,y+h),(x+w-t,y+h-t),(x+w,y+h-t)],'pgray',c,.55)
    else:rect(a,x,y,w,h,fc,c,.7,.022)
    for i in range(rows):line(a,[(x+w*.18,y+h*(.25+i*.18)),(x+w*.78,y+h*(.25+i*.18))],c,.7)
def coin(a,x,y,s=1):
    for dy in [0,.035,.07]:
        a.add_patch(Ellipse((x,y+dy*s),.20*s,.08*s,facecolor=co('pamber'),edgecolor=co('amber'),lw=.6,zorder=3))
    line(a,[(x-.03*s,y+.07*s),(x+.025*s,y+.07*s)],'amber',.6)
def save(f,name,role):
    for ext in ['pdf','svg','png']:
        tmp=OUT/f'{name}.next.{ext}';f.savefig(tmp,format=ext,dpi=420,bbox_inches=None);tmp.replace(OUT/f'{name}.{ext}')
    f.canvas.draw()
    renderer=f.canvas.get_renderer()
    texts=[t for aa in f.axes for t in aa.texts if t.get_text()]
    bounds=[t.get_window_extent(renderer) for t in texts]
    overlap_pairs=[]
    for i,bi in enumerate(bounds):
        for j,bj in enumerate(bounds[i+1:],i+1):
            if bi.overlaps(bj):
                overlap_pairs.append([texts[i].get_text(),texts[j].get_text()])
    svg=(OUT/f'{name}.svg').read_text(encoding='utf-8')
    META.append({'figure':name,'palette':PALETTE,'size_inches':list(f.get_size_inches()),'role':role,
        'icon_provenance':'Original native illustrations plus bundled licensed Lucide SVG role symbols; assets/lucide includes SVG originals and ISC/MIT notices; native editable paths only',
        'reference_roles':{'ETBP-TD':'Repeated identity records and typed data carriers','QPTD':'Catalog stacks, structured object dossiers and visible intermediate representations','CMAB-DTD':'Scene with independently controlled actors and a structured report matrix'},
        'minimum_explicit_pt':min(t.get_fontsize() for t in texts),
        'maximum_explicit_pt':max(t.get_fontsize() for t in texts),
        'text_items':len(texts),
        'pure_black_text_items':sum(to_rgba(t.get_color())[:3]==(0,0,0) for t in texts),
        'layout_description':'canvas, main frames, stage anchors and external routes retained; richer original-size insets, symbolic headers, weighted WLS residuals, interval/depth scan and explicit S formula; optical icon-text centering',
        'centered_icon_text_units':[u for aa in f.axes for u in getattr(aa,'_centered_units',[])],
        'text_overlap_pairs':overlap_pairs,'text_within_canvas':all(f.bbox.contains(b.x0,b.y0) and f.bbox.contains(b.x1,b.y1) for b in bounds),
        'native_svg_no_image':('<image' not in svg),'live_svg_text':('<text' in svg),
        'content_scope':'Declared drawn features, reviewed against overview.tex and method.tex; not automatic theorem verification',
        'new_graphic_provenance':('affine-response and catalog interval schematics retained. Public specification cells and precision receipt receive licensed role symbols and centered set/cost/radius units, without changing outer geometry or adding observations.' if name=='motivation' else 'directional feasible-set glyphs retained for the actual three-variable LP. WLS uses explanatory synthetic weighted marks and residuals, with public midpoint proxies d_j (not ground truth). MLNI illustrates residual-bias/variance updates without convergence or global-optimum claims. The same four intervals yield depth scan q=.45 and H=[.30,.57]. S is drawn as [u-R,l+R] intersect T with independently quoted R.'),
        'scientific_content':(
            ['one continuous scalar q_t','heterogeneous affine source responses',
             'calibration locations j distinct from source identities S_i',
             'public intervals, gain bounds, target domain and error bounds before purchase',
             'precision quote with A*, C_r*, R <= r',
             'unattainable request implies no purchase'] if name=='motivation' else
            ['public specifications and request before collection','m > 2f complete source identities',
             'two directional LP values per support of at most two objects',
             'shared-cost union of at most four objects','purchase only on feasible quote',
             'unattainable request implies no purchase','same complete matrix feeds both branches',
             'nominal WLS + proper MLNI branch','source intervals and depth >= m-f branch',
             'R enters safe projection independently of reports, H and z',
             'conditional mathematical error bound under declared model']),
        'controlled_identity_encoding':('Third complete row: neutral hatch; third source interval: neutral dashed line' if name=='workflow' else None),
        'report_header_fill_role':('text_red / text_teal' if name=='workflow' else None),
        'schematic_depth':({'intervals':[[.10,.62],[.24,.72],[.69,.89],[.30,.57]],
                            'm':4,'f':1,'scan_q':.45,'scan_depth':3,'retained_depth_region':[.30,.57],'retained_hull':[.30,.57],
                            'status':'geometrically consistent illustration, not experimental data'}
                            if name=='workflow' else None),
        'schematic_projection_axis':({'T':[.48,1.75],'H':[.66,1.39],'R':.615,'S':[.775,1.275],'z':1.65,'projected_point':1.275,'status':'Unified drawing coordinates only; S=[u-R,l+R] intersect T'} if name=='workflow' else None),
        'schematic_wls':({'x_role':'uncertain public midpoint proxy d_j','y_role':'Y_ij','weights':[1,3,2,4],'marker_encoding':'area proportional to illustrative weight','residual_check':'weighted intercept and slope residual moments equal zero','status':'Synthetic explanatory geometry; no empirical result'} if name=='workflow' else None)})
    plt.close(f)


# These native glyphs represent scientific objects, not measured observations.

def tree(a,x,y,s=1):
    line(a,[(x,y),(x,y+.15*s)],'muted',1.0,z=1)
    circle(a,x-.055*s,y+.18*s,.071*s,'ground_border','ground_border',0,z=1)
    circle(a,x+.054*s,y+.19*s,.065*s,'ground_border','ground_border',0,z=1)
    circle(a,x,y+.25*s,.080*s,'pteal','ground_border',.4,z=1)


def target_gauge(a,x,y,w,c='red'):
    interval(a,x,x+w,y,c,1.2,False,.028)
    for q in [.25,.50,.75]:line(a,[(x+w*q,y-.018),(x+w*q,y+.018)],c,.45)
    circle(a,x+w/2,y,.026,c,'white',.4)

def source_thumb(a,x,y,w,h,c,index):
    rect(a,x,y,w,h,'white','line',.60,.023)
    line(a,[(x+.07,y+.06),(x+.07,y+h-.05)],'muted',.45)
    line(a,[(x+.07,y+.06),(x+w-.04,y+.06)],'muted',.45)
    # Different illustrative slopes AND intercepts, without data points/numbers.
    lo,hi=[(.08,.21),(.16,.235),(.12,.195)][index]
    lo*=h/.29;hi*=h/.29;band=.027*h/.29
    # A bounded-error envelope enriches the same affine response schematic.
    poly(a,[(x+.11,y+lo-band),(x+w-.06,y+hi-band),
            (x+w-.06,y+hi+band),(x+.11,y+lo+band)],'mid_'+c,'none',0,z=2)
    # Identical normalized scalar input in all three schematics. Dashed axis
    # projections and a bounded-error cross-section are model geometry, not data.
    rx=x+.11+.58*(w-.17);ry=y+lo+.58*(hi-lo)
    line(a,[(rx,y+.06),(rx,ry+band)],c,.55,'--',z=2.5)
    line(a,[(x+.07,ry),(rx,ry)],c,.45,':',z=2.5)
    line(a,[(rx-.018,ry-band),(rx+.018,ry-band)],c,.65)
    line(a,[(rx,ry-band),(rx,ry+band)],c,.80)
    line(a,[(rx-.018,ry+band),(rx+.018,ry+band)],c,.65)
    line(a,[(rx,y+.047),(rx,y+.07)],c,.60)
    line(a,[(x+.11,y+lo),(x+w-.06,y+hi)],c,1.20)
    line(a,[(x+.11,y+lo+band),(x+w-.06,y+hi+band)],c,.45,':')
    line(a,[(x+.11,y+lo-band),(x+w-.06,y+hi-band)],c,.45,':')


def receipt(a,x,y,w,h,small=False):
    # Original receipt contour, with centered set/cost/radius fields inside.
    points=[(x,y+.045),(x+.065,y),(x+.13,y+.045),(x+.195,y),(x+w-.195,y),(x+w-.13,y+.045),(x+w-.065,y),(x+w,y+.045),(x+w,y+h),(x,y+h)]
    poly(a,points,'white','red',.75)
    rect(a,x+.008,y+h-.225,w-.016,.217,'pred','none',0,z=2.1)
    line(a,[(x+.08,y+h-.27),(x+w-.08,y+h-.27)],'red_panel_border',.65)
    if not small:
        icon_text(a,x+w/2,y+h-.12,'Precision quote','receipt-text',8.9,'ink','red',.135,.06,True)
        cell_w=(w-.22)/2
        for xx,role in [(x+.07,'red'),(x+.15+cell_w,'amber')]:
            rect(a,xx,y+.18,cell_w,.17,'surface_'+role,'red_panel_border' if role=='red' else 'fee_border',.45,.015,z=2.2)
        icon_text(a,x+.07+cell_w/2,y+.265,r'$A^*$','map-pin',8.6,'red','red',.115,.07)
        icon_text(a,x+.15+1.5*cell_w,y+.265,r'$C_r^*$','coins',8.6,'red','amber',.12,.075)
        target_gauge(a,x+.15,y+.10,.68,'red')
        text(a,x+1.18,y+.105,r'$R\leq r$',8.6,'red',ha='center')

def site_token(a,x,y,label,c='blue',w=.23,h=.27,highlight=False):
    # A reference-object dossier. The identity inside is j, never source i.
    document(a,x,y,w,h,c,'pred' if highlight else 'white',0)
    text(a,x+w*.47,y+h*.49,label,8.2,c,ha='center')
    line(a,[(x+.035,y+.035),(x+w-.035,y+.035)],c,.45)

def compact_directory(a,x,y,w,h):
    # One genuinely shared table replaces four repetitive large cards.
    rect(a,x+.045,y-.025,w,h,'pgray','line',.4,.025)
    rect(a,x,y,w,h,'white','blue',.65,.025)
    rect(a,x+.012,y+h-.25,w-.024,.23,'pblue','pblue',0,.015)
    icon_text(a,x+.23,y+h-.125,'Object','database',8.4,'ink','blue',.09,.03)
    text(a,x+.86,y+h-.125,r'$I_j$',8.6,'blue',ha='center')
    text(a,x+w-.20,y+h-.125,'Fee',8.4,'ink',ha='center')
    for k,(lo,hi) in enumerate([(.14,.59),(.02,.96),(.43,.69),(.31,.88)]):
        yy=y+h-.43-k*((h-.60)/3)
        if k%2==0:rect(a,x+.02,yy-.13,w-.04,.26,'surface_blue','none',0)
        pin(a,x+.14,yy-.058,.29,'blue')
        text(a,x+.33,yy,rf'$j_{k+1}$',8.6,'blue',ha='center')
        ruler_l=x+.57;ruler_w=.48
        line(a,[(ruler_l,yy),(ruler_l+ruler_w,yy)],'interval_grid',1.05)
        for u in [0,.5,1]:line(a,[(ruler_l+u*ruler_w,yy-.065),(ruler_l+u*ruler_w,yy+.065)],'interval_grid',.35)
        rect(a,ruler_l+lo*ruler_w,yy-.042,(hi-lo)*ruler_w,.084,'mid_blue','none',0,.012,z=2)
        interval(a,ruler_l+lo*ruler_w,ruler_l+hi*ruler_w,yy,'blue',1.45,False,.029)
        # A fee tag makes the existing cost symbol a distinct carrier.
        tx=x+w-.37
        poly(a,[(tx,yy),(tx+.055,yy-.105),(x+w-.045,yy-.105),
                (x+w-.045,yy+.105),(tx+.055,yy+.105)],'pamber','fee_border',.45)
        circle(a,tx+.049,yy,.010,'white','fee_border',.35)
        text(a,x+w-.20,yy,rf'$c_{k+1}$',8.6,'amber',ha='center')

def motivation():
    """Fig. 1: shared scalar -> heterogeneous sources -> public data -> quote.

    The muted city supplies context; the typed catalog and precision contract
    carry the research claim. Every element is native editable vector geometry.
    """
    f,a=canvas(7.16,2.90)
    # Primary boundaries and header bands organize the retained open scenes.
    for x,w,c,border in [(.055,3.215,'blue','blue_phase_border'),
                         (3.365,1.77,'blue','blue_phase_border'),
                         (5.25,1.855,'red','projection_border')]:
        rect(a,x,.03,w,2.775,'surface_'+c,border,1.0,.025,z=0)
        rect(a,x+.006,2.49,w-.012,.309,'p'+c,'none',0,.018,z=.1)
    for x0,x1,c in [(.12,3.16,'blue'),(3.47,4.99,'blue'),(5.30,7.04,'red')]:
        line(a,[(x0,2.46),(x1,2.46)],c,.75)
    text(a,.12,2.64,'(a) One target, heterogeneous sources',9.4,bold=True)
    text(a,3.47,2.64,'(b) Public catalog',9.4,bold=True)
    text(a,5.30,2.64,'(c) Precision contract',9.4,bold=True)

    # Native city elements are kept pale and sparse. They are a context layer,
    # never an encoding of measured values or target ground-truth observations.
    city_ground=poly(a,[(.16,1.01),(3.14,1.01),(3.03,2.03),(.28,2.03)],'ground','ground_border',.40,z=.2)
    for yy in [1.18,1.64,1.95]:
        line(a,[(.23,yy),(3.06,yy)],'white',7.0,z=.3)
        a.lines[-1].set_clip_path(city_ground)
    for xx in [.85,2.14]:
        line(a,[(xx-.05,1.04),(xx+.04,2.03)],'white',7.5,z=.3)
        a.lines[-1].set_clip_path(city_ground)
    for xx,yy,ww,hh in [(.28,1.95,.24,.29),(.57,1.96,.20,.35),(2.51,1.94,.23,.35),(2.80,1.95,.20,.27)]:
        draw_icon(a,'building',xx,yy,ww,hh,color=C['muted'],accent=C['ground_border'],face=C['pgray'],lw=.36,z=1)
    tree(a,1.00,1.71,.48)
    tree(a,2.87,1.37,.48)
    for xx,yy,j in [(.47,1.90,1),(.88,1.94,2),(2.63,1.88,3)]:
        pin(a,xx,yy,.27,'blue')
        text(a,xx+.105,yy+.035,rf'$j_{j}$',8.4,'blue')

    text(a,1.69,2.28,r'Shared PM$_{2.5}$ target $q_t$',8.6,'ink',ha='center')
    draw_icon(a,'air_target',1.475,1.83,.43,.43,color=C['teal'],accent=C['teal'],face=C['white'],lw=.65,z=4)
    # The response branches originate at exactly the same scalar target.
    for endpoint,c in [((.51,1.63),'blue'),((1.57,1.62),'purple'),((2.55,1.49),'teal')]:
        arrow(a,[(1.69,1.85),endpoint],c,.80,'--',5.2,z=2)
    draw_icon(a,'station',.29,1.05,.43,.59,color=C['ink'],accent=C['blue'],face=C['pblue'],lw=.55,z=5)
    draw_icon(a,'participant',1.24,1.05,.57,.58,color=C['ink'],accent=C['purple'],face=C['ppurple'],lw=.50,z=5)
    draw_icon(a,'vehicle',2.13,1.075,.81,.49,color=C['ink'],accent=C['teal'],face=C['pteal'],lw=.55,z=5)
    for k,(xx,c,lab) in enumerate([(.51,'blue',r'$S_1$'),(1.535,'purple',r'$S_2$'),(2.56,'teal',r'$S_m$')]):
        text(a,xx,.91,lab,8.6,c,ha='center')
        line(a,[(xx,.835),(xx,.825)],c,.6)
        source_thumb(a,xx-.33,.49,.66,.33,c,k)
    text(a,1.65,.365,'Different gains / offsets; bounded errors',8.6,'ink',ha='center')
    text(a,1.65,.17,r'$y_{it}=a_iq_t+b_i+e_{it}$',8.6,ha='center')

    # Public calibration objects are spatial locations, distinct from S_i.
    compact_directory(a,3.47,.85,1.49,1.46)
    text(a,4.215,.66,'Available before purchase',8.6,'ink',ha='center')
    line(a,[(3.49,.535),(4.96,.535)],'line',.45)
    rect(a,3.47,.285,1.49,.205,'white','blue_phase_border',.5,.018)
    rect(a,3.47,.075,1.49,.195,'white','blue_phase_border',.5,.018)
    icon_text(a,4.215,.3875,r'$T,\ [a_{\min},a_{\max}]$','sliders-horizontal',8.6,'ink','blue',.115,.06)
    icon_text(a,4.215,.1725,r'Error bounds $\epsilon_j$','ruler',8.6,'ink','blue',.115,.06)
    arrow(a,[(3.18,1.62),(3.40,1.62)],'blue',0.9,size=6.0)
    arrow(a,[(5.035,1.62),(5.23,1.62)],'red',0.9,size=6.0)

    # Request, computation and receipt use different, recognizable carriers.
    draw_icon(a,'requester',5.36,1.98,.68,.46,color=C['ink'],accent=C['purple'],face=C['ppurple'],lw=.52,z=5)
    text(a,6.57,2.285,'Tolerance $r$',8.6,ha='center')
    target_gauge(a,6.28,2.105,.58,'purple')
    arrow(a,[(6.57,1.99),(6.57,1.86)],'red',.85,size=5.4)
    draw_icon(a,'server',5.40,1.34,.36,.43,color=C['ink'],accent=C['red'],face=C['pred'],lw=.53,z=5)
    text(a,5.87,1.735,'SourceBridge',9.4,'ink',bold=True)
    text(a,5.87,1.515,'Select calibration objects',8.3)
    for xx,lab in [(5.66,r'$j_a$'),(6.19,r'$j_b$'),(6.72,r'$j_c$')]:
        site_token(a,xx-.105,1.055,lab,'red',.21,.26)
    line(a,[(5.40,1.012),(6.99,1.012)],'red',.7)
    arrow(a,[(6.19,1.01),(6.19,.885)],'red',.85,size=5.5)
    receipt(a,5.39,.245,1.60,.625)
    text(a,6.19,.105,'Unattainable: no purchase',8.35,'ink',ha='center')
    save(f,'motivation','Shared continuous scalar q_t, heterogeneously calibrated source identities, public calibration-object directory and ex-ante precision contract. Sparse native city context and typed information carriers; no measured observations or privacy claims.')


def calibration_plot(a,x,y,w,h):
    """Explanatory forward weighted regression, never empirical observations."""
    line(a,[(x,y),(x,y+h)],'muted',.45)
    line(a,[(x,y),(x+w,y)],'muted',.45)
    line(a,[(x+.04*w,y+.10*h),(x+.96*w,y+.84*h)],'purple',1.0)
    qs=np.array([.15,.38,.63,.87]);weights=np.array([1.,3.,2.,4.])
    design=np.column_stack([np.ones_like(qs),qs]);raw=np.array([.18,-.16,.22,-.14])
    residuals=raw-design@np.linalg.solve(design.T@(weights[:,None]*design),design.T@(weights*raw))
    assert np.allclose(design.T@(weights*residuals),0)
    for q,e,weight in zip(qs,residuals,weights):
        fitted=.10+(q-.04)*(.74/.92);xx=x+q*w;fy=y+fitted*h
        line(a,[(xx,y+(fitted+e)*h),(xx,fy)],'purple',.7,(0,(1.15,.85)))
        line(a,[(xx-.009,fy),(xx+.009,fy)],'purple',.45)
        circle(a,xx,y+(fitted+e)*h,.009*np.sqrt(weight),'purple','white',.35)

def stage(a,x,y,n,label,c='blue',size=8.9):
    """Small, consistent stage marker; white numerals remain live text."""
    circle(a,x,y,.062,C['text_'+c],C['text_'+c],.35)
    text(a,x,y,str(n),8.0,'white',ha='center',bold=True)
    text(a,x+.12,y,label,size,'ink',bold=True)


def reverse_nominal(a,x,y):
    """Right-to-left WLS -> MLNI conditional cycle -> replaceable point z."""
    # Shallow insets occupy the existing module; outer box and arrows stay.
    rect(a,2.30,.895,.69,.333,'surface_purple','nominal_border',.35,.015)
    rect(a,3.10,.895,.715,.333,'surface_purple','none',0,.015)
    rect(a,4.045,.895,.76,.333,'surface_purple','nominal_border',.35,.015)
    calibration_plot(a,4.24,1.020,.50,.185)
    text(a,4.145,1.11,r'$Y_{ij}$',8.0,'purple',ha='center')
    text(a,4.49,.943,r'$d_j$',8.0,'ink',ha='center')
    arrow(a,[(x+1.67,y+.10),(x+1.47,y+.10)],'purple',.80,size=5)
    nodes=[(x+1.04,y+.14,r'$z_k$',.22),
           (x+.89,y-.025,r'$g_i$',.22),
           (x+1.29,y-.025,r'$\sigma_i^2$',.28)]
    for xx,yy,lab,ww in nodes:
        a.add_patch(Ellipse((xx,yy),ww,.145,facecolor=co('ppurple'),edgecolor=co('purple'),lw=.55))
        text(a,xx,yy,lab,8.0,'purple',ha='center')
    arrow(a,[(x+.99,y+.08),(x+.95,y+.053)],'purple',.60,size=3.9)
    arrow(a,[(x+1.004,y-.025),(x+1.146,y-.025)],'purple',.60,size=3.9)
    arrow(a,[(x+1.232,y+.053),(x+1.126,y+.105)],'purple',.60,size=3.9)
    arrow(a,[(x+.74,y+.10),(x+.40,y+.10)],'purple',.85,size=5)
    interval(a,2.405,2.88,1.055,'muted',.8,False,.02)
    circle(a,x+.295,1.055,.029,'purple','white',.35)
    text(a,x+.295,1.165,r'$z$',8.6,'purple',ha='center')
    text(a,2.65,.945,r'$T$',8.0,'ink',ha='center')
    draw_symbol(a,'target',2.405,1.128,.09,color=C['purple'],lw=.60,badge=C['ppurple'])
    draw_symbol(a,'iteration-cw',3.095,1.128,.072,color=C['purple'],lw=.55)


def reverse_depth(a,x,y):
    """Right-hand source intervals feed the left-hand exact depth silhouette."""
    rect(a,2.345,.19,1.16,.37,'surface_teal','hull_border',.35,.015)
    rect(a,3.84,.175,.91,.385,'surface_teal','hull_border',.35,.015)
    spans=[(.10,.62),(.24,.72),(.69,.89),(.30,.57)]
    sx=x+1.50;sw=.61;h=.25
    for k,(lo,hi) in enumerate(spans):
        yy=y+h-k*.070
        if k==2:
            line(a,[(sx+lo*sw,yy),(sx+hi*sw,yy)],'ink',1.05,'--')
            for xx in [sx+lo*sw,sx+hi*sw]:
                line(a,[(xx,yy-.018),(xx,yy+.018)],'ink',.65)
        else:
            rect(a,sx+lo*sw,yy-.020,(hi-lo)*sw,.040,'pteal','none',0,.008,z=2)
            interval(a,sx+lo*sw,sx+hi*sw,yy,'teal',1.05,False,.021)
    line(a,[(sx+.45*sw,y+.015),(sx+.45*sw,y+h+.01)],'teal',.65,':')
    text(a,sx+.45*sw,.225,r'$q$',8.0,'teal',ha='center')
    text(a,x+2.20,y+.165,r'$C_i$',8.6,'teal',ha='center')
    arrow(a,[(x+1.49,y+.16),(x+1.24,y+.16)],'teal',.85,size=5)
    px=x+.16;pw=.94;py=y+.045;ph=.27
    edges=sorted({0,1,*[v for span in spans for v in span]})
    xs=[0];ys=[0]
    for lo,hi in zip(edges[:-1],edges[1:]):
        depth=sum(v<=(lo+hi)/2<=u for v,u in spans)
        xs.extend([lo,hi]);ys.extend([depth,depth])
    xs.append(1);ys.append(0)
    retained=[(lo,hi) for lo,hi in zip(edges[:-1],edges[1:])
              if sum(v<=(lo+hi)/2<=u for v,u in spans)>=3]
    hull_l=min(lo for lo,hi in retained);hull_u=max(hi for lo,hi in retained)
    assert (hull_l,hull_u)==(.30,.57)
    profile=list(zip(px+np.array(xs)*pw,py+np.array(ys)/4*ph))
    poly(a,[(px,py),*profile,(px+pw,py)],'pteal','none',0,z=1)
    # Filled area comes from the exact same illustrative interval depths.
    rect(a,px+hull_l*pw,py,(hull_u-hull_l)*pw,3/4*ph,'mid_teal','none',0,z=2)
    for d in [1,2]:line(a,[(px,py+d/4*ph),(px+pw,py+d/4*ph)],'interval_grid',.35,z=.8)
    line(a,profile,'teal',1.10)
    line(a,[(px,py+3/4*ph),(px+pw,py+3/4*ph)],'muted',.5,'--')
    line(a,[(px,py),(px+pw,py)],'muted',.45)
    line(a,[(px+.45*pw,py),(px+.45*pw,py+3/4*ph)],'teal',.65,':')
    circle(a,px+.45*pw,py+3/4*ph,.015,'white','teal',.6)
    interval(a,px+hull_l*pw,px+hull_u*pw,py-.065,'teal',2.2,False,.022)
    text(a,px+.85*pw,py-.065,r'$H$',8.6,'teal',ha='center')


def workflow():
    """Fig. 2: numbered, serpentine two-phase information flow.

    A local purchase gate feeds packets directly below it. The quoted radius
    leaves a separate port and has its own uncluttered lane to the projection.
    Acquisition control therefore never implies that R is estimated from Y.
    """
    f,a=canvas(7.16,3.65)
    # Two matched phase frames retain the numbered serpentine flow.
    rect(a,.06,2.16,7.04,1.43,'surface_blue','blue_phase_border',1.05,.035,z=0)
    rect(a,.06,.055,7.04,1.66,'surface_teal','teal_phase_border',1.05,.035,z=0)
    rect(a,.065,3.33,7.03,.25,'pblue','pblue',0,.027,z=.2)
    rect(a,.065,1.45,7.03,.255,'pteal','pteal',0,.027,z=.2)
    text(a,.17,3.46,'Before collection',9.4,'ink',bold=True)
    text(a,2.61,3.46,r'Public $T,\ [a_{\min},a_{\max}],\ \epsilon_j$; $m>2f$; request $r$',8.6,'ink')
    stage(a,.24,3.155,1,'Public directory',size=8.9)
    stage(a,1.56,3.155,2,'Supports',size=8.9)
    stage(a,2.63,3.155,3,'Two directional LPs',size=8.9)
    stage(a,4.35,3.155,4,'Cost union','red',8.9)
    stage(a,5.85,3.155,5,'Precision quote','red',8.9)

    # (1) Object records remain visibly different from source rows below.
    rect(a,.19,2.425,1.10,.545,'white','blue',.65,.025)
    for k,(lo,hi) in enumerate([(.32,.62),(.18,.75),(.47,.70)]):
        yy=2.865-k*.16
        pin(a,.31,yy-.036,.245,'blue')
        line(a,[(.46,yy),(.99,yy)],'interval_grid',.9)
        interval(a,.46+lo*.53,.46+hi*.53,yy,'blue',1.3,False,.023)
        text(a,1.13,yy,rf'$c_{k+1}$',8.2,'amber',ha='center')
    text(a,.74,2.295,r'Public intervals $I_j$',8.5,'ink',ha='center')
    arrow(a,[(1.31,2.705),(1.45,2.705)],'blue',0.9,size=5.5)

    # (2) Three admissible set sizes, not an empirical example selection.
    text(a,1.65,2.895,r'$\varnothing$',8.6,'muted',ha='center')
    site_token(a,1.87,2.785,r'$j_a$','blue',.21,.25)
    for xx,lab in [(1.55,r'$j_a$'),(1.90,r'$j_b$')]:
        site_token(a,xx,2.405,lab,'blue',.21,.25)
    text(a,1.83,2.29,r'$|P|\leq2$',8.6,'blue',ha='center')
    arrow(a,[(2.16,2.705),(2.32,2.705)],'blue',0.9,size=5.5)

    # (3) A paired, reusable LP record makes the two directional values explicit.
    lx=2.37;ly=2.35;lw=1.55;lh=.69
    rect(a,lx+.035,ly+.025,lw,lh,'pgray','line',.40,.02)
    rect(a,lx,ly,lw,lh,'white','blue',.65,.02)
    for k,sgn in enumerate(['+','-']):
        cc='blue' if k==0 else 'teal';xx=lx+k*lw/2;ww=lw/2
        rect(a,xx+.01,ly+lh-.205,ww-.02,.19,'p'+cc,'p'+cc,0,.012)
        text(a,xx+ww/2,ly+lh-.107,rf'$W_{sgn}(P)$',8.6,cc,ha='center')
        l,u=xx+.10,xx+ww-.10;yy=ly+.18
        # Abstract projected feasible-set glyph, not the actual three-variable
        # LP feasible region or a numerical experiment. Both directions share
        # the same polygon and retain the L/U domain and directional value.
        py=ly+.28;ph=.185;pw=u-l
        shape=[(.16,.22),(.48,.08),(.84,.42),(.66,.94),(.27,.83)]
        poly(a,[(l+qx*pw,py+qy*ph) for qx,qy in shape],
             'mid_'+cc,cc,.65,z=2)
        vx,vy=shape[2] if k==0 else shape[0]
        v=l+vx*pw;support_y=py+vy*ph
        line(a,[(v,py-.016),(v,py+ph+.015)],cc,.85)
        line(a,[(v,yy),(v,support_y)],cc,.50,':')
        arrow(a,[(l+(.38 if k==0 else .63)*pw,py+.11),
                 (l+(.77 if k==0 else .23)*pw,py+.11)],cc,.70,size=4.0)
        line(a,[(l,yy),(u,yy)],'interval_grid',.95)
        interval(a,l if k==0 else v,v if k==0 else u,yy,cc,1.40,False,.021)
        circle(a,v,yy,.024,cc,'white',.35)
        text(a,l,ly+.075,r'$L$',8.0,cc,ha='center')
        text(a,u,ly+.075,r'$U$',8.0,cc,ha='center')
    line(a,[(lx+lw/2,ly+.015),(lx+lw/2,ly+lh-.02)],'line',.5)
    text(a,3.145,2.265,'Reuse certificate catalog',8.5,'ink',ha='center')
    arrow(a,[(3.975,2.705),(4.135,2.705)],'red',0.9,size=5.5)

    # (4) Duplicate object j_b visibly merges and is paid exactly once.
    # Matching pale support enclosures emphasize the two selected object sets.
    rect(a,4.145,2.625,.535,.315,'surface_blue','blue_phase_border',.45,.023)
    rect(a,4.875,2.625,.535,.315,'surface_teal','teal_phase_border',.45,.023)
    for x0,x1,lab,cc in [(4.18,4.64,r'$P$','blue'),(4.91,5.37,r'$Q$','teal')]:
        text(a,(x0+x1)/2,3.01,lab,8.3,cc,ha='center')
        line(a,[(x0,2.895),(x0,2.92),(x1,2.92),(x1,2.895)],cc,.60)
    for xx,lab,cc in [(4.18,r'$j_a$','blue'),(4.44,r'$j_b$','red'),(4.91,r'$j_b$','red'),(5.17,r'$j_c$','teal')]:
        site_token(a,xx,2.65,lab,cc,.20,.22,highlight=(lab==r'$j_b$'))
    for sx,dx,cc in [(4.28,4.34,'blue'),(4.54,4.79,'red'),(5.01,4.79,'red'),(5.27,5.19,'teal')]:
        if cc=='red':line(a,[(sx,2.64),(dx,2.60)],cc,.70)
        else:arrow(a,[(sx,2.64),(dx,2.585)],cc,.70,size=4.4)
    # Two incoming copies merge at one joint before the purchased union.
    circle(a,4.79,2.60,.019,'red','white',.4)
    line(a,[(4.79,2.60),(4.79,2.56)],'red',.75)
    for xx,lab in [(4.23,r'$j_a$'),(4.68,r'$j_b$'),(5.08,r'$j_c$')]:
        site_token(a,xx,2.34,lab,'red',.22,.22,highlight=(lab==r'$j_b$'))
    text(a,4.82,2.225,r'$A^*=P\cup Q$, $|A^*|\leq4$',8.5,'ink',ha='center')
    arrow(a,[(5.42,2.705),(5.62,2.705)],'red',0.9,size=5.5)

    # (5) Radius and purchase control have separate, labelled output ports.
    rect(a,5.68,2.19,1.25,.805,'white','red',.7,.022)
    rect(a,5.735,2.715,1.145,.245,'surface_amber','fee_border',.45,.016)
    rect(a,5.735,2.535,1.145,.155,'surface_red','red_panel_border',.45,.016)
    icon_text(a,6.3075,2.8375,r'Fee $C_r^*$','coins',8.3,'ink','amber',.15,.10)
    icon_text(a,6.3075,2.6125,r'$R=W(A^*)/2$','ruler',8.3,'ink','red',.09,.035,icon_right=True)
    line(a,[(5.77,2.51),(6.84,2.51)],'red_panel_border',.5)
    poly(a,[(6.025,2.375),(6.305,2.26),(6.585,2.375),(6.305,2.49)],'pred','red',.65)
    text(a,6.305,2.375,r'$R\leq r?$',8.3,'red',ha='center')
    # Short local purchase gate feeds the report matrix immediately below it.
    arrow(a,[(6.305,2.255),(6.305,1.715)],'red',0.9,size=6)
    text(a,6.14,2.070,'yes',8.2,'ink',ha='right')
    text(a,6.215,1.925,'Purchase',8.2,'ink',ha='right')
    arrow(a,[(6.59,2.375),(6.985,2.375),(6.985,2.12)],'muted',.65,'--',4.7)
    text(a,6.785,2.455,'no',8.2,'ink',ha='center')
    text(a,6.97,2.015,'Unattainable',8.2,'ink',ha='right')
    text(a,6.97,1.865,'no purchase',8.2,'ink',ha='right')
    # This independent R route bypasses report acquisition and both inference
    # branches. Its single straight lane has no crossing or shared junction.
    circle(a,5.79,2.62,.018,'red','white',.3)
    line(a,[(5.79,2.62),(5.57,2.62),(5.57,2.015)],'red',.9)
    arrow(a,[(5.57,2.015),(5.57,1.82),(1.80,1.82),(1.80,1.425)],'red',.9,size=5.8)
    text(a,3.25,1.94,'Quoted radius $R$ for projection',8.6,'ink',ha='center')

    # After-collection flow runs right-to-left, with numbered stages 6 -> 7 -> 8.
    text(a,.17,1.585,'After collection',9.4,'ink',bold=True)
    stage(a,5.41,1.535,6,'Complete packets','teal',8.9)
    stage(a,2.39,1.535,7,'Parallel inference','teal',8.9)
    # Complete source rows; one controlled row is hatched across ALL columns.
    rect(a,5.24,.115,1.76,1.315,'white','hull_border',.75,.022,z=.8)
    draw_symbol(a,'file-spreadsheet',5.335,1.215,.145,color=C['teal'],lw=.65,badge=C['pteal'])
    text(a,5.70,1.2875,r'$Y$',8.3,'teal',ha='center')
    mx=5.925;my=.515;cw=.26;ch=.165
    for j,lab in enumerate([r'$j_a$',r'$j_b$',r'$j_c$',r'$t$']):
        cc='red' if j<3 else 'teal'
        rect(a,mx+j*cw,my+4*ch,cw,.19,C['text_'+cc],'white',.45)
        text(a,mx+(j+.5)*cw,my+4*ch+.095,lab,8.5,'white',ha='center')
    for i,(kind,lab) in enumerate([('station',r'$S_1$'),('phone',r'$S_2$'),('sensor',r'$S_3$'),('vehicle',r'$S_m$')]):
        yy=my+(3-i)*ch;cc='ink' if i==2 else 'blue'
        iw=.16 if kind!='vehicle' else .27
        draw_icon(a,kind,5.31,yy+.012,iw,.14,color=C['ink'],accent=C[cc],face=C['pgray'] if i==2 else C['pblue'],lw=.40,z=5)
        text(a,5.70,yy+ch/2,lab,8.2,cc,ha='center')
        line(a,[(5.81,yy+ch/2),(mx-.015,yy+ch/2)],cc,.45)
        for j in range(4):
            fill='pgray' if i==2 else ('pred' if j<3 else 'pteal')
            p=rect(a,mx+j*cw,yy,cw,ch,fill,'white',.45)
            if i==2:
                p.set_hatch('////');p.set_edgecolor(co('muted'));p.set_linewidth(.20)
            text(a,mx+(j+.5)*cw,yy+ch/2,r'$*$' if i==2 else r'$y$',8.5,cc,ha='center')
    rect(a,mx,my+ch,4*cw,ch,'none','ink',.65,z=4)
    text(a,6.155,.365,r'$m\times(|A^*|+1)$',8.6,ha='center')
    text(a,6.155,.18,r'At most $f$ entire rows controlled',8.25,'ink',ha='center')
    # Both branches consume the identical complete report matrix.
    line(a,[(5.235,.86),(5.075,.86)],'teal',.95)
    circle(a,5.075,.86,.021,'teal','white',.3)
    arrow(a,[(5.075,.86),(5.075,1.125),(4.905,1.125)],'purple',.90,size=5.7)
    arrow(a,[(5.075,.86),(5.075,.435),(4.905,.435)],'teal',.90,size=5.7)
    rect(a,2.21,.875,2.66,.565,'white','nominal_border',.75,.022)
    rect(a,2.215,1.235,2.65,.20,'ppurple','none',0,.018)
    text(a,2.35,1.305,'Nominal: calibrated estimator',8.8,'ink',bold=True)
    draw_symbol(a,'chart-scatter',4.675,1.255,.14,color=C['purple'],lw=.65,badge=C['surface_purple'])
    reverse_nominal(a,2.36,1.005)
    rect(a,2.21,.155,2.66,.605,'white','hull_border',.75,.022)
    rect(a,2.215,.565,2.65,.19,'pteal','none',0,.018)
    text(a,2.35,.645,r'Depth $\geq m-f$',8.8,'ink',bold=True)
    text(a,4.275,.645,'Source intervals',8.8,'ink',ha='center',bold=True)
    draw_symbol(a,'layers',2.235,.599,.095,color=C['teal'],lw=.6)
    draw_symbol(a,'files',3.66,.59,.11,color=C['teal'],lw=.6)
    reverse_depth(a,2.34,.27)
    arrow(a,[(2.185,1.125),(1.965,1.125)],'purple',0.9,size=5.8)
    arrow(a,[(2.185,.435),(1.965,.435)],'teal',0.9,size=5.8)

    # (8) Conditional deterministic certificate: z, H and quoted R are inputs.
    rect(a,.16,.115,1.775,1.31,'white','projection_border',.80,.025)
    rect(a,.165,1.20,1.765,.22,'pred','none',0,.018)
    stage(a,.31,1.295,8,'Safe projection','red',8.9)
    icon_text(a,1.0475,1.09,r'$S=[u-R,\ell+R]\cap T$','ruler',8.0,'ink','red',.10,.045)
    text(a,1.07,.870,r'$H=[\ell,u]$',8.6,'teal',ha='center')
    line(a,[(.48,.97),(1.75,.97)],'line',.70)
    text(a,.34,.97,r'$T$',8.4,'muted',ha='center')
    interval(a,.66,1.39,.97,'teal',2.4,False,.03)
    interval(a,.775,1.275,.745,'red',2.6,False,.03)
    text(a,.43,.745,r'$S$',8.6,'red',ha='center')
    circle(a,1.65,.745,.030,'purple','white',.35)
    text(a,1.65,.84,r'$z$',8.6,'purple',ha='center')
    arrow(a,[(1.61,.745),(1.315,.745)],'red',.85,size=5.3)
    circle(a,1.275,.745,.034,'red','white',.4)
    rect(a,.245,.17,1.61,.425,'surface_red','red_panel_border',.5,.018)
    line(a,[(.31,.38),(1.79,.38)],'red_panel_border',.4)
    icon_text(a,1.05,.485,r'$\widehat q_t=\Pi_{S(H,R)}(z)$','receipt-text',8.6,'ink','red',.105,.055)
    text(a,1.047,.275,r'$|\widehat q_t-q_t|\leq R$',8.6,'ink',ha='center')
    save(f,'workflow','Numbered serpentine workflow. Public inputs -> supports -> two directional LPs -> duplicate-aware union -> feasibility gate; a short local gate purchases complete packets. The same matrix feeds nominal and interval-depth branches right-to-left. Quoted R bypasses data in its own lane and joins H and z at safe projection. No raster artwork, invented observations, or privacy claims.')

if __name__=='__main__':
    motivation();workflow()
    (QA_OUT/'system_figure_manifest.json').write_text(json.dumps(META,indent=2),encoding='utf-8')
    print(json.dumps(META,indent=2))
