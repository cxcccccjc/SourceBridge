"""Regenerate SourceBridge's experimental evidence from unchanged archived data.

Run with Python + numpy + matplotlib; exports are copied to both manuscript
figure directories. No measurements are generated and no bootstrap is rerun.
All panels are 1 x 4 at 7.16 inches; uncertainty retains its original unit.
"""
from pathlib import Path
from collections import Counter
import json, hashlib, shutil, os, tempfile
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'sourcebridge_matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from matplotlib.ticker import NullLocator, NullFormatter, MaxNLocator, AutoMinorLocator, LogLocator

for n in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    p=Path(os.environ.get('SOURCEBRIDGE_FONT_DIR','C:/Windows/Fonts'))/n
    if p.is_file(): font_manager.fontManager.addfont(str(p))
font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,
    'mathtext.fontset':'stix','axes.labelsize':8,'axes.titlesize':8.4,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
    'axes.edgecolor':'black','axes.linewidth':.7,'axes.spines.top':True,
    'axes.spines.right':True,'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.65,
    'ytick.major.width':.65,'svg.fonttype':'none','pdf.fonttype':42,
    'savefig.facecolor':'white','hatch.linewidth':.45})
OUT=ROOT/'exports'; OUT.mkdir(exist_ok=True)
def read(n): return json.loads((ROOT/'data'/n).read_text('utf-8'))
D=read('metrics.json');V=read('plot_metrics.json');J=read('procurement_runs.json')
G=read('greedy_control_results.json');CL=read('greedy_cleanup_results.json')
E=read('end_to_end_verified.json');RAW=read('end_to_end_raw.json')
T=read('inference_timing_distribution.json')
assert E['status']==T['status']=='PASS'
# Stable method colors AND line/marker redundancies, shared by every panel.
STYLE={
 'SourceBridge':('#C43C39','-','o'), 'JB-RobustD':('#3973A6','--','s'),
 'Greedy':('#238064','-.','^'), 'Greedy + cleanup':('#8859A3',':','D'),
 'CRH + WLS':('#3973A6','--','s'),'EPTD + WLS':('#CC8C22','-.','^'),
 'FETD-D + WLS':('#238064',':','D'),'FETD-AK + WLS':('#8859A3','--','v'),
 'WLS median':('#555555','-.','P'),
 'Midpoint':('#927157',':','+'),'Enumeration':('#303030','--','s'),
 'LP catalog':('#8D8D8D',':','^'), 'Exact':('#315A67','-.','P'),
 'Heterogeneous':('#996413',':','X')}
METHODS=['SourceBridge','JB-RobustD','Greedy','Greedy + cleanup']
COMPS=METHODS[1:];FEAS=[r for r in V['procurement_requests'] if r['feasible']]
REGIMES=[(p,s) for p in ['exact','heterogeneous'] for s in [20,150,450]]
REG_LABELS=['E20','E150','E450','H20','H150','H450']
KEYS={'CRH + WLS':'CRH_TKDE2016__weighted_source_affine',
 'EPTD + WLS':'EPTD_TIFS2022__weighted_source_affine',
 'FETD-D + WLS':'FETD_D_AAAI2023__weighted_source_affine',
 'FETD-AK + WLS':'FETD_AK_AAAI2023__weighted_source_affine',
 'SourceBridge':'SourceBridge_proper_MLNI_WLS',
 'WLS median':'weighted_source_affine_median','Midpoint':'SourceBridge_packet'}
META={};QA={}
GRID_STYLE={'major_color':'#D2D2D2','major_width_pt':.28,
            'minor_color':'#E1E1E1','minor_width_pt':.16,
            'direction':'horizontal','linear_subdivisions':4,
            'log_subs':[1,2,5],'minor_tick_labels':False}
def style(name):
    c,ls,m=STYLE[name]
    return dict(color=c,ls=ls,marker=m,ms=3.5,lw=1.05,mew=.8,
                mfc=c if name=='SourceBridge' else 'white')
def handles(names):
    return [Line2D([0],[0],**style(n),label=n) for n in names]
def panels(titles,height=2.18,left=.078,width=.161,step=.239,bottom=.35,top=.82):
    fig=plt.figure(figsize=(7.16,height))
    axes=[fig.add_axes([left+i*step,bottom,width,top-bottom]) for i in range(4)]
    for i,(ax,title) in enumerate(zip(axes,titles)):
        ax.set_title(f'({chr(97+i)}) {title}',pad=7,fontweight='normal')
        ax.grid(axis='y',color=GRID_STYLE['major_color'],lw=GRID_STYLE['major_width_pt'],zorder=0);ax.set_axisbelow(True)
        for spine in ax.spines.values():spine.set(color='black',linewidth=.7)
        ax.tick_params(top=False,right=False,pad=2)
    return fig,axes
def legend(fig,names,ncol=None,y=.005):
    fig.legend(handles=handles(names),loc='lower center',bbox_to_anchor=(.5,y),
        ncol=ncol or len(names),frameon=False,handlelength=2,columnspacing=1.1,
        handletextpad=.4,labelspacing=.35)
def group_legend(fig,hh,x,y=.008,ncol=1):
    """Place separate, panel-aligned keys without adding explanatory labels."""
    return fig.legend(handles=hh,loc='lower center',bbox_to_anchor=(x,y),
        ncol=ncol,frameon=False,handlelength=1.75,columnspacing=.8,
        handletextpad=.35,labelspacing=.28,borderaxespad=0)
def ecdf(ax,values,name):
    x=np.sort(np.asarray(values,float));y=100*np.arange(1,len(x)+1)/len(x)
    c,ls,m=STYLE[name];ax.step(np.r_[0,x],np.r_[0,y],where='post',color=c,ls=ls,lw=1.1)
    ii=np.unique(np.linspace(0,len(x)-1,7).astype(int))
    ax.plot(x[ii],y[ii],ls='none',marker=m,color=c,ms=3.3,mfc=c if name=='SourceBridge' else 'white',mew=.75)
def box(ax,vals,positions,name,width=.4,scatter=False,horizontal=False):
    c=STYLE[name][0]
    result=ax.boxplot(vals,positions=positions,widths=width,orientation='horizontal' if horizontal else 'vertical',whis=(0,100),
       showfliers=False,patch_artist=True,manage_ticks=False,
       boxprops=dict(facecolor=to_rgba(c,.13),edgecolor=c,lw=.8),
       medianprops=dict(color='black',lw=.9),whiskerprops=dict(color=c,lw=.75),capprops=dict(color=c,lw=.75))
    if scatter:
        for p,v in zip(positions,vals):
            jit=.085*np.sin(np.arange(len(v))*2.39996323)
            ax.scatter(v,np.full(len(v),p)+jit,s=4,color=c,alpha=.32,linewidths=0) if horizontal else ax.scatter(np.full(len(v),p)+jit,v,s=4,color=c,alpha=.32,linewidths=0)
    return result
def logn(ax,ns=[8,32,64]):
    ax.set_xscale('log',base=2);ax.set_xticks(ns,ns);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Candidate anchors $n$');ax.yaxis.set_minor_locator(NullLocator())
def refine_horizontal_grid(fig):
    """Add fine, unlabeled guides without changing data, ranges or major ticks."""
    for ax in fig.axes:
        # Horizontal box plots use categorical rows; intermediate fractional
        # row guides would imply an ordering finer than the reported strata.
        categorical=any(t.get_text() in REG_LABELS for t in ax.get_yticklabels())
        if categorical:
            ax.yaxis.set_minor_locator(NullLocator())
        elif ax.get_yscale()=='log':
            ax.yaxis.set_minor_locator(LogLocator(base=10,subs=GRID_STYLE['log_subs'],numticks=40))
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(GRID_STYLE['linear_subdivisions']))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='y',which='minor',left=False,right=False,
                       labelleft=False,labelright=False,length=0)
        ax.grid(axis='y',which='major',color=GRID_STYLE['major_color'],
                lw=GRID_STYLE['major_width_pt'],zorder=0)
        ax.grid(axis='y',which='minor',color=GRID_STYLE['minor_color'],
                lw=GRID_STYLE['minor_width_pt'],zorder=0)
        ax.set_axisbelow(True)

def save(fig,name,meta):
    refine_horizontal_grid(fig)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    outside=[];ignored_ticks=[];hidden_tick_ids=set()
    # Matplotlib creates ticks beyond the axis limits but does not draw them.
    # Check rendered labels only; retain a record of these excluded locator ticks.
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            low,high=sorted(limits)
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low-1e-10 <= tick.get_loc() <= high+1e-10:
                    for label in [tick.label1,tick.label2]:
                        hidden_tick_ids.add(id(label))
                        if label.get_visible() and label.get_text():ignored_ticks.append(label.get_text())
    for text in fig.findobj(matplotlib.text.Text):
        if id(text) in hidden_tick_ids or not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > w+.5 or b.y1 > h+.5:outside.append(text.get_text())
    QA[name]={'inches':list(fig.get_size_inches()),'outside_canvas_text':outside,
              'axis_spines':'four black sides','svg_live_text':True,'undrawn_out_of_limits_locator_ticks':ignored_ticks,
              'legend_entries':[[t.get_text() for t in leg.get_texts()] for leg in fig.legends],
              'axes_legend_entries':{str(i):[t.get_text() for t in ax.get_legend().get_texts()] for i,ax in enumerate(fig.axes) if ax.get_legend() is not None},
              'minimum_visible_font_pt':min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text() and id(t) not in hidden_tick_ids)}
    QA[name]['horizontal_grid']=[{'panel':chr(97+i),'scale':ax.get_yscale(),
       'major_width_pt':GRID_STYLE['major_width_pt'],'minor_width_pt':GRID_STYLE['minor_width_pt'],
       'visible_major_positions':[float(t.get_loc()) for t in ax.yaxis.get_major_ticks() if min(ax.get_ylim())<=t.get_loc()<=max(ax.get_ylim()) and t.gridline.get_visible()],
       'visible_minor_positions':[float(t.get_loc()) for t in ax.yaxis.get_minor_ticks() if min(ax.get_ylim())<=t.get_loc()<=max(ax.get_ylim()) and t.gridline.get_visible()],
       'minor_tick_labels':False} for i,ax in enumerate(fig.axes)]
    meta['horizontal_grid']=GRID_STYLE
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    for language in ['en','zh']:
        fd=ROOT.parents[1]/f'manuscript_{language}'/'figures';fd.mkdir(parents=True,exist_ok=True)
        for ext in ['pdf','svg','png']:shutil.copy2(OUT/f'{name}.{ext}',fd/f'{name}.{ext}')
    META[name]=meta;plt.close(fig)

def procurement():
    fig,(a,b,c,d)=panels(['Requested precision','Feasible requests','Minimum-cost match','Budget-radius gap'])
    for j,m in enumerate(COMPS):
        rr=[r for r in V['procurement_stats'] if r['method']==m and r['r']!='all']
        y=np.array([100*r['mean_fraction_saving'] for r in rr]);ci=100*np.array([r['whole_seed_bootstrap_95'] for r in rr])
        a.errorbar(np.array([r['r'] for r in rr])+(j-1)*2,y,yerr=[y-ci[:,0],ci[:,1]-y],capsize=1.8,**style(m))
    a.set(xlim=(12,158),ylim=(-.8,22),xticks=[20,50,100,150],yticks=[0,10,20],xlabel='Requested radius $r$',ylabel='Mean fee reduction (%)')
    for p,name in [('exact','Exact'),('heterogeneous','Heterogeneous')]:
        rr=[sum(r['feasible'] for r in V['procurement_requests'] if r['profile']==p and r['r']==x)/60*100 for x in [20,50,100,150]]
        b.plot([20,50,100,150],rr,**style(name))
    b.set(xlim=(12,158),ylim=(-3,105),xticks=[20,50,100,150],yticks=[0,50,100],xlabel='Requested radius $r$',ylabel='Feasible pools (%)')
    for i,m in enumerate(METHODS):
        r=next(r for r in V['procurement_stats'] if r['r']=='all' and r['method']==m)
        c.bar(i,100*r['matches']/r['feasible'],.65,color=to_rgba(STYLE[m][0],.75),edgecolor='black',lw=.45,hatch=['','//','..','xx'][i])
    c.set(xticks=range(4),xticklabels=['SB','JB','Gr.','Cl.'],ylim=(0,105),yticks=[0,50,100],xlabel='Procurement rule',ylabel='Optimal-cost requests (%)')
    for m in COMPS[:2]:ecdf(d,V['budget_radius_gaps'][m],m)
    d.set(xlim=(-1,80),ylim=(0,104),xticks=[0,40,80],yticks=[0,50,100],xlabel='Radius above optimum',ylabel='Budget decisions (%)')
    group_legend(fig,handles(METHODS),.5,ncol=4)
    b.legend(handles=handles(['Exact','Heterogeneous']),loc='lower right',
        frameon=True,facecolor='white',edgecolor='#CCCCCC',framealpha=.97,
        handlelength=1.4,handletextpad=.3,borderpad=.3,labelspacing=.3,
        borderaxespad=.35)
    save(fig,'procurement',{'sources':['plot_metrics.json'],'scope':'480 requests, 120 pools; cost panels use all 329 jointly feasible requests; budget CDF uses 240 decisions','interval':'20,000 paired whole-seed bootstrap draws, 95%','panels':['mean savings by radius','all-request feasibility by profile','exact-cost match rate','budget radius gaps'],'profile_legend':'Exact/Heterogeneous key inside panel b, lower right; outside shared key contains methods only'})

def pairs():
    qi={(r['pool_id'],r['r']):r for r in J['quotes']};gi={(r['pool_id'],r['r']):r for r in G['quote_rows']};ci={(r['pool_id'],r['r']):r for r in CL['rows']}
    out=[]
    for r in FEAS:
        key=(r['pool_id'],r['r']);q=qi[key];g=gi[key];cl=ci[key];fixed=q['coarse_cost']+9
        counts={'SourceBridge':len(q['ours_ids']),'JB-RobustD':len(q['jb_ids']),'Greedy':len(g['greedy']['ids']),'Greedy + cleanup':len(cl['cleaned_ids'])}
        out.append(dict(profile=r['profile'],span=r['span'],counts=counts,fixed=fixed,
          anchor_fees={m:r['costs'][m]-fixed for m in METHODS},greedy_extra=cl['original_gap'],cleanup_extra=cl['total_cost_gap']))
    assert len(out)==329
    return out
P=pairs()
def regimes():
    fig,axs=panels(['Versus JB-RobustD','Versus Greedy','Versus cleanup','Anchor-report fees'],height=2.23,left=.078,width=.16,step=.24,bottom=.38,top=.83)
    for i,(ax,m) in enumerate(zip(axs[:3],COMPS)):
        vals=[[100*r['saving'][m] for r in FEAS if r['profile']==p and r['span']==s] for p,s in REGIMES]
        assert sum(map(len,vals))==329
        box(ax,vals,range(6),m,horizontal=True,scatter=True,width=.5)
        ax.axvline(0,color='#777777',ls=':',lw=.65);ax.axhline(2.5,color='#CCCCCC',lw=.5)
        ax.set(yticks=range(6),yticklabels=REG_LABELS,ylim=(5.6,-.6),xlim=(-1,45),xticks=[0,20,40],xlabel='Fee reduction (%)')
        if i==0:ax.set_ylabel('Profile / reference span')
    ax=axs[3]
    for i,m in enumerate(METHODS):
        vals=[np.mean([r['anchor_fees'][m] for r in P if r['profile']==p and r['span']==s]) for p,s in REGIMES]
        ax.plot(range(6),vals,**style(m))
    ax.set(xticks=range(6),xticklabels=REG_LABELS,ylim=(0,40),yticks=[0,20,40],xlabel='Profile / reference span',ylabel='Mean anchor-report fee')
    ax.tick_params(axis='x',labelrotation=60)
    legend(fig,METHODS,ncol=4)
    save(fig,'procurement_regimes',{'sources':['plot_metrics.json','procurement_runs.json','greedy_control_results.json','greedy_cleanup_results.json'],'scope':'All 329 feasible requests; E/H denote exact/heterogeneous. Six strata pool radii. No zero-saving request omitted.','box':'quartiles, median, full minimum/maximum, every request shown','panels':['JB saving distributions','greedy saving distributions','cleanup saving distributions','mean anchor-report fees excluding common fixed charge']})

def mechanism():
    fig,(a,b,c,d)=panels(['Purchased support','Fixed and anchor fees','Complementarity','Cleanup effect'],height=2.28,bottom=.38,top=.82)
    for i,m in enumerate(METHODS):
        v=[100*sum(r['counts'][m]==n for r in P)/329 for n in range(1,5)]
        a.bar(np.arange(1,5)+(i-1.5)*.19,v,.17,color=to_rgba(STYLE[m][0],.8),edgecolor='black',lw=.35,hatch=['','//','..','xx'][i])
    a.set(xticks=range(1,5),xlim=(.5,4.5),ylim=(0,105),yticks=[0,50,100],xlabel='Purchased anchors',ylabel='Requests (%)')
    fee=[]
    for pi,p in enumerate(['exact','heterogeneous']):
        rr=[r for r in P if r['profile']==p]
        for mi,m in enumerate(METHODS):
            x=pi*5+mi;fixed=np.mean([r['fixed'] for r in rr]);anchor=np.mean([r['anchor_fees'][m] for r in rr])
            b.bar(x,fixed,.72,color='#DDDDDD',edgecolor='black',lw=.35)
            b.bar(x,anchor,.72,bottom=fixed,color=STYLE[m][0],edgecolor='black',lw=.35,hatch=['','//','..','xx'][mi]);fee.append([p,m,fixed,anchor])
    b.set(xticks=[1.5,6.5],xticklabels=['Exact','Heterog.'],ylim=(0,480),yticks=[0,200,400],xlabel='Reference profile',ylabel='Mean total fee')
    pool={r['pool_id']:r for r in J['catalogs']}
    for i,(p,n) in enumerate([('exact','Exact'),('heterogeneous','Heterogeneous')]):
        vv=[[r['maximum_complementarity'] for r in G['complementarity_by_pool'] if pool[r['pool_id']]['profile']==p and pool[r['pool_id']]['span']==s] for s in [20,150,450]]
        assert [len(v) for v in vv]==[20]*3
        box(c,vv,np.arange(3)+(i-.5)*.33,n,width=.27,scatter=True)
    c.set(xticks=range(3),xticklabels=[20,150,450],xlabel='Reference span',ylabel='Max. complementarity')
    transitions=Counter((r['greedy_extra'],r['cleanup_extra']) for r in P)
    xx=np.array([q[0] for q in transitions]);yy=np.array([q[1] for q in transitions]);nn=np.array(list(transitions.values()));lim=max(xx.max(),yy.max())
    d.plot([-4,lim+4],[-4,lim+4],ls=':',color='#777777',lw=.7)
    d.scatter(xx,yy,s=9+5*np.sqrt(nn),color=STYLE['Greedy + cleanup'][0],edgecolor='white',linewidth=.4,alpha=.75)
    d.set(xlim=(-4,lim+4),ylim=(-4,lim+4),xlabel='Greedy extra fee',ylabel='Cleanup extra fee');d.xaxis.set_major_locator(MaxNLocator(3));d.yaxis.set_major_locator(MaxNLocator(3))
    method_keys=[Patch(facecolor=to_rgba(STYLE[n][0],.8),edgecolor='black',lw=.4,hatch=h,label=n) for n,h in zip(METHODS,['','//','..','xx'])]
    group_legend(fig,method_keys,.27,ncol=2)
    hh=[Patch(facecolor=to_rgba(STYLE[n][0],.13),edgecolor=STYLE[n][0],lw=.8,label=n) for n in ['Exact','Heterogeneous']]
    hh+=[Patch(facecolor='#DDDDDD',edgecolor='black',lw=.4,label='Fixed charge')]
    group_legend(fig,hh,.76,ncol=2)
    save(fig,'joint_selection',{'sources':['procurement_runs.json','greedy_control_results.json','greedy_cleanup_results.json','plot_metrics.json'],'scope':'329 feasible requests; all 120 pools for complementarity','fee_means':fee,'complementarity':'maximum positive interval-width complementarity per pool; 240 set/pair combinations per pool; min/max whiskers','cleanup':'coincident requests merged; marker area 9+5*sqrt(count); diagonal is unchanged excess fee','panels':['complete support counts','fixed/anchor fee decomposition','all complementarity distributions','all cleanup transitions']})

def robustness():
    fig,(a,b,c,d)=panels(['Target-shift attacks','Anchor attacks','Precision admission','Error / radius'],height=2.40,bottom=.43,top=.83)
    names=['SourceBridge','CRH + WLS','EPTD + WLS','FETD-D + WLS','FETD-AK + WLS','WLS median','Midpoint']
    shifts=[-80,-20,-5,0,5,20,80]
    aa=['clean' if x==0 else f'target_{x:+d}' for x in shifts]
    ab=[f'anchor_g{g}_q{q}' for g in ['0.7','1.3'] for q in [0,250,500]]+['reverse_anchors']
    all_values={};panel_values={'a':{},'b':{}}
    for n in names:
        vals=[]
        for ax,xs,att in [(a,shifts,aa),(b,range(7),ab)]:
            yy=[next(r['rmse'] for r in D['public']['fixed_attacks'] if r['profile']=='heterogeneous' and r['label']==KEYS[n] and r['attack']==q) for q in att]
            vals+=yy
            # Panel b is the user-requested six-comparator visual subset;
            # retain the omitted observations in audit metadata and the data.
            if ax is b and n=='FETD-D + WLS':continue
            st=style(n)
            if ax is b and n=='SourceBridge':st.update(lw=1.9,ms=4.2,zorder=6)
            ax.plot(xs,yy,**st)
            panel_values['a' if ax is a else 'b'][n]=yy
        all_values[n]=vals
    assert max(max(v) for v in all_values.values()) < 16
    a.set(xlim=(-85,85),xticks=[-80,0,80],ylim=(0,8),yticks=[0,4,8],xlabel='Target shift',ylabel=r'RMSE ($\mu$g/m$^3$)')
    b.set(xlim=(-.2,6.2),xticks=range(7),xticklabels=['0','250','500','0','250','500','R'],ylim=(0,16),yticks=[0,8,16],xlabel='Anchor center $q$ / R',ylabel=r'RMSE ($\mu$g/m$^3$)')
    b.tick_params(axis='x',labelrotation=60)
    # Gain is categorical, so never join the two gain groups or reversal as a continuous stress axis.
    for line in b.lines:
        x=line.get_xdata();y=line.get_ydata();line.set_data(np.insert(np.asarray(x,float),[3,6],[np.nan,np.nan]),np.insert(np.asarray(y,float),[3,6],[np.nan,np.nan]))
    for p,n in [('exact','Exact'),('heterogeneous','Heterogeneous')]:
        yy=[next(r['feasible'] for r in D['public']['admission'] if r['profile']==p and r['r']==rad)/28*100 for rad in [20,50,100,150]]
        c.plot([20,50,100,150],yy,**style(n))
        ecdf(d,[r['empirical_worst_absolute_error']/r['prequery_radius'] for r in V['public_radius_events'] if r['profile']==p],n)
    c.set(xlim=(12,158),ylim=(-3,105),xticks=[20,50,100,150],yticks=[0,50,100],xlabel='Requested radius $r$',ylabel='Admitted pools (%)')
    d.set(xlim=(0,.24),ylim=(0,104),xticks=[0,.1,.2],yticks=[0,50,100],xlabel='Worst error / radius',ylabel='Targets (%)')
    method_keys=handles(names)
    for handle in method_keys:
        if handle.get_label()=='FETD-D + WLS':handle.set_label('FETD-D + WLS (a)')
        if handle.get_label()=='SourceBridge':handle.set_linewidth(1.9)
    group_legend(fig,method_keys,.285,y=.004,ncol=3)
    group_legend(fig,handles(['Exact','Heterogeneous']),.815,y=.025,ncol=1)
    save(fig,'robustness',{'sources':['metrics.json','plot_metrics.json'],'scope':'all 14 fixed conditions, heterogeneous profile in panels a/b; both profiles in c/d. Panel a retains all seven method/control series. Panel b shows six method series; FETD-D + WLS is tabulated separately; its full comparison remains in Tables II/III. The nominal estimator equals SourceBridge and is explained in prose without a duplicate graphical series. Two whole-packet translations remain separate controls.','anchor_axis':'first 3 q values g=0.7; next 3 g=1.3; R reversed anchors; line gaps separate gain groups/reversal','attack_rmse':all_values,'displayed_attack_rmse':panel_values,'axis_limits':{'a':[0,8],'b':[0,16]},'legend_scope':'FETD-D + WLS (a) explicitly restricts that key to panel a; the other six method keys apply to both a and b','panels':['all target shifts plus clean zero','six anchor spoofing conditions plus reversal, six displayed series','all 56 public-pool admission rates','all 426 target/profile error/radius observations']})

def scaling():
    fig,(a,b,c,d)=panels(['Catalog and selection','$K=1$: complete batch','$K=10$: complete batch','Inference modules'],height=2.25,bottom=.35,top=.82)
    rows=D['scaling']['groups'];ns=[r['n'] for r in rows]
    for key,n in [('catalog_median_seconds','LP catalog'),('old_phase_median_of_medians_seconds','Enumeration'),('new_phase_median_of_medians_seconds','SourceBridge')]:
        a.plot(ns,[1000*r[key] for r in rows],**style(n))
    a.set(yscale='log',ylim=(1,12000),yticks=[1,100,10000],ylabel='Time (ms, log)');logn(a,ns)
    for ax,K in [(b,1),(c,10)]:
        rr=[r for r in E['groups'] if r['K']==K]
        for method,n in [('enumeration','Enumeration'),('fast','SourceBridge')]:
            vals=[r['methods'][method]['whole_batch_seconds'] for r in rr];y=np.array([r['median'] for r in vals]);lo=np.array([r['min'] for r in vals]);hi=np.array([r['max'] for r in vals]);ax.errorbar([r['n'] for r in rr],y,yerr=[y-lo,hi-y],capsize=2,**style(n))
        ax.set(yscale='log',ylim=(.04,60),yticks=[.1,1,10],ylabel='Batch time (s, log)');logn(ax)
    vals=[[r[k] for r in T['records']] for k in ['proper_core_ms','certificate_ms','projection_ms']];assert all(len(v)==6469 for v in vals)
    bb=d.boxplot(vals,positions=range(3),widths=.5,patch_artist=True,whis=1.5,showfliers=True,
       medianprops=dict(color='black',lw=1),boxprops=dict(edgecolor='black',lw=.7),whiskerprops=dict(color='black',lw=.7),capprops=dict(color='black',lw=.7),
       flierprops=dict(marker='.',markersize=1.3,markeredgecolor='#555555',alpha=.5))
    for patch,col in zip(bb['boxes'],['#AACDD5','#B9CABA','#DEA4A0']):patch.set_facecolor(col)
    d.set(yscale='log',ylim=(.018,650),xticks=range(3),xticklabels=['Nominal','Cert.','Proj.'],yticks=[.1,10,100],ylabel='Time (ms, log)',xlabel='Inference module');d.yaxis.set_minor_locator(NullLocator())
    legend(fig,['SourceBridge','Enumeration','LP catalog'],ncol=3)
    save(fig,'scaling',{'sources':['metrics.json','end_to_end_verified.json','inference_timing_distribution.json'],'scope':'16 matched selection conditions; 3 runs per batch condition; 6,469 recorded distinct fits per module','interval':'complete batch medians and min/max of 3 alternating runs; inference quartiles, 1.5-IQR whiskers, every outlier','panels':['catalog and selection','K=1 whole batches','K=10 whole batches','complete inference timing distributions']})

def coverage():
    fig,(a,b,c,d)=panels(['Exact feasibility','Heterog. feasibility','Request-level savings','Seed-level savings'],height=2.25,bottom=.36,top=.82)
    spanstyles={20:('#555555','-','o'),150:('#3973A6','--','s'),450:('#8859A3','-.','^')}
    for ax,p in [(a,'exact'),(b,'heterogeneous')]:
        for span,(color,ls,mk) in spanstyles.items():
            yy=[100*sum(r['feasible'] for r in V['procurement_requests'] if r['profile']==p and r['span']==span and r['r']==rad)/20 for rad in [20,50,100,150]]
            ax.plot([20,50,100,150],yy,color=color,ls=ls,marker=mk,ms=3.5,mfc='white',lw=1,label=f'Span {span}')
        ax.set(xlim=(12,158),ylim=(-3,105),xticks=[20,50,100,150],yticks=[0,50,100],xlabel='Requested radius $r$',ylabel='Feasible requests (%)')
    seed_values={}
    for i,m in enumerate(COMPS):
        ecdf(c,[100*r['saving'][m] for r in FEAS],m)
        vv=[100*np.mean([r['saving'][m] for r in FEAS if r['seed']==seed]) for seed in sorted({r['seed'] for r in FEAS})];seed_values[m]=vv;box(d,[vv],[i],m,scatter=True,width=.5)
    c.set(xlim=(-1,45),ylim=(0,104),xticks=[0,20,40],yticks=[0,50,100],xlabel='Fee reduction (%)',ylabel='Requests (%)')
    d.set(xticks=range(3),xticklabels=['JB','Greedy','Cleanup'],xlim=(-.6,2.6),ylim=(-.5,21),yticks=[0,10,20],xlabel='Comparator',ylabel='Mean fee reduction (%)')
    hh=[Line2D([0],[0],color=c,ls=ls,marker=m,ms=3.5,mfc='white',lw=1,label=f'Span {s}') for s,(c,ls,m) in spanstyles.items()]
    group_legend(fig,hh,.27,ncol=3)
    group_legend(fig,handles(COMPS),.77,ncol=2)
    save(fig,'supp_procurement_coverage',{'sources':['plot_metrics.json'],'scope':'all 480 requests, all 329 feasible requests, all 20 seed clusters','seed_mean_savings_percent':seed_values,'panels':['all exact feasibility strata','all heterogeneous feasibility strata','all request savings','all seed-level means']})

def inference():
    fig,axs=panels(['Exact: radius and error','Heterog.: radius and error','Exact: observed error','Heterog.: observed error'],height=2.22,bottom=.35,top=.82)
    for i,p in enumerate(['exact','heterogeneous']):
        rows=sorted([r for r in V['public_radius_events'] if r['profile']==p],key=lambda r:(r['prequery_radius'],r['day'],r['station']));assert len(rows)==213
        axs[i].plot(range(1,214),[r['prequery_radius'] for r in rows],color='#555555',ls='--',lw=1,label='Purchased radius')
        axs[i].plot(range(1,214),[r['empirical_worst_absolute_error'] for r in rows],**style('SourceBridge'),markevery=31,label='SourceBridge')
        axs[i].set(xlim=(1,213),ylim=(0,160),xticks=[1,100,213],yticks=[0,80,160],xlabel='Target rank by radius',ylabel=r'Radius / error ($\mu$g/m$^3$)')
        values={}
        for n in ['SourceBridge','Midpoint']:
            vv=[float(np.sqrt(r['worst_se'])) for r in D['public']['event_stats'] if r['profile']==p and r['label']==KEYS[n]];assert len(vv)==213;values[n]=vv;ecdf(axs[i+2],vv,n)
        nominal=[float(np.sqrt(r['worst_se'])) for r in D['public']['event_stats'] if r['profile']==p and r['label']=='MLNI_JSAC2022__proper_public_WLS']
        assert np.array_equal(values['SourceBridge'],nominal)
        axs[i+2].set(xlim=(0,25),ylim=(0,104),xticks=[0,10,20],yticks=[0,50,100],xlabel='Worst absolute error',ylabel='Targets (%)')
    hh=[Line2D([0],[0],color='#555555',ls='--',lw=1,label='Purchased radius')]+handles(['SourceBridge','Midpoint'])
    group_legend(fig,hh,.5,ncol=3)
    save(fig,'supp_conditional_inference',{'sources':['metrics.json','plot_metrics.json'],'scope':'all 426 target/profile events; observed error finite attack-library maximum; certificate radius computed before reports','equality':'proper MLNI and SourceBridge event errors exactly equal in both profiles','panels':['exact radius/error','heterogeneous radius/error','exact all-event ECDF','heterogeneous all-event ECDF']})

def reuse():
    fig,(a,b,c,d)=panels(['Matched selections','$K=1$: amortized','$K=10$: amortized','Catalog share'],height=2.27,bottom=.36,top=.82)
    raw=[r for r in RAW['records'] if not r['warmup']];assert len(raw)==36
    cols={8:('#555555','o'),16:('#238064','s'),32:('#3973A6','^'),64:('#8859A3','D')}
    a.plot([1,10000],[1,10000],color='#999999',ls=':',lw=.7)
    for n,(color,mk) in cols.items():
        rr=[r for r in D['scaling']['records'] if r['n']==n];a.scatter([1000*r['old_phase_median_seconds'] for r in rr],[1000*r['new_phase_median_seconds'] for r in rr],s=15,color=color,marker=mk,edgecolor='white',lw=.3)
    a.set(xscale='log',yscale='log',xlim=(1,10000),ylim=(1,10000),xticks=[1,100,10000],yticks=[1,100,10000],xlabel='Enumeration time (ms)',ylabel='SourceBridge time (ms)');a.xaxis.set_minor_locator(NullLocator());a.yaxis.set_minor_locator(NullLocator())
    derived=[]
    for K,ax in [(1,b),(10,c)]:
        for method,name in [('enumeration','Enumeration'),('fast','SourceBridge')]:
            tt=[];ss=[]
            for n in [8,32,64]:
                rr=[r for r in raw if r['n']==n and r['K']==K and r['method']==method];assert len(rr)==3
                times=np.array([r['whole_batch_seconds']/K for r in rr]);shares=np.array([100*r['catalog_seconds']/r['whole_batch_seconds'] for r in rr]);tt.append(times);ss.append(shares);derived.append(dict(K=K,n=n,method=method,amortized_seconds=times.tolist(),catalog_share_percent=shares.tolist()))
            tt=np.array(tt);ss=np.array(ss)
            for target,v in [(ax,tt),(d,ss)]:
                med=np.median(v,axis=1);st=style(name)
                st.update(mfc='white' if K==1 else st['color'])
                target.errorbar([8,32,64],med,yerr=[med-v.min(axis=1),v.max(axis=1)-med],capsize=2,**st)
        ax.set(yscale='log',ylim=(.004,10),yticks=[.01,.1,1,10],ylabel='Time per query (s, log)');logn(ax)
    d.set(ylim=(0,105),yticks=[0,50,100],ylabel='Catalog / batch (%)');logn(d)
    hh=[Line2D([0],[0],marker=m,ls='none',color=c,mfc=c,ms=3.5,label=f'$n={n}$') for n,(c,m) in cols.items()]
    group_legend(fig,hh,.16,ncol=2)
    hh=[]
    for n in ['SourceBridge','Enumeration']:
        for K in [1,10]:
            st=style(n);st['mfc']='white' if K==1 else st['color']
            hh.append(Line2D([0],[0],**st,label=f'{n}, $K={K}$'))
    group_legend(fig,hh,.63,ncol=2)
    save(fig,'catalog_reuse',{'sources':['metrics.json','end_to_end_raw.json'],'scope':'16 selection conditions and all 36 retained batch measurements; warmups omitted from summaries but retained in source','interval':'batchwise derived ratios; medians and complete 3-run ranges','derived_batches':derived,'panels':['all matched phase times','K=1 amortized cost','K=10 amortized cost','catalog fraction of whole batch']})

if __name__=='__main__':
    for fn in [procurement,regimes,mechanism,robustness,scaling,coverage,inference,reuse]:fn()
    manifest={'font':'Times New Roman / STIX math','publication_width_inches':7.16,'layout':'1 x 4','minimum_font_pt':7.5,'styles':STYLE,
      'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.json'))},'figures':META,'horizontal_grid':GRID_STYLE,
      'rendering':'Data-derived coordinates and intervals. Horizontal guides: 0.28-pt major and 0.16-pt minor; categorical rows use one guide per stratum.',
      'display_scope':'Paired accuracy intervals appear in a manuscript table. Proper MLNI and SourceBridge coincide, so a single curve represents their predictions. FETD-D + WLS appears in robustness(a); the complete numerical comparison appears in Tables II and III.',
      'evidence_scope':'All plotting inputs retain their recorded data and uncertainty definitions.'}
    (ROOT/'data_mapping_and_style.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'visual_qa.json').write_text(json.dumps(QA,indent=2),encoding='utf-8')
    for language in ['en','zh']:
        target=ROOT.parents[1]/f'manuscript_{language}'/'figures'
        for filename,is_supp in [('experimental_figure_sources.json',False),('supplement_figure_sources.json',True)]:
            local={'source_bundle':'../../code/plots',
                   'full_manifest':'../../code/plots/data_mapping_and_style.json',
                   'font':manifest['font'],'layout':'1 x 4, 7.16 inches',
                   'figures':{k:v for k,v in META.items() if k.startswith('supp_')==is_supp},
                   'evidence_scope':'Plotting data and uncertainty are preserved; paired accuracy differences are tabulated.'}
            (target/filename).write_text(json.dumps(local,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'figures':len(META),'qa':QA},ensure_ascii=True))
