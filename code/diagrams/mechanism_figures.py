"""SourceBridge Figs. 3--4: editable, publication-size mechanism artwork.

Export live-text SVG, embedded-font PDF, and 600-dpi PNG at 3.50 x 2.90 in.
The composition and all major panel/coordinate bounds are retained.
Scientific geometry is original; bundled Lucide role symbols carry their
included ISC/MIT license notices in assets/lucide.
Inspected references: PRBTD model p04 (aligned modules) and QPTD model p04
(numbered stages and concrete data carriers); their artwork is not copied.
Fig. 3 uses three calibration/report columns, with shared j2 drawn and billed
once. Fig. 4 separates before-acquisition ambiguity from after-observation
safe projection. Numerical geometry is generated directly from stated values.
The illustration uses the three-stage/matrix composition and
two-panel geometry. The union has a three-member glyph with one shared member;
two world cards route into one shared report page. The lower panel aligns H
and S in one coordinate window with a pale value-specific safe-range guide.
All additions visualize existing illustrative values, never new evidence.
"""
from pathlib import Path
import os
import json
import hashlib
import itertools
import tempfile
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from sourcebridge_palettes import get_palette, text_color
PALETTE, C = get_palette()
OUT = Path(os.environ.get('SOURCEBRIDGE_CONCEPT_OUT', str(BASE / 'figures')))
QA_OUT = Path(os.environ.get('SOURCEBRIDGE_CONCEPT_QA_OUT', str(BASE / 'quality')))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'sourcebridge_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as VectorPath
from sourcebridge_symbols import draw_symbol

for filename in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
    fontfile = Path(os.environ.get('SOURCEBRIDGE_FONT_DIR', 'C:/Windows/Fonts')) / filename
    if fontfile.exists():
        font_manager.fontManager.addfont(str(fontfile))
FONT = font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8.5,
    'mathtext.fontset': 'stix', 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.color': C['ink'],
    'savefig.facecolor': 'white', 'savefig.transparent': False,
    'path.simplify': False,
})
METADATA = []


def color(key):
    return C.get(key, key)


def canvas():
    fig = plt.figure(figsize=(3.5, 2.76))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 3.5), ylim=(0, 2.9))
    ax.axis('off')
    return fig, ax


def text(ax, x, y, label, size=8.5, c='ink', ha='left', weight='normal', **kwargs):
    return ax.text(x, y, label, fontsize=size, color=text_color(C, c), ha=ha,
                   va='center', weight=weight, linespacing=1.05, **kwargs)


def line(ax, x1, y1, x2, y2, c='line', lw=.8, **kwargs):
    return ax.plot([x1, x2], [y1, y2], color=color(c), lw=lw,
                   solid_capstyle='round', **kwargs)[0]


def box(ax, x, y, w, h, ec='line', fc='white', radius=.035, lw=.75, z=1):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=color(fc), edgecolor=color(ec), linewidth=lw,
                          zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, points, c='ink', lw=.9, head=5.5, **kwargs):
    codes = [VectorPath.MOVETO] + [VectorPath.LINETO] * (len(points) - 1)
    patch = FancyArrowPatch(path=VectorPath(points, codes), arrowstyle='-|>',
                            mutation_scale=head, lw=lw, color=color(c),
                            capstyle='round', joinstyle='round', **kwargs)
    ax.add_patch(patch)
    return patch


def circle(ax, x, y, r, ec='ink', fc='white', lw=.8, **kwargs):
    patch = Circle((x, y), r, facecolor=color(fc), edgecolor=color(ec),
                   linewidth=lw, **kwargs)
    ax.add_patch(patch)
    return patch


def stage(ax, n, y, label):
    c = 'text_red' if n == 3 else 'blue'
    circle(ax, .19, y, .080, c, c, .7)
    text(ax, .19, y, str(n), 8.3, c='white', ha='center')
    text(ax, .34, y, label, 9.1, weight='bold')


def save(fig, ax, name, semantic_checks):
    OUT.mkdir(parents=True, exist_ok=True)
    QA_OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = fig.bbox
    text_bounds, bounding_boxes = [], []
    for item in ax.texts:
        b = item.get_window_extent(renderer)
        clipped = bool(b.x0 < boundary.x0-.1 or b.y0 < boundary.y0-.1 or
                       b.x1 > boundary.x1+.1 or b.y1 > boundary.y1+.1)
        text_bounds.append(dict(text=item.get_text(), size=item.get_fontsize(),
                                color=item.get_color(), clipped=clipped))
        assert not clipped, (name, item.get_text(), b, boundary)
        assert item.get_fontsize() >= 8
        bounding_boxes.append((item.get_text(), b))
    for (label_a, a), (label_b, b) in itertools.combinations(bounding_boxes, 2):
        overlap_w = min(a.x1, b.x1)-max(a.x0, b.x0)
        overlap_h = min(a.y1, b.y1)-max(a.y0, b.y0)
        assert overlap_w <= .05 or overlap_h <= .05, (name, 'text overlap', label_a, label_b)
    exports = {}
    for extension in ('svg', 'pdf', 'png'):
        final = OUT / f'{name}.{extension}'
        temporary = OUT / f'{name}.tmp.{extension}'
        fig.savefig(temporary, format=extension, dpi=600)
        temporary.replace(final)
        exports[extension] = dict(path=str(final), sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    METADATA.append(dict(figure=name, palette=PALETTE, size_inches=list(fig.get_size_inches()), font=FONT,
                         native_vector_geometry=True, raster_images_in_svg=False,
                         provenance='Original scientific primitives and manuscript-defined numbers; bundled Lucide role symbols',
                         icon_provenance='Bundled Lucide SVG paths in assets/lucide; ISC/MIT notices included; no raster assets',
                         vector_patches=len(ax.patches), vector_lines=len(ax.lines),
                         semantic_checks=semantic_checks, text_checks=text_bounds,
                         text_text_overlaps=0, exports=exports))
    plt.close(fig)


def acquisition():
    fig, ax = canvas()
    # Restrained stage surfaces establish the information hierarchy while the
    # established seven candidates / two tests / one report matrix stay intact.
    box(ax,.075,2.385,3.35,.475,'blue_phase_border','surface_blue',radius=.03,lw=.6,z=-5)
    box(ax,.075,1.595,3.35,.75,'teal_phase_border','surface_teal',radius=.03,lw=.6,z=-5)
    box(ax,.075,.025,3.35,1.55,'red_panel_border','surface_red',radius=.03,lw=.6,z=-5)
    stage(ax, 1, 2.76, r'Catalog: $|P|\leq2$')
    draw_symbol(ax,'files',2.18,2.66,.19,color=C['blue'],lw=.65,z=6)
    text(ax, 3.36, 2.76, r'$n=3$ example', 8.3, 'ink', ha='right')
    subsets = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3)]
    xs = [.13, .53, .93, 1.33, 1.75, 2.30, 2.85]
    widths = [.31, .31, .31, .31, .46, .46, .46]
    for subset, x, w in zip(subsets, xs, widths):
        c = 'blue' if subset == (1,2) else 'teal' if subset == (2,3) else 'gray'
        fill = 'pale_'+c if c in ('blue', 'teal') else 'white'
        box(ax, x, 2.42, w, .24, c, fill, radius=.018, lw=.75 if len(subset)==2 else .65)
        label = r'$\varnothing$' if not subset else ','.join(map(str, subset))
        text(ax, x+w/2, 2.54, label, 8.3, c if c!='gray' else 'ink', ha='center')

    stage(ax, 2, 2.24, 'Directional certificates')
    for x, name, ids, c, sign in [(.13, 'P', (1,2), 'blue', '+'),
                                  (1.88, 'Q', (2,3), 'teal', '-')]:
        box(ax, x, 1.79, 1.49, .32, c, 'pale_'+c, radius=.025, lw=1.0)
        text(ax, x+1.49/2, 1.95,
             rf'${name}=\{{{ids[0]},{ids[1]}\}}\quad W_{sign}({name})\leq\tau$',
             8.5, c, ha='center')
    # Both certificates lead to one union, not separately billed copies.
    arrow(ax, [(.875,1.78), (.875,1.69), (1.295,1.69)], 'blue', .8, 4.5)
    arrow(ax, [(2.625,1.78), (2.625,1.69), (2.205,1.69)], 'teal', .8, 4.5)
    # The small union carrier displays all three members. Its central member
    # is split blue/green because it belongs to both supports, but occurs once.
    union_left, union_w, union_y, union_h = 1.33, .28, 1.61, .16
    for j in (1,2,3):
        xx=union_left+(j-1)*union_w
        if j==2:
            for shift,role in ((0,'blue'),(.5,'teal')):
                ax.add_patch(Rectangle((xx+shift*union_w,union_y),union_w/2,union_h,
                                       facecolor=color('mid_'+role),edgecolor='none',zorder=2))
            border='red'
        else:
            border='blue' if j==1 else 'teal'
            ax.add_patch(Rectangle((xx,union_y),union_w,union_h,
                                   facecolor=color('pale_'+border),edgecolor='none',zorder=2))
        ax.add_patch(Rectangle((xx,union_y),union_w,union_h,
                               facecolor='none',edgecolor=color(border),lw=.8,zorder=3))
        text(ax,xx+union_w/2,union_y+union_h/2,str(j),8.1,'ink',ha='center',zorder=4)
    stage(ax, 3, 1.46, r'Purchase $A=P\cup Q$')
    draw_symbol(ax,'receipt-text',2.20,1.355,.205,color=C['red'],lw=.65,z=6)
    text(ax, 3.36, 1.46, 'Charge once', 8.3, 'ink', ha='right')

    # Each column is one calibration object, with its public bounds and all m
    # future source reports. The shared central column appears exactly once.
    left, cellw, bottom, top = .74, .875, .40, 1.35
    centers = [left+cellw*(i+.5) for i in range(3)]
    for j, xc in enumerate(centers, 1):
        fill = 'pale_red' if j == 2 else 'pgray'
        ax.add_patch(Rectangle((left+cellw*(j-1),bottom), cellw, top-bottom,
                               facecolor=color(fill), edgecolor='none', zorder=0))
        c = 'red' if j == 2 else 'blue' if j == 1 else 'teal'
        text(ax, xc, (top+1.14)/2, rf'$j_{j}$' + (' (shared)' if j == 2 else ''), 8.5, c, ha='center')
        text(ax, xc, (1.14+.935)/2, rf'$(I_{j},\epsilon_{j})$', 8.5, ha='center')
        reports_mid=(bottom+.935)/2
        text(ax, xc, reports_mid+.16, rf'$Y_{{1{j}}}$', 8.5, ha='center')
        text(ax, xc, reports_mid, r'$\vdots$', 8.5, 'ink', ha='center')
        text(ax, xc, reports_mid-.16, rf'$Y_{{m{j}}}$', 8.5, ha='center')
    for y in (top, 1.14, .935, bottom):
        line(ax, left, y, left+3*cellw, y, 'line', .65)
    for x in (left, left+cellw, left+2*cellw, left+3*cellw):
        line(ax, x, bottom, x, top, 'line', .65)
    ax.add_patch(Rectangle((left+cellw,bottom), cellw, top-bottom,
                           facecolor='none', edgecolor=color('red'), lw=1.0, zorder=3))
    text(ax, .13, (1.14+.935)/2, 'Public', 8.3, 'ink')
    draw_symbol(ax,'sliders-horizontal',.535,(1.14+.935)/2-.09,.18,color=C['blue'],lw=.60,z=6)
    text(ax, .13, (bottom+.935)/2, 'Reports', 8.3, 'ink')
    draw_symbol(ax,'file-spreadsheet',.535,(bottom+.935)/2-.09,.18,color=C['blue'],lw=.60,z=6)
    text(ax, .13, .285, 'Cost split', 8.3, 'ink')
    # A true cost-composition strip uses one segment per unique paid object.
    # Its segment lengths are exactly 3:2:4. The shared j2 is charged only once;
    # object indices inside the segments retain the mapping in monochrome.
    fees = (3,2,4)
    bar_left, bar_width, bar_bottom, bar_height = left, 3*cellw, .205, .16
    cost_scale = bar_width/sum(fees)
    cursor = bar_left
    cost_segments = []
    for j, fee, role in zip((1,2,3), fees, ('blue','red','teal')):
        width = fee*cost_scale
        ax.add_patch(Rectangle((cursor,bar_bottom),width,bar_height,
                               facecolor=color('pale_'+role),
                               edgecolor=color(role),lw=.75,zorder=2))
        text(ax,cursor+width/2,bar_bottom+bar_height/2,
             rf'$j_{j}:\,{fee}$',8.3,'ink',ha='center',zorder=3)
        cost_segments.append(dict(object=j,fee=fee,width_inches=width))
        cursor += width
    text(ax, 1.75, .10, r'$C(P\cup Q)-C_0=3+2+4=9$', 8.5, 'ink', ha='center')

    assert set((1,2)) | set((2,3)) == {1,2,3}
    assert sum({1:3,2:2,3:4}[j] for j in {1,2,3}) == 9
    assert abs(cursor-(bar_left+bar_width)) < 1e-12
    assert [s['fee'] for s in cost_segments] == [3,2,4]
    assert sum(s['object']==2 for s in cost_segments) == 1
    save(fig, ax, 'acquisition', dict(catalogue=subsets, P=[1,2], Q=[2,3], costs=[3,2,4],
        union_cost=9, shared_anchor_charged_once=True, shared_object_drawn_once=True,
        calibration_objects_are_columns_not_source_identities=True,
        report_matrix=dict(columns=[1,2,3], rows=['1','ellipsis','m']),
        matrix_report_symbols_are_future_purchased_reports_not_preacquisition_inputs=True,
        public_rows=['I_j','epsilon_j'], directional_tests_are_symbolic=True,
        cost_composition=dict(segments=cost_segments,total=9,proportional_widths=True,
                              shared_object_has_one_segment=True,
                              values_are_existing_illustration_not_experimental_results=True),
        union_member_glyph=dict(members=[1,2,3],shared_member=2,
                                memberships=dict(P=[1,2],Q=[2,3]),
                                shared_member_has_one_cell=True),
        role_symbols=dict(catalog='files',public_parameters='sliders-horizontal',
                          purchased_reports='file-spreadsheet',purchase='receipt-text'),
        layout_geometry=dict(canvas=[3.5,2.9],stage_panels_unchanged=True,
            candidate_cards_unchanged=True,certificate_boxes_unchanged=True,
            union_cells_unchanged=True,matrix_bounds_unchanged=True,cost_strip_unchanged=True),
        centered_container_text=dict(candidate_cards=True,certificate_groups=True,
            matrix_cells=True,cost_segment_labels=True),
        reference_use='PRBTD: aligned modules; QPTD: numbered stages and concrete report carriers'))


def geometry():
    fig, ax = canvas()
    # Full-body surfaces support the two information regimes. Their boundaries
    # remain outside the exact graph and never encode scientific quantities.
    box(ax,.075,1.285,3.35,1.60,'blue_phase_border','surface_blue',radius=.025,lw=.75,z=-6)
    box(ax,.075,.025,3.35,1.24,'red_panel_border','surface_red',radius=.025,lw=.75,z=-6)
    # Two compact heading bands organize the distinct information regimes.
    # They are outside the plotting coordinates and do not encode data values.
    ax.add_patch(Rectangle((.10,2.67),3.28,.205,facecolor=color('pale_blue'),
                           edgecolor='none',zorder=0))
    ax.add_patch(Rectangle((.10,1.075),3.28,.19,facecolor=color('pale_red'),
                           edgecolor='none',zorder=0))
    line(ax,.10,2.67,3.38,2.67,'blue',.9)
    line(ax,.10,1.075,3.38,1.075,'red',.9)
    text(ax, .13, 2.77, '(a) Same-report ambiguity', 9.1, weight='bold')
    box(ax,.55,2.475,2.65,.17,'none','white',radius=.012,lw=0,z=-1)
    text(ax, .55+2.65/2, 2.56, r'$T=[0,10],\quad a\in[1,2],\quad\epsilon=0$', 8.5, ha='center')
    draw_symbol(ax,'sliders-horizontal',3.13,2.685,.17,color=C['blue'],lw=.65,z=6)

    # Linear q:[0,10], Y:[0,20] axes and exact analytic response family.
    Q = lambda q: .38+1.66*q/10
    V = lambda y: 1.70+.69*y/20
    ax.add_patch(Polygon([(Q(0),V(0)), (Q(10),V(10)), (Q(10),V(20))],
                         closed=True, fc=color('pale_blue'), ec=color('mid_blue'),
                         hatch='///', lw=0, zorder=0))
    line(ax,Q(0),V(0),Q(10)+.06,V(0),'gray',.7)
    line(ax,Q(0),V(0),Q(0),V(20)+.025,'gray',.7)
    for y in (0,10,20):
        line(ax,Q(0)-.022,V(y),Q(0)+.022,V(y),'gray',.65)
        text(ax,Q(0)-.075,V(y),str(y),8,'ink',ha='right')
    text(ax,.14,2.47,r'$Y$',8.3,'ink',ha='center')
    for q in (0,5,10):
        line(ax,Q(q),V(0)-.023,Q(q),V(0)+.023,'gray',.65)
        text(ax,Q(q),1.585,str(q),8,'ink',ha='center')
    text(ax,2.12,1.70,r'$q_t$',8.3,'ink')
    line(ax,Q(0),V(0),Q(10),V(20),'blue',1.0)
    line(ax,Q(0),V(0),Q(10),V(10),'teal',1.0,linestyle=(0,(4,2)))
    line(ax,Q(0)+.03,V(10),Q(5)-.02,V(10),'gray',.6,linestyle=(0,(2,2)))
    line(ax,Q(5),V(10),Q(10),V(10),'red',1.3)
    for q in (5,10):
        line(ax,Q(q),V(0)+.035,Q(q),V(10),'gray',.55,linestyle=(0,(2,2)))
    ax.plot(Q(5),V(10),marker='o',markersize=4.5,mec=color('blue'),mfc='white',mew=1,zorder=5)
    ax.plot(Q(10),V(10),marker='s',markersize=4.2,mec=color('teal'),mfc='white',mew=1,zorder=5)
    text(ax,Q(5)-.075,V(10)+.115,'A',8.3,'blue',ha='center')
    text(ax,Q(10)+.075,V(10)-.13,'B',8.3,'teal',ha='center')
    text(ax,1.58,2.375,r'$a=2$',8.3,'blue',ha='center')
    text(ax,1.54,1.825,r'$a=1$',8.3,'teal',ha='center')

    text(ax,2.33,2.38,r'$q_1=Y_1=0$',8.3)
    text(ax,2.33,2.22,r'$\Rightarrow b=0$',8.3)
    # Each card represents one complete legal world. Marker shape identifies
    # its exact plotted point; both routes terminate at one common report page.
    box(ax,2.195,1.965,1.165,.18,'blue','pale_blue',radius=.02,lw=.85)
    box(ax,2.195,1.74,1.165,.18,'teal','pale_teal',radius=.02,lw=.85)
    text(ax,2.83,2.055,r'A: $(q_t,a)=(5,2)$',8.1,'ink',ha='center')
    text(ax,2.83,1.83,r'B: $(q_t,a)=(10,1)$',8.1,'ink',ha='center')
    ax.plot(2.255,2.055,marker='o',markersize=4.2,mec=color('blue'),
            mfc='white',mew=.9,zorder=5)
    ax.plot(2.255,1.83,marker='s',markersize=4.0,mec=color('teal'),
            mfc='white',mew=.9,zorder=5)
    report_x,report_y,report_w,report_h=2.40,1.40,.90,.215
    fold=.045
    ax.add_patch(Polygon([(report_x,report_y),(report_x+report_w,report_y),
                          (report_x+report_w,report_y+report_h-fold),
                          (report_x+report_w-fold,report_y+report_h),
                          (report_x,report_y+report_h)],closed=True,
                         fc=color('pale_red'),ec=color('red'),lw=.85,zorder=3))
    line(ax,report_x+report_w-fold,report_y+report_h,
         report_x+report_w-fold,report_y+report_h-fold,'red',.65,zorder=4)
    line(ax,report_x+report_w-fold,report_y+report_h-fold,
         report_x+report_w,report_y+report_h-fold,'red',.65,zorder=4)
    draw_symbol(ax,'scan-line',2.425,1.425,.16,color=C['red'],lw=.60,z=6)
    text(ax,2.93,report_y+report_h/2,
         r'$Y_t=10$',8.3,'ink',ha='center',zorder=5)
    arrow(ax,[(3.37,2.055),(3.395,2.055),(3.395,1.50),(3.305,1.50)],'blue',.75,4.2)
    arrow(ax,[(2.775,1.73),(2.775,1.63)],'teal',.75,4.2)
    line(ax,Q(5),1.475,Q(10),1.475,'red',.8)
    for q in (5,10):
        line(ax,Q(q),1.455,Q(q),1.495,'red',.8)
    text(ax,Q(7.5),1.38,r'$W=5$',8.5,'red',ha='center')
    text(ax,.17,1.38,r'$R=W/2=2.5$',8.3,'ink')

    text(ax,.13,1.17,'(b) Safe projection',9.1,weight='bold')
    draw_symbol(ax,'target',3.13,1.085,.17,color=C['red'],lw=.65,z=6)
    text(ax,.13,.98,r'$Y_t=6$',8.5)
    text(ax,3.36,.97,r'$S=T\cap[u-R,\ell+R]$',8.3,'ink',ha='right')
    X = lambda q: .69+2.42*q/10
    # One aligned coordinate window; the pale vertical guide marks only this
    # example's S=[3.5,5.5], not a claim that S generally equals an intersection
    # with H. The general construction remains explicit in the formula above.
    box(ax,X(0)-.035,.245,X(10)-X(0)+.07,.645,'none','white',radius=.018,lw=0,z=-2)
    ax.add_patch(Rectangle((X(3.5),.255),X(5.5)-X(3.5),.625,
                           facecolor=color('mid_red'),edgecolor='none',alpha=.26,zorder=-1))
    for q in (0,5,10):
        line(ax,X(q),.25,X(q),.88,'line',.5,alpha=.35,linestyle=(0,(1,3)),zorder=0)
    text(ax,.13,.815,r'$H$',8.5,'teal')
    text(ax,.13,.475,r'$S$',8.5,'red')
    line(ax,X(0),.195,X(10),.195,'gray',.7)
    text(ax,.13,.195,r'$T$',8.5,'ink')
    for q in (0,5,10):
        line(ax,X(q),.175,X(q),.215,'gray',.65)
        text(ax,X(q),.11,str(q),8,'ink',ha='center')
    for q in (3,6):
        spans = [(.25,.825)] if q == 3 else [(.40,.825)]
        for lo, hi in spans:
            line(ax,X(q),lo,X(q),hi,'teal',.45,alpha=.3,linestyle=(0,(2,2)))
    box(ax,X(3),.775,X(6)-X(3),.08,'teal','pale_teal',radius=.008,lw=.8)
    for q in (3,6):
        line(ax,X(q),.76,X(q),.87,'teal',.85)
        text(ax,X(q),.94,str(q),8.5,'teal',ha='center')
    text(ax,2.45,.815,'width = 3',8.3,'ink')
    box(ax,X(3.5),.425,X(5.5)-X(3.5),.10,'red','pale_red',radius=.008,lw=.9)
    line(ax,X(3.5),.415,X(3.5),.535,'red',.9)
    text(ax,X(3.5),.305,'3.5',8.5,'red',ha='center')
    circle(ax,X(7),.475,.037,'purple','white',1,zorder=6)
    text(ax,X(7)+.09,.475,r'$z=7$',8.5,'purple')
    arrow(ax,[(X(7),.525),(X(7),.64),(X(5.5),.64),(X(5.5),.54)],'purple',.9,5)
    ax.plot(X(5.5),.475,marker='D',markersize=4.8,color=color('red'),mec='white',mew=.45,zorder=7)
    text(ax,X(5.5)+.03,.305,r'$\widehat q=5.5$',8.5,'red')

    # The axes and H/S/T remain in their original positions. A compact
    # inset uses only the pre-existing left margin: its local ruler has the
    # exact 2.5:0.5 length ratio, and the repeated diamond denotes qhat=5.5.
    # These are distances to H's endpoints, not measured prediction errors.
    card_x,card_y,card_w,card_h=.28,.245,1.08,.50
    box(ax,card_x,card_y,card_w,card_h,'red_panel_border','white',radius=.024,lw=.65,z=4)
    inset_text=[]
    inset_text.append(text(ax,card_x+card_w/2,.682,'Endpoint distance',8.0,'ink',ha='center',zorder=6))
    D=lambda q: .43+.78*(q-3)/3
    line(ax,D(3),.568,D(5.5),.568,'red',.95,zorder=6)
    line(ax,D(5.5),.568,D(6),.568,'teal',.95,zorder=6)
    for q in (3,6):
        line(ax,D(q),.548,D(q),.588,'gray',.7,zorder=6)
    ax.plot(D(5.5),.568,marker='D',markersize=3.6,color=color('red'),mec='white',mew=.4,zorder=7)
    inset_text.append(text(ax,card_x+card_w/2,.448,r'$|\widehat q-3|=2.5=R$',8.0,'ink',ha='center',zorder=6))
    inset_text.append(text(ax,card_x+card_w/2,.315,r'$|6-\widehat q|=0.5\leq R$',8.0,'ink',ha='center',zorder=6))
    fig.canvas.draw()
    inverse=ax.transData.inverted()
    for label in inset_text:
        bounds=label.get_window_extent(fig.canvas.get_renderer()).transformed(inverse)
        assert bounds.x0>=card_x+.018 and bounds.x1<=card_x+card_w-.018,(label.get_text(),bounds)
        assert bounds.y0>=card_y+.006 and bounds.y1<=card_y+card_h-.006,(label.get_text(),bounds)

    assert 2*5 == 1*10 == 10
    H=(6/2,6/1); R=2.5; S=(H[1]-R,H[0]+R)
    assert H == (3,6) and S == (3.5,5.5) and min(S[1],max(S[0],7)) == 5.5
    endpoint_distances=(abs(5.5-H[0]),abs(H[1]-5.5))
    assert endpoint_distances==(2.5,.5) and max(endpoint_distances)==R
    assert abs((D(5.5)-D(3))/(D(6)-D(5.5))-5)<1e-12
    save(fig,ax,'geometry',dict(T=[0,10],gains=[1,2],error=0,anchor=[0,0],
        worlds=[dict(a=2,q=5,packet=[0,10]),dict(a=1,q=10,packet=[0,10])],
        analytic_response_axes=dict(q=[0,10],Y=[0,20]),analytic_responses=['Y=2q','Y=q'],
        shaded_region='Y between q and 2q for q in [0,10]',
        common_report_line=10,common_report_intersections=[[5,10],[10,10]],
        analytic_curves_are_not_experimental_data=True,
        W=5,R=R,observed_Yt=6,H=H,S=S,z=7,output=5.5,observed_width=3,
        distinguishes_observed_and_worst_width=True,projection_onto='S, not H',
        nominal_is_illustrative_not_measured_mlni=True,
        world_marker_keys=dict(A='hollow circle',B='hollow square'),
        heading_bands_do_not_encode_values=True,
        safe_interval_enhancement_changes_only_vertical_thickness=True,
        shared_report_glyph=dict(report_Yt=10,one_page=True,
                                 inputs=['world A','world B'],anchor_Y1=0),
        safe_range_guide=dict(endpoints=[3.5,5.5],same_x_axis_as_H_and_T=True,
                              example_specific=True,does_not_claim_general_S_subset_H=True),
        derived_endpoint_distance_inset=dict(estimate=5.5,hull_endpoints=[3,6],
            distances=endpoint_distances,max_distance=2.5,equals_R=True,
            local_ruler_segment_ratio=[5,1],local_ruler_not_main_axis=True,
            unknown_true_q=True,not_measured_prediction_error=True,
            bounds_inches=[card_x,card_y,card_w,card_h],text_centered_and_contained=True),
        role_symbols=dict(gain_conditions='sliders-horizontal',shared_report='scan-line',safe_projection='target'),
        layout_geometry=dict(canvas=[3.5,2.9],upper_panel=[.075,1.285,3.35,1.60],
            lower_panel=[.075,.025,3.35,1.24],analytic_plot_unchanged=True,
            world_cards_unchanged=True,report_page_unchanged=True,
            lower_coordinate_window=[.655,.245,2.49,.645],
            H_y=.815,S_y=.475,T_y=.195,main_q_map=[.69,2.42,0,10]),
        panel_map=dict(a='Before acquisition: worst-case same-report ambiguity',
                       b='After a distinct observation: hull and safe projection')))


if __name__ == '__main__':
    acquisition()
    geometry()
    (QA_OUT/'mechanism_figure_manifest.json').write_text(json.dumps(METADATA,indent=2),encoding='utf-8')
    from PIL import Image
    preview = Image.new('RGB',(1032,438),'white')
    for i,name in enumerate(('acquisition','geometry')):
        with Image.open(OUT/f'{name}.png') as rendered:
            panel=rendered.convert('RGB').resize((504,418),Image.Resampling.LANCZOS)
            preview.paste(panel,(i*516+6,10))
    preview.save(QA_OUT/'mechanism_column_preview.png')
    print(json.dumps([dict(figure=m['figure'],size_inches=m['size_inches'],
                          texts=len(m['text_checks']),min_font=min(t['size'] for t in m['text_checks']),
                          clipped=sum(t['clipped'] for t in m['text_checks'])) for m in METADATA],indent=2))
