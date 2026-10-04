"""Original normalized vector illustrations for SourceBridge paper diagrams.

All geometry is authored here (no third-party assets, no raster). These icons
denote entities and data carriers, never measured data. Sizes use axis units;
stroke weights remain in points for consistency at the final paper size.
"""
from matplotlib.patches import Rectangle,FancyBboxPatch,Polygon,Circle,Ellipse,Arc,PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb

def draw_icon(ax,kind,x,y,w,h=None,*,color='#111214',accent='#8B2944',
              face='#F3F3F4',skin='#EBCBB4',hair='#181A1E',lw=.65,z=5):
    """Draw kind in x,y,w,h (bottom-left), returning its vector artists.

    Kinds: phone, participant, station, vehicle, building, air_target,
    report_packet, catalog, server, requester, coins, receipt, calibration_pin,
    shield, gauge, sensor. No text is inserted. Icons are inside their bounds.
    """
    h=w if h is None else h
    t=Affine2D().scale(w,h).translate(x,y)+ax.transData
    art=[]
    def add(p):
        p.set_transform(t);p.set_zorder(z);p.set_clip_on(False);ax.add_patch(p);art.append(p);return p
    def rect(x,y,w,h,fc=face,ec=color,r=0,l=lw):
        return add(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=l) if r else Rectangle((x,y),w,h,fc=fc,ec=ec,lw=l))
    def line(points,c=color,l=lw):return add(PathPatch(Path(points),fill=False,ec=c,lw=l,capstyle='round',joinstyle='round'))
    def poly(points,fc=face,ec=color,l=lw):return add(Polygon(points,closed=True,fc=fc,ec=ec,lw=l,joinstyle='round'))
    def circle(x,y,r,fc=face,ec=color,l=lw):return add(Circle((x,y),r,fc=fc,ec=ec,lw=l))
    def ellipse(x,y,w,h,fc=face,ec=color,l=lw):return add(Ellipse((x,y),w,h,fc=fc,ec=ec,lw=l))
    def arc(x,y,w,h,a,b,c=color,l=lw):return add(Arc((x,y),w,h,theta1=a,theta2=b,ec=c,lw=l))
    def mix(a,b,amount):
        """Theme-relative shading; no fixed accent colors in the illustrations."""
        aa,bb=to_rgb(a),to_rgb(b)
        return tuple((1-amount)*u+amount*v for u,v in zip(aa,bb))
    def curve(commands,fc='none',ec=color,l=lw):
        vertices=[];codes=[]
        for command in commands:
            op,*values=command
            if op=='Z':
                vertices.append((0,0));codes.append(Path.CLOSEPOLY)
            else:
                code={'M':Path.MOVETO,'L':Path.LINETO,'C':Path.CURVE4,'Q':Path.CURVE3}[op]
                vertices.extend(zip(values[::2],values[1::2]));codes.extend([code]*(len(values)//2))
        return add(PathPatch(Path(vertices,codes),fc=fc,ec=ec,lw=l,capstyle='round',joinstyle='round'))
    shade=mix(accent,color,.23)
    tint=mix(accent,'white',.82)
    skinshade=mix(skin,color,.12)
    softline=mix(color,'white',.35)
    if kind=='phone':
        rect(.22,.03,.56,.94,'white',color,.07)
        rect(.27,.17,.46,.67,face,accent,.025,.5)
        line([(.43,.90),(.57,.90)],color,.8);circle(.5,.10,.025,accent,accent,.4)
        line([(.33,.36),(.41,.44),(.50,.41),(.60,.63),(.67,.57)],accent,.75)
        line([(.33,.25),(.67,.25)],color,.5)
    elif kind=='participant':
        # Original half-body illustration: relaxed shoulders, natural bent arm,
        # a tapered face in three-quarter view, and a visibly held phone.
        # Intentionally no eyes, ties, or fingers that turn into noise in print.
        curve([('M',.30,.64),('C',.24,.68,.24,.83,.29,.91),
               ('C',.35,1.01,.53,.99,.59,.92),('C',.66,.84,.62,.69,.57,.64),
               ('L',.30,.64),('Z',)],hair,hair,.35)
        curve([('M',.35,.66),('L',.35,.57),('C',.34,.54,.30,.54,.27,.52),
               ('C',.32,.44,.54,.44,.60,.54),('L',.51,.59),('L',.51,.69),('Z',)],skin,skin,.3)
        curve([('M',.36,.65),('C',.41,.60,.46,.61,.51,.66),
               ('L',.51,.61),('C',.46,.56,.38,.60,.36,.65),('Z',)],skinshade,skinshade,0)
        curve([('M',.08,.04),('L',.11,.35),('C',.12,.45,.19,.51,.32,.55),
               ('C',.37,.47,.47,.46,.55,.55),('C',.66,.53,.71,.47,.74,.37),
               ('L',.78,.07),('C',.62,.02,.22,.02,.08,.04),('Z',)],accent,shade,.55)
        curve([('M',.55,.54),('C',.64,.50,.68,.42,.68,.33),('L',.74,.08),
               ('C',.68,.06,.60,.05,.55,.05),('C',.58,.22,.59,.40,.55,.54),('Z',)],shade,shade,0)
        curve([('M',.22,.44),('C',.17,.35,.19,.28,.20,.22)],ec=tint,l=.55)
        curve([('M',.31,.55),('C',.37,.44,.49,.44,.56,.55)],ec=tint,l=.75)
        # Face and one ear above collar. The side profile is continuous.
        curve([('M',.31,.82),('C',.30,.89,.35,.93,.43,.93),
               ('C',.51,.94,.57,.90,.58,.83),('L',.59,.76),
               ('C',.59,.68,.54,.62,.48,.62),('C',.40,.62,.34,.69,.33,.76),
               ('C',.29,.75,.28,.79,.31,.82),('Z',)],skin,skinshade,.42)
        curve([('M',.30,.80),('C',.29,.87,.30,.93,.36,.96),
               ('C',.44,1.00,.58,.97,.61,.90),('C',.62,.86,.61,.82,.58,.79),
               ('L',.55,.87),('C',.49,.86,.40,.88,.37,.91),
               ('C',.34,.88,.35,.83,.33,.80),('Z',)],hair,hair,.28)
        curve([('M',.48,.67),('C',.51,.66,.54,.68,.55,.70)],ec=skinshade,l=.4)
        # Bare forearm curves across the body to the hand under the device.
        curve([('M',.18,.30),('C',.17,.22,.22,.14,.29,.13),
               ('C',.40,.11,.57,.14,.69,.20),('L',.71,.28),
               ('C',.58,.24,.45,.20,.33,.20),('C',.30,.20,.27,.26,.26,.32),('Z',)],skin,skinshade,.45)
        curve([('M',.15,.39),('C',.17,.42,.23,.43,.28,.40),
               ('L',.29,.30),('C',.25,.28,.21,.28,.17,.29),('Z',)],accent,shade,.45)
        rect(.68,.22,.19,.33,color,color,.027,.45)
        rect(.70,.255,.15,.257,'white','white',.01,.2)
        line([(.738,.53),(.810,.53)],face,.42)
        # Phone screen carries a small sensing trace, not an invented value.
        line([(.722,.37),(.751,.40),(.777,.35),(.817,.43)],accent,.52)
        curve([('M',.65,.23),('C',.66,.27,.69,.29,.73,.28),
               ('L',.77,.26),('C',.79,.24,.78,.21,.75,.21),
               ('L',.70,.20),('C',.68,.18,.65,.19,.65,.23),('Z',)],skin,skinshade,.4)
    elif kind=='station':
        # Compact outdoor sensing station, with an inlet hood, vented cabinet,
        # service display and stable pedestal, without radio-envelope clutter.
        poly([(.22,.04),(.75,.04),(.67,.10),(.32,.10)],tint,color,.5)
        rect(.46,.10,.085,.24,face,color,.01,.55)
        rect(.26,.32,.50,.45,face,color,.035,.65)
        poly([(.76,.34),(.84,.39),(.84,.76),(.76,.78)],tint,color,.4)
        rect(.30,.60,.34,.13,'white',accent,.016,.45)
        line([(.34,.65),(.39,.68),(.45,.64),(.50,.69),(.58,.68)],accent,.45)
        circle(.70,.665,.022,accent,accent,.2)
        for yy in [.40,.45,.50]:line([(.34,yy),(.68,yy)],softline,.5)
        line([(.69,.56),(.71,.56)],color,.5)
        rect(.47,.77,.09,.07,face,color,.005,.4)
        for yy in [.845,.88,.915]:
            ellipse(.51,yy,.30,.045,face,color,.4)
        curve([('M',.32,.94),('C',.36,.985,.65,.985,.70,.94),('Z',)],accent,color,.45)
        line([(.80,.77),(.80,.95)],color,.55)
    elif kind=='vehicle':
        # Compact hatchback in an accurate side elevation. Wheel wells, glazing,
        # door seams, thin sill and lamps remain legible at the figure's 0.6 in.
        # slot; the shallow roof instrument identifies the sensing vehicle.
        wr=.105*h/w
        curve([('M',.055,.26),('L',.050,.40),('C',.05,.46,.10,.48,.18,.51),
               ('L',.31,.69),('C',.35,.75,.40,.77,.48,.77),
               ('L',.62,.77),('C',.68,.77,.74,.73,.80,.61),
               ('L',.86,.51),('L',.94,.46),('C',.97,.45,.98,.40,.97,.35),
               ('L',.96,.25),('C',.94,.23,.89,.23,.77+wr+.010,.24),
               ('C',.77+wr,.40,.77-wr,.40,.77-wr-.010,.25),('L',.235+wr+.010,.25),
               ('C',.235+wr,.40,.235-wr,.40,.235-wr-.010,.25),('Z',)],accent,color,.65)
        # Body shoulder highlight and the dark lower sill define its volume.
        curve([('M',.09,.43),('C',.30,.48,.63,.47,.91,.44)],ec=tint,l=.55)
        line([(.36,.275),(.64,.275)],shade,1.0)
        # Cabin glazing follows roof curvature, separated by a slender B pillar.
        curve([('M',.23,.53),('L',.35,.68),('C',.38,.72,.41,.73,.46,.73),
               ('L',.49,.73),('L',.49,.53),('Z',)],face,color,.45)
        curve([('M',.53,.73),('L',.63,.73),('C',.68,.73,.73,.67,.79,.54),
               ('L',.53,.53),('Z',)],tint,color,.45)
        line([(.56,.57),(.63,.69)],'white',.50)
        curve([('M',.51,.51),('L',.51,.33),('Q',.53,.30,.61,.31)],ec=shade,l=.45)
        line([(.38,.48),(.42,.48)],shade,.70)
        line([(.65,.48),(.69,.48)],shade,.70)
        # Mirrors and lamps are flush to the continuous body silhouette.
        curve([('M',.76,.55),('L',.81,.55),('Q',.83,.55,.83,.51),('L',.77,.51),('Z',)],shade,color,.35)
        curve([('M',.91,.43),('L',.97,.41),('L',.96,.36),('L',.90,.37),('Z',)],'white',color,.35)
        rect(.058,.365,.038,.095,tint,color,.012,.35)
        line([(.88,.27),(.95,.27)],shade,.65)
        for xx in [.235,.770]:
            ellipse(xx,.25,.204*h/w,.204,color,color,.3)
            ellipse(xx,.25,.130*h/w,.130,face,face,.2)
            ellipse(xx,.25,.054*h/w,.054,softline,softline,.2)
            for dx,dy in [(0,.049),(.047,.015),(.029,-.040),(-.029,-.040),(-.047,.015)]:
                line([(xx+dx*h/w*.40,.25+dy*.40),(xx+dx*h/w,.25+dy)],softline,.35)
        # Low profile roof sensor mounted on a crossbar, not an oversized mast.
        line([(.45,.785),(.64,.785)],color,.6)
        rect(.49,.797,.11,.055,face,color,.009,.45)
        ellipse(.545,.855,.14,.048,accent,color,.45)
        line([(.66,.785),(.68,.87)],color,.5)
    elif kind=='building':
        # Restrained three-quarter municipal facade with aligned window bays.
        poly([(.16,.06),(.69,.06),(.69,.91),(.16,.91)],face,color,.55)
        poly([(.69,.06),(.88,.17),(.88,.92),(.69,.91)],tint,color,.5)
        poly([(.16,.91),(.35,.98),(.88,.98),(.69,.91)],'white',color,.5)
        line([(.10,.04),(.93,.04)],color,.6)
        for xx in [.25,.44]:
            for yy in [.33,.50,.67,.81]:
                rect(xx,yy,.11,.08,'white',accent,0,.4)
                line([(xx+.055,yy),(xx+.055,yy+.08)],tint,.35)
        for yy in [.30,.48,.66,.82]:
            poly([(.75,yy),(.82,yy+.03),(.82,yy+.09),(.75,yy+.06)],'white',accent,.35)
        rect(.31,.07,.22,.16,accent,color,0,.4)
        line([(.42,.075),(.42,.23)],'white',.4)
        line([(.28,.25),(.56,.25)],color,.5)
    elif kind=='air_target':
        # Schematic airborne particles enclosed by a location crosshair.
        circle(.5,.5,.34,face,accent,.8)
        for xx,yy,rr in [(.32,.54,.04),(.45,.68,.025),(.59,.58,.05),(.44,.42,.045),(.64,.36,.025),(.64,.73,.025),(.3,.34,.018)]:circle(xx,yy,rr,accent,accent,.3)
        for pts in [[(.5,.02),(.5,.21)],[(.5,.79),(.5,.98)],[(.02,.5),(.21,.5)],[(.79,.5),(.98,.5)]]:line(pts,color,.65)
    elif kind in ('report_packet','receipt'):
        if kind=='report_packet':
            for dx,dy in [(.12,.10),(.06,.05)]:rect(.10+dx,.06+dy,.64,.76,'white',color,.02,.5)
        poly([(.12,.04),(.79,.04),(.79,.77),(.60,.96),(.12,.96)],'white',color)
        poly([(.60,.96),(.60,.77),(.79,.77)],face,color,.5)
        if kind=='report_packet':
            for i,yy in enumerate([.61,.43,.25]):
                circle(.25,yy,.038,accent,accent,.3)
                for xx in [.37,.51,.65]:rect(xx,yy-.036,.075,.075,face,accent,0,.35)
        else:
            line([(.24,.72),(.48,.72)],accent,.8)
            for yy in [.58,.44]:line([(.24,yy),(.65,yy)],color,.5)
            line([(.25,.25),(.36,.16),(.63,.34)],accent,1.2)
    elif kind=='catalog':
        for dx,dy in [(.13,.11),(.06,.05)]:rect(.08+dx,.07+dy,.70,.75,'white',color,.025,.5)
        rect(.08,.07,.70,.75,face,color,.025)
        rect(.08,.61,.70,.21,accent,accent,.015,.4)
        for xx in [.22,.45,.68]:line([(xx,.77),(xx,.95)],color,.85)
        for yy in [.48,.32,.17]:
            rect(.20,yy,.09,.075,accent,accent,0,.3)
            line([(.36,yy+.04),(.65,yy+.04)],color,.5)
    elif kind=='server':
        # One coherent rack enclosure rather than three floating boxes.
        poly([(.16,.08),(.73,.08),(.73,.86),(.16,.86)],face,color,.6)
        poly([(.73,.08),(.89,.18),(.89,.94),(.73,.86)],tint,color,.5)
        poly([(.16,.86),(.32,.94),(.89,.94),(.73,.86)],'white',color,.5)
        for yy in [.15,.38,.61]:
            rect(.21,yy,.47,.19,'white',color,.015,.45)
            circle(.275,yy+.095,.024,accent,accent,.2)
            circle(.35,yy+.095,.013,tint,accent,.3)
            for xx in [.44,.49,.54,.59]:line([(xx,yy+.063),(xx,yy+.127)],softline,.40)
        for yy in [.32,.39,.46,.53,.60,.67]:
            line([(.785,yy),(.84,yy+.032)],softline,.4)
        rect(.24,.035,.11,.04,color,color,.006,.2)
        rect(.60,.035,.11,.04,color,color,.006,.2)
    elif kind=='requester':
        # Seated analyst in side profile: a separate chair, curved torso and
        # forearm visibly reaching the keyboard, next to an upright monitor.
        rect(.08,.16,.12,.42,tint,color,.035,.45)
        rect(.10,.135,.31,.045,face,color,.018,.45)
        line([(.22,.13),(.22,.045)],color,.6)
        line([(.10,.035),(.33,.035)],color,.65)
        curve([('M',.235,.715),('L',.235,.585),('Q',.28,.55,.32,.555),
               ('L',.345,.675),('L',.325,.715),('Z',)],skin,skinshade,.3)
        curve([('M',.14,.22),('C',.14,.37,.11,.49,.18,.57),
               ('C',.22,.61,.28,.59,.33,.56),('C',.40,.53,.42,.43,.44,.34),
               ('L',.37,.23),('C',.30,.20,.20,.19,.14,.22),('Z',)],accent,shade,.5)
        curve([('M',.19,.57),('C',.20,.50,.20,.35,.16,.25),
               ('C',.22,.22,.27,.24,.30,.24)],ec=tint,l=.55)
        curve([('M',.31,.46),('C',.34,.37,.40,.30,.46,.30),
               ('L',.60,.31),('C',.62,.31,.63,.34,.60,.35),
               ('L',.48,.36),('C',.44,.37,.42,.43,.41,.49),('Z',)],skin,skinshade,.45)
        curve([('M',.28,.53),('C',.31,.57,.37,.56,.40,.51),('L',.43,.44),
               ('C',.40,.41,.36,.41,.33,.43),('Z',)],accent,shade,.4)
        curve([('M',.20,.83),('C',.21,.91,.31,.93,.35,.87),
               ('C',.38,.84,.37,.79,.40,.77),('L',.37,.75),
               ('L',.36,.69),('C',.34,.65,.27,.65,.24,.69),('L',.21,.74),('Z',)],skin,skinshade,.4)
        curve([('M',.21,.74),('C',.17,.78,.18,.87,.21,.90),
               ('C',.27,.96,.36,.92,.38,.86),('C',.32,.87,.29,.85,.27,.82),
               ('L',.25,.76),('L',.24,.71),('Z',)],hair,hair,.3)
        # Equipment: single monitor, pedestal, input board and a clean desk.
        rect(.55,.41,.39,.39,face,color,.022,.6)
        rect(.58,.45,.33,.30,'white',softline,.009,.35)
        line([(.605,.67),(.675,.67)],accent,.8)
        for xx,hh in [(.62,.07),(.68,.13),(.74,.10),(.80,.19)]:
            rect(xx,.485,.035,hh,tint,accent,0,.35)
        line([(.605,.48),(.865,.48)],softline,.35)
        rect(.72,.33,.045,.08,face,color,0,.4)
        poly([(.65,.30),(.84,.30),(.81,.33),(.68,.33)],face,color,.4)
        poly([(.47,.275),(.65,.275),(.61,.30),(.49,.30)],face,color,.4)
        line([(.07,.255),(.96,.255)],color,.85)
        line([(.16,.25),(.16,.04)],softline,.6)
        line([(.89,.25),(.89,.04)],softline,.6)
    elif kind=='coins':
        for xx,base,n in [(.28,.12,3),(.65,.08,5)]:
            for j in range(n):
                yy=base+j*.105
                rect(xx-.19,yy,.38,.11,face,accent,0,.5)
                ellipse(xx,yy+.105,.38,.11,face,accent,.5)
        arc(.64,.62,.18,.09,200,345,accent,.5)
    elif kind=='calibration_pin':
        path=Path([(.50,.03),(.10,.50),(.11,.73),(.12,1.00),(.88,1.00),(.89,.73),(.90,.50),(.50,.03)],
                  [Path.MOVETO,Path.CURVE3,Path.CURVE3,Path.CURVE4,Path.CURVE4,Path.CURVE4,Path.CURVE3,Path.CURVE3])
        add(PathPatch(path,fc=accent,ec=accent,lw=lw))
        circle(.50,.68,.16,'white','white',.4)
        line([(.41,.68),(.59,.68)],color,.5);line([(.50,.59),(.50,.77)],color,.5)
    elif kind=='shield':
        poly([(.50,.96),(.86,.80),(.82,.37),(.68,.17),(.50,.04),(.32,.17),(.18,.37),(.14,.80)],face,accent,.9)
        line([(.32,.51),(.46,.37),(.70,.65)],accent,1.1)
    elif kind=='gauge':
        arc(.5,.30,.84,.84,0,180,color,.8)
        for xx,yy,xe,ye in [(.09,.30,.18,.30),(.20,.60,.26,.54),(.50,.72,.50,.63),(.80,.60,.74,.54),(.91,.30,.82,.30)]:line([(xx,yy),(xe,ye)],color,.6)
        line([(.5,.30),(.72,.56)],accent,1.0);circle(.50,.30,.055,accent,accent,.4)
        line([(.15,.16),(.85,.16)],color,.5)
    elif kind=='sensor':
        # Handheld source instrument: continuous rounded casing, inset display,
        # input controls and a sampling port. Its trace is purely schematic.
        rect(.18,.09,.62,.80,face,color,.075,.65)
        curve([('M',.76,.17),('L',.76,.78),('C',.75,.84,.72,.86,.68,.86)],ec=tint,l=1.1)
        rect(.24,.46,.50,.30,'white',softline,.025,.5)
        line([(.29,.52),(.35,.60),(.43,.56),(.51,.68),(.59,.58),(.67,.63)],accent,.7)
        line([(.29,.70),(.40,.70)],tint,.65)
        circle(.49,.28,.065,accent,accent,.3)
        circle(.31,.28,.028,'white',color,.35)
        circle(.68,.28,.028,'white',color,.35)
        for xx in [.35,.42,.49,.56,.63]:line([(xx,.15),(xx,.18)],softline,.35)
        rect(.43,.89,.12,.045,face,color,.015,.45)
        line([(.49,.935),(.49,.985)],color,.5)
    else:raise ValueError(f'Unknown icon: {kind}')
    return art
