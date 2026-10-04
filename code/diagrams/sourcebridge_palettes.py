"""Semantic palettes shared by all four native-vector SourceBridge concept figures.

The default ``indigo_vivid`` adds stronger semantic separation and light panel surfaces. Named
alternatives change colors only. Text is independently darkened when necessary
to meet 4.5:1 contrast against every light figure surface; human skin/hair stay
neutral. Standard-library only so this module travels with the plotting source.
"""
import os
DEFAULT_PALETTE = 'indigo_vivid'

CLASSIC = dict(
    ink='#26384A', muted='#667789', line='#C4D2DD', blue='#356FA1',
    teal='#318C7A', red='#C54950', purple='#8872B1', amber='#D29B39',
    pblue='#EFF5FA', pteal='#EEF7F4', pred='#FBEFF0', ppurple='#F3F0F8',
    pamber='#FCF6E9', white='#FFFFFF', pgray='#F5F7F9', skin='#EBCBB4',
    hair='#181A1E', red_panel_border='#E7C7CD', ground='#E5EFEA',
    ground_border='#C4D7D1', interval_grid='#DDE6EE', fee_border='#E3D3B2',
    quote_border='#DEAFB7', blue_phase_border='#BECDD9',
    teal_phase_border='#BDD6CF', selection_border='#DAAAB2',
    nominal_border='#CFC5E0', hull_border='#B5D6CA', projection_border='#DDA7B0')

PALETTES = {
    'indigo_vivid': dict(label='Indigo / teal / amber detailed graphics', label_zh='深靛蓝·青绿·琥珀精致图形',
        accents=dict(ink='#000000', blue='#214C9A', teal='#087768', red='#B86B0A',
                     purple='#79408D', amber='#865214'),
        description='Stronger blue-green-warm contrast, black prose, pale body surfaces and medium emphasis fills.'),
    'indigo_contrast': dict(label='Indigo / amber contrast and structure', label_zh='靛蓝琥珀·对比与层次增强',
        accents=dict(ink='#000000', blue='#304E8A', teal='#226B58', red='#AC7018',
                     purple='#725084', amber='#845B20'),
        description='Black prose, stronger semantic colors, clearer grouping; original illustrative composition retained.'),
    'indigo_refined': dict(label='Indigo / amber detail refinement', label_zh='靛蓝琥珀·原构图细节精修',
        accents=dict(ink='#000000', blue='#52608D', teal='#597267', red='#A4772E',
                     purple='#7B6986', amber='#887045'),
        description='Original semantic families retained with gentler saturation; headings and prose use black explicitly.'),
    'classic': dict(label='Classic (legacy)', label_zh='经典原版（历史配色）',
                    accents={k:CLASSIC[k] for k in ('ink','blue','teal','red','purple','amber')},
                    description='Approved red, blue and green colors, unchanged.'),
    'classic_print': dict(label='Classic / print contrast', label_zh='经典红蓝绿 · 印刷增强',
        accents=dict(ink='#17181A', blue='#356FA1', teal='#287965', red='#B4404B',
                     purple='#786092', amber='#B18128'),
        description='Familiar red, blue and green hierarchy with darker small labels.'),
    'navy_teal': dict(label='Navy / teal', label_zh='海军蓝 · 青绿',
        accents=dict(ink='#111820', blue='#284C70', teal='#496D88', red='#147468',
                     purple='#716080', amber='#A47C2A'),
        description='Navy evidence branches with a teal SourceBridge mechanism.'),
    'indigo_amber': dict(label='Indigo / amber', label_zh='靛蓝 · 琥珀',
        accents=dict(ink='#181720', blue='#4E5895', teal='#576F64', red='#AC7216',
                     purple='#815D83', amber='#816739'),
        description='Indigo public information with a warm amber mechanism; selected paper theme.'),
    'graphite_burgundy': dict(label='Graphite / burgundy', label_zh='石墨 · 酒红',
        accents=dict(ink='#111214', blue='#293540', teal='#3F5B52', red='#8B2944',
                     purple='#60536A', amber='#80683E'),
        description='Near-black graphite structure with burgundy mechanisms.'),
}
VARIANT_ORDER = ('classic_print','navy_teal','indigo_amber','graphite_burgundy')
SEMANTIC_ROLES = dict(blue='Public calibration / positive directional certificate',
    teal='Compatibility hull / negative directional certificate',
    red='SourceBridge selection / purchased guarantee / safe projection',
    purple='Nominal estimator / nominal point',
    amber='Fees and acquisition costs', ink='Main text', muted='Secondary text')

def rgb(value):
    return tuple(int(value[i:i+2],16)/255 for i in (1,3,5))

def blend(foreground,background='#FFFFFF',amount=.10):
    a,b=rgb(foreground),rgb(background)
    return '#'+''.join(f'{round(255*(amount*x+(1-amount)*y)):02X}' for x,y in zip(a,b))

def luminance(value):
    channels=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb(value)]
    return sum(v*w for v,w in zip(channels,(.2126,.7152,.0722)))

def contrast(a,b):
    x,y=sorted((luminance(a),luminance(b)))
    return (y+.05)/(x+.05)

def darken_for_surfaces(value,surfaces,minimum=4.5):
    for darkness in range(101):
        candidate=blend('#000000',value,darkness/100)
        if min(contrast(candidate,bg) for bg in surfaces)>=minimum:
            return candidate
    return '#000000'

def get_palette(name=None):
    name=name or os.environ.get('SOURCEBRIDGE_PALETTE',DEFAULT_PALETTE)
    if name not in PALETTES:
        raise ValueError(f'Unknown SourceBridge palette {name!r}; choose {tuple(PALETTES)}')
    c=CLASSIC.copy()
    if name!='classic':
        c.update(PALETTES[name]['accents'])
        # Matrix headers use white text on these two source colors.
        for role in ('blue','teal'):
            c[role]=darken_for_surfaces(c[role],['#FFFFFF'])
        for role in ('blue','teal','red','purple','amber'):
            c['p'+role]=blend(c[role],amount=.14 if name=='indigo_vivid' else .115 if name=='indigo_contrast' else .075)
        c['muted']=blend(c['ink'],amount=.76)
        c['line']=blend(c['ink'],amount=.22)
        c['pgray']=blend(c['ink'],amount=.035)
        for token,role,amount in (
            ('red_panel_border','red',.25),('ground','teal',.09),
            ('ground_border','teal',.23),('interval_grid','blue',.13),
            ('fee_border','amber',.28),('quote_border','red',.37),
            ('blue_phase_border','blue',.26),('teal_phase_border','teal',.26),
            ('selection_border','red',.40),('nominal_border','purple',.28),
            ('hull_border','teal',.30),('projection_border','red',.43)):
            c[token]=blend(c[role],amount=amount)
        if name in ('indigo_contrast','indigo_vivid'):
            c['line']=blend(c['ink'],amount=.30)
            for token,role,amount in (
                ('red_panel_border','red',.50),('ground','teal',.12),
                ('ground_border','teal',.40),('interval_grid','blue',.20),
                ('fee_border','amber',.48),('quote_border','red',.65),
                ('blue_phase_border','blue',.48),('teal_phase_border','teal',.48),
                ('selection_border','red',.64),('nominal_border','purple',.46),
                ('hull_border','teal',.52),('projection_border','red',.68)):
                c[token]=blend(c[role],amount=amount)
        # Body fills stay lighter than headers; stronger fills are for graphics
        # and report cells. Never use saturated backgrounds behind black prose.
    for role in ('blue','teal','red','purple','amber'):
        c['surface_'+role]=blend(c[role],amount=.045)
        c['mid_'+role]=blend(c[role],amount=.24)
    c['panel_base']='#FAFBFD'
    c['gray']=c['muted']
    for role in ('blue','teal','red','purple','amber'):
        c['pale_'+role]=c['p'+role]
    surfaces=[c[k] for k in ('white','pblue','pteal','pred','ppurple','pamber','pgray','ground',
                            'mid_blue','mid_teal','mid_red','mid_purple','mid_amber')]
    for role in ('ink','muted','gray','blue','teal','red','purple','amber'):
        c['text_'+role]=c[role] if name=='classic' else darken_for_surfaces(c[role],surfaces)
    return name,c

def text_color(colors,role):
    return colors.get('text_'+role,colors.get(role,role))

def palette_record(name):
    _,colors=get_palette(name)
    surface_names=('white','pblue','pteal','pred','ppurple','pamber','pgray','ground',
                   'mid_blue','mid_teal','mid_red','mid_purple','mid_amber')
    roles=('ink','muted','gray','blue','teal','red','purple','amber')
    ratios={role:round(min(contrast(colors['text_'+role],colors[s]) for s in surface_names),3)
            for role in roles}
    headers={role:round(contrast('#FFFFFF',colors[role]),3) for role in ('blue','teal')}
    # Acquired-object columns use the contrast-adjusted mechanism color.
    headers['acquired_objects']=round(contrast('#FFFFFF',colors['text_red']),3)
    return dict(id=name,**PALETTES[name],colors=colors,semantic_roles=SEMANTIC_ROLES,
                minimum_text_contrast_by_role=ratios,white_matrix_header_contrast=headers,
                text_contrast_4_5_pass=min(*ratios.values(),*headers.values())>=4.5,
                skin_and_hair_unchanged=colors['skin']==CLASSIC['skin'] and colors['hair']==CLASSIC['hair'])
