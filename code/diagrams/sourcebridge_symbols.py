"""Lucide SVG symbols drawn as editable Matplotlib vector paths.

The original SVGs and ISC / MIT license notices are in assets/lucide.
These symbols identify method roles, not measurements or validation claims.
No image is rasterized. Source files are bundled, so regeneration is offline.
"""
from pathlib import Path
from functools import lru_cache
import xml.etree.ElementTree as ET
from fontTools.svgLib.path import parse_path
from fontTools.pens.recordingPen import RecordingPen
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Circle, Ellipse, Rectangle, Polygon, FancyBboxPatch
from matplotlib.transforms import Affine2D

ASSETS = Path(__file__).resolve().parent / 'assets' / 'lucide'
KINDS = tuple(sorted(p.stem for p in ASSETS.glob('*.svg')))


@lru_cache(maxsize=None)
def _elements(kind):
    if kind not in KINDS:
        raise ValueError(f'Unknown symbol {kind!r}; available: {KINDS}')
    root = ET.parse(ASSETS / (kind+'.svg')).getroot()
    elements = []
    for node in root:
        tag=node.tag.rsplit('}',1)[-1]
        if tag=='path':
            pen=RecordingPen();parse_path(node.attrib['d'],pen)
            vertices,codes=[],[]
            for op,args in pen.value:
                if op=='moveTo':
                    vertices.append(args[0]);codes.append(MplPath.MOVETO)
                elif op=='lineTo':
                    vertices.extend(args);codes.extend([MplPath.LINETO]*len(args))
                elif op=='curveTo':
                    vertices.extend(args);codes.extend([MplPath.CURVE4]*len(args))
                elif op=='qCurveTo':
                    vertices.extend(args);codes.extend([MplPath.CURVE3]*len(args))
                elif op=='closePath':
                    vertices.append((0,0));codes.append(MplPath.CLOSEPOLY)
                elif op!='endPath':
                    raise ValueError(op)
            elements.append(('path',MplPath(vertices,codes)))
        else:
            elements.append((tag,dict(node.attrib)))
    return tuple(elements)


def draw_symbol(ax,kind,x,y,w,h=None,*,color='#000000',lw=.85,z=6,
                badge=None,badge_edge=None,pad=.12):
    """Draw a symbol inside x,y,w,h in axis units; optional pale badge.

    SVG coordinates are normalized from 24x24 and retain aspect ratio by
    default. Explicit h can be supplied for deliberate non-square use.
    """
    h=w if h is None else h
    artists=[]
    if badge is not None:
        b=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size='+str(min(w,h)*.15),
                        facecolor=badge,edgecolor=badge_edge or 'none',linewidth=.45,zorder=z-1)
        ax.add_patch(b);artists.append(b)
        x+=w*pad;y+=h*pad;w*=1-2*pad;h*=1-2*pad
    tr=Affine2D().scale(w/24,-h/24).translate(x,y+h)+ax.transData
    for tag,data in _elements(kind):
        if tag=='path':p=PathPatch(data,fill=False)
        elif tag=='circle':p=Circle((float(data['cx']),float(data['cy'])),float(data['r']),fill=False)
        elif tag=='ellipse':p=Ellipse((float(data['cx']),float(data['cy'])),2*float(data['rx']),2*float(data['ry']),fill=False)
        elif tag=='rect':
            p=FancyBboxPatch((float(data['x']),float(data['y'])),float(data['width']),float(data['height']),
                            boxstyle='round,pad=0,rounding_size='+data.get('rx','0'),fill=False)
        elif tag=='line':
            p=PathPatch(MplPath([(float(data['x1']),float(data['y1'])),(float(data['x2']),float(data['y2']))]),fill=False)
        elif tag in ('polyline','polygon'):
            values=[float(v) for v in data['points'].replace(',',' ').split()]
            p=Polygon(list(zip(values[::2],values[1::2])),closed=(tag=='polygon'),fill=False)
        elif tag in ('title','desc'):
            continue
        else:raise ValueError(f'Unsupported SVG element {tag}')
        p.set_transform(tr);p.set_edgecolor(color);p.set_linewidth(lw)
        p.set_capstyle('round');p.set_joinstyle('round');p.set_zorder(z)
        ax.add_patch(p);artists.append(p)
    return artists
