"""Assemble previews from unaltered figure exports; no scientific image edits."""
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
FIGURES = Path(os.environ.get('SOURCEBRIDGE_CONCEPT_OUT', str(ROOT / 'figures')))
NAMES = ('motivation', 'workflow', 'acquisition', 'geometry')


def main():
    writer = PdfWriter()
    for name in NAMES:
        writer.append(PdfReader(FIGURES / (name + '.pdf')))
    writer.add_metadata({'/Title': 'SourceBridge Conceptual Figures',
                         '/Subject': 'Native vector figure masters'})
    with (ROOT / 'SourceBridge_Figures1-4.pdf').open('wb') as stream:
        writer.write(stream)
    dpi, width, margin, gap = 180, 1349, 30, 42
    font_path = Path(os.environ.get('SOURCEBRIDGE_FONT_DIR', 'C:/Windows/Fonts')) / 'timesbd.ttf'
    try:
        font = ImageFont.truetype(str(font_path), 25)
    except OSError:
        font = ImageFont.load_default(size=25)
    panels = []
    for name in NAMES:
        image = Image.open(FIGURES / (name + '.png')).convert('RGB')
        pdf = PdfReader(FIGURES / (name + '.pdf')).pages[0]
        size = (round(float(pdf.mediabox.width) / 72 * dpi),
                round(float(pdf.mediabox.height) / 72 * dpi))
        panels.append(image.resize(size, Image.Resampling.LANCZOS))
    y1 = margin + 34
    y2 = y1 + panels[0].height + gap + 34
    y3 = y2 + panels[1].height + gap + 34
    contact = Image.new('RGB', (width, y3 + max(panels[2].height, panels[3].height) + margin), 'white')
    draw = ImageDraw.Draw(contact)
    for i, (x, y) in enumerate([(margin, y1), (margin, y2), (margin, y3), (width // 2 + 15, y3)]):
        draw.text((x, y - 33), f'Figure {i+1}', font=font, fill='#000000')
        contact.paste(panels[i], (x, y))
    contact.save(ROOT / 'SourceBridge_Figures1-4_overview.png', dpi=(dpi, dpi))
    qa = Path(os.environ.get('SOURCEBRIDGE_CONCEPT_QA_OUT', str(ROOT / 'quality')))
    qa.mkdir(parents=True, exist_ok=True)
    ImageOps.grayscale(contact).save(qa / 'grayscale_preview.png', dpi=(dpi, dpi))


if __name__ == '__main__':
    main()
