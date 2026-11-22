"""Build editable faculty slides; optional measured figures never trigger experiments."""
from pathlib import Path
import argparse
import json

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, help='Completed build-report output directory')
    parser.add_argument('--out', type=Path, default=Path('slides/presentation_draft.pptx'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    content = json.loads((root / 'slides/content.json').read_text(encoding='utf-8'))
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    navy, teal = RGBColor(18, 35, 58), RGBColor(0, 120, 125)
    for i, entry in enumerate(content, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = RGBColor(248, 250, 252)
        title = slide.shapes.add_textbox(Inches(.65), Inches(.4), Inches(12), Inches(.8))
        paragraph = title.text_frame.paragraphs[0]
        paragraph.text = entry['title']
        paragraph.font.size, paragraph.font.bold = Pt(30), True
        paragraph.font.color.rgb = navy
        textbox = slide.shapes.add_textbox(Inches(.8), Inches(1.55), Inches(11.7), Inches(4.8))
        frame = textbox.text_frame
        frame.word_wrap = True
        for j, bullet in enumerate(entry['bullets']):
            p = frame.paragraphs[0] if j == 0 else frame.add_paragraph()
            p.text = bullet
            p.font.size = Pt(19 if i > 18 else 24)
            p.font.color.rgb = navy
            p.space_after = Pt(20 if i <= 18 else 16)
        footer = slide.shapes.add_textbox(Inches(.7), Inches(7.02), Inches(12), Inches(.3))
        p = footer.text_frame.paragraphs[0]
        p.text = f'{i:02d} | SVD + RBF SVM | DRAFT — empirical conclusions require local results'
        p.font.size = Pt(11)
        p.font.color.rgb = teal
        slide.notes_slide.notes_text_frame.text = entry['notes'] + f"\nSuggested time: {entry['minutes']} minutes."
        if i == 10 and args.report:
            figure = args.report / 'accuracy.png'
            if not figure.is_file():
                raise FileNotFoundError(figure)
            frame.clear()
            slide.shapes.add_picture(str(figure), Inches(2), Inches(1.45), width=Inches(9))
            slide.notes_slide.notes_text_frame.text += '\nMeasured figure included; revise conclusions using the saved report.'
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f'Saved {len(content)} slides ({sum(s["minutes"] for s in content):g} scheduled minutes): {args.out}')


if __name__ == '__main__':
    main()
