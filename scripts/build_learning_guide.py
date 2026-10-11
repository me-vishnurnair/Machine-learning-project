"""Build the readable PDF from the checked-in Markdown guide (requires reportlab)."""
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted

ROOT=Path(__file__).resolve().parents[1]


def inline(text):
    text=escape(text)
    text=re.sub(r"\*\*(.*?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"`([^`]+)`",r'<font name="Courier">\1</font>',text)
    return text


def build():
    destination=ROOT/"docs/FraudShield_Learning_Guide.pdf"
    font_root=Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("GuideSans",str(font_root/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("GuideSans-Bold",str(font_root/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("GuideSans",normal="GuideSans",bold="GuideSans-Bold",italic="GuideSans",boldItalic="GuideSans-Bold")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="GuideBody",fontName="GuideSans",fontSize=10,leading=15,spaceAfter=8,textColor=colors.HexColor("#26354b")))
    styles.add(ParagraphStyle(name="GuideCell",fontName="GuideSans",fontSize=9,leading=12,spaceAfter=0))
    styles["Title"].fontName="GuideSans-Bold"
    styles["Heading2"].fontName="GuideSans-Bold"
    styles["Title"].textColor=colors.HexColor("#123e46")
    styles["Heading2"].textColor=colors.HexColor("#12665b")
    styles["Heading2"].spaceBefore=16
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=20*mm,bottomMargin=20*mm,
        title="FraudShield AI — Learning and Presentation Guide",author="FraudShield AI project")
    lines=(ROOT/"docs/LEARNING_GUIDE.md").read_text().splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith("```"):
            code=[];i+=1
            while i<len(lines) and not lines[i].strip().startswith("```"):code.append(lines[i]);i+=1
            story.append(Preformatted("\n".join(code),ParagraphStyle(name="Code",fontName="Courier",fontSize=8,leading=11,spaceAfter=12),maxLineLength=90))
        elif line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[value.strip() for value in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+",value) for value in cells):
                    rows.append([Paragraph(inline(value),styles["GuideCell"]) for value in cells])
                i+=1
            table=Table(rows,colWidths=[doc.width*.34,doc.width*.66],repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dcefe9")),("VALIGN",(0,0),(-1,-1),"TOP"),
                ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cfdae0")),("LEFTPADDING",(0,0),(-1,-1),8),
                ("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,10)]);continue
        elif line.startswith("# "):story.append(Paragraph(inline(line[2:]),styles["Title"]))
        elif line.startswith("## "):story.append(Paragraph(inline(line[3:]),styles["Heading2"]))
        else:
            text=line.removeprefix("> ")
            if text.startswith("- "):text="• "+text[2:]
            story.append(Paragraph(inline(text),styles["GuideBody"]))
        i+=1
    def footer(canvas,document):
        canvas.setStrokeColor(colors.HexColor("#c5d6d3"));canvas.line(18*mm,15*mm,A4[0]-18*mm,15*mm)
        canvas.setFont("GuideSans",8);canvas.setFillColor(colors.HexColor("#526879"))
        canvas.drawString(18*mm,11*mm,"FraudShield AI | Synthetic credit-card ML project")
        canvas.drawRightString(A4[0]-18*mm,11*mm,str(document.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    # Preserve the old public download location for the user's existing bookmark.
    (ROOT/"docs/CreditVault_Learning_Guide.pdf").write_bytes(destination.read_bytes())
    print(destination)


if __name__=="__main__":build()
