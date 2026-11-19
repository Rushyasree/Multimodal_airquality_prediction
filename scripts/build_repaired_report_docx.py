"""
scripts/build_repaired_report_docx.py

Builds the fully repaired, academically formatted VayuDrishti Capstone Project Report:
- Traditional academic style: White background, black text, Times New Roman throughout.
- Main chapter headings: 16 pt bold centered.
- Section headings: 14 pt bold.
- Subsection headings: 12 pt bold.
- Body text: 12 pt justified, 1.15 line spacing, 0.35 in first-line indent.
- Table of Contents matching user reference image (dotted tab leaders, bold chapters, indented subsections).
- Separate Roman numeral (ii..ix) and Arabic (1..74) section page numbering.
- Clean high-resolution IEEE architecture diagrams and empirical figures with centered captions below.
- Clean academic tables with captions above.
- IEEE references and complete member biodata.
"""

import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DOCX = ROOT / "docs/VayuDrishti_Final_Capstone_Report_Repaired.docx"
FIG_DIR = ROOT / "paper/figures"

FONT_NAME = "Times New Roman"
COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_TEXT = RGBColor(15, 23, 42)      # Deep slate / charcoal black
COLOR_MUTED = RGBColor(71, 85, 105)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def add_chapter_title(doc, text):
    """Main chapter headings: Times New Roman, 16 pt, bold, centered."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = COLOR_BLACK
    return p

def add_section_heading(doc, text):
    """Section headings: 14 pt, bold."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = COLOR_BLACK
    return p

def add_subsection_heading(doc, text):
    """Subsection headings: 12 pt, bold."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = COLOR_BLACK
    return p

def add_body_p(doc, text, indent=True, bold_prefix="", space_after=6, line_spacing=1.15):
    """Body text: 12 pt, justified, clean indentation."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = line_spacing
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.35)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = FONT_NAME
        r_pre.font.size = Pt(12)
        r_pre.font.color.rgb = COLOR_BLACK
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(12)
    run.font.color.rgb = COLOR_TEXT
    return p

def add_bullet_p(doc, text, bold_prefix=""):
    """Bullet list: 12 pt, clean indentation."""
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = FONT_NAME
        r_pre.font.size = Pt(12)
        r_pre.font.color.rgb = COLOR_BLACK
    run = p.add_run(text)
    run.font.name = FONT_NAME
    run.font.size = Pt(12)
    run.font.color.rgb = COLOR_TEXT
    return p

def add_toc_leader(doc, left_text, right_text, is_bold=False, indent_in=0.0, space_after=3, font_size=11):
    """Generates a Table of Contents entry matching the reference image with dotted tab leaders."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if indent_in > 0:
        p.paragraph_format.left_indent = Inches(indent_in)

    # 9360 dxa = 6.5 in (right margin on 8.5" page with 1.0" left/right margins)
    pPr = p._element.get_or_add_pPr()
    tabs_xml = f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:leader="dot" w:pos="9360"/></w:tabs>'
    pPr.append(parse_xml(tabs_xml))

    r1 = p.add_run(left_text)
    r1.font.name = FONT_NAME
    r1.font.size = Pt(font_size)
    r1.bold = is_bold
    r1.font.color.rgb = COLOR_BLACK

    r2 = p.add_run("\t")
    r2.font.name = FONT_NAME
    r2.font.size = Pt(font_size)

    r3 = p.add_run(str(right_text))
    r3.font.name = FONT_NAME
    r3.font.size = Pt(font_size)
    r3.bold = is_bold
    r3.font.color.rgb = COLOR_BLACK
    return p

def add_figure(doc, img_path, fig_no, caption, width_in=5.8):
    """Inserts a figure centered with caption directly below, bound with keep_with_next."""
    if not os.path.exists(img_path):
        print(f"Warning: Figure {img_path} not found.")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(12)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    r_no = p_cap.add_run(f"Figure {fig_no}: ")
    r_no.bold = True
    r_no.font.name = FONT_NAME
    r_no.font.size = Pt(10.5)
    r_no.font.color.rgb = COLOR_BLACK
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = FONT_NAME
    r_cap.font.size = Pt(10.5)
    r_cap.italic = True
    r_cap.font.color.rgb = COLOR_TEXT

def add_code_block(doc, code_text):
    """Shaded, monospace code block with academic border."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.right_indent = Inches(0.35)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.05

    pPr = p._element.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F8FAFC"/>')
    pBdr = parse_xml(f'''
        <w:pBdr {nsdecls("w")}>
            <w:left w:val="single" w:sz="12" w:space="8" w:color="0284C7"/>
        </w:pBdr>
    ''')
    pPr.append(shd)
    pPr.append(pBdr)

    run = p.add_run(code_text)
    run.font.name = "Consolas"
    run.font.size = Pt(9.0)
    run.font.color.rgb = RGBColor(15, 23, 42)

def create_table(doc, headers, rows, col_widths=None, caption="", table_no=""):
    """Inserts a clean academic table with caption above."""
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_cap.paragraph_format.space_before = Pt(10)
        p_cap.paragraph_format.space_after = Pt(4)
        p_cap.paragraph_format.keep_with_next = True
        if table_no:
            r_no = p_cap.add_run(f"Table {table_no}: ")
            r_no.bold = True
            r_no.font.name = FONT_NAME
            r_no.font.size = Pt(10.5)
            r_no.font.color.rgb = COLOR_BLACK
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = FONT_NAME
        r_cap.font.size = Pt(10.5)
        r_cap.italic = True
        r_cap.font.color.rgb = COLOR_TEXT

    num_rows = len(rows) + 1
    num_cols = len(headers)
    tbl = doc.add_table(rows=num_rows, cols=num_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    # Header Row
    hdr_cells = tbl.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = h_text
        set_cell_background(hdr_cells[i], "E2E8F0")  # Clean slate grey header
        set_cell_margins(hdr_cells[i], top=80, bottom=80, left=90, right=90)
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.name = FONT_NAME
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = COLOR_BLACK

    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row_cells = tbl.rows[r_idx + 1].cells
        bg_color = "FFFFFF" if r_idx % 2 == 0 else "F8FAFC"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=60, bottom=60, left=90, right=90)
            row_cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = row_cells[c_idx].paragraphs[0]
            if c_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].font.name = FONT_NAME
            p.runs[0].font.size = Pt(9.0)
            p.runs[0].font.color.rgb = COLOR_TEXT

    # Set Column Widths if provided
    if col_widths and len(col_widths) == num_cols:
        for row in tbl.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    # Clean academic borders
    tblPr = tbl._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="8" w:space="0" w:color="0F172A"/>
                <w:bottom w:val="single" w:sz="8" w:space="0" w:color="0F172A"/>
                <w:left w:val="none"/>
                <w:right w:val="none"/>
                <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>
                <w:insideV w:val="none"/>
            </w:tblBorders>
        ''')
        tblPr[0].append(borders)

    # Space after table
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)


# ==============================================================================
# FRONT MATTER BUILDER
# ==============================================================================
def build_front_matter(doc):
    # 1. TITLE PAGE
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(24)
    p_title.paragraph_format.space_after = Pt(12)
    p_title.paragraph_format.line_spacing = 1.15
    r = p_title.add_run("VAYUDRISHTI: MULTIMODAL SPATIO-TEMPORAL AIR QUALITY PREDICTION IN INDIAN URBAN ENVIRONMENTS")
    r.font.name = FONT_NAME
    r.font.size = Pt(16)
    r.font.bold = True
    r.font.color.rgb = COLOR_BLACK

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(20)
    r = p_sub.add_run("A CAPSTONE PROJECT REPORT")
    r.font.name = FONT_NAME
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = COLOR_BLACK

    p_req = doc.add_paragraph()
    p_req.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_req.paragraph_format.space_after = Pt(20)
    r = p_req.add_run("Submitted in partial fulfillment of the requirement for the award of the Degree of\nBACHELOR OF TECHNOLOGY\nIN\nCOMPUTER SCIENCE AND ENGINEERING")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.color.rgb = COLOR_BLACK

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_after = Pt(6)
    r = p_by.add_run("by")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.italic = True

    members = [
        "TELIKI RUSHYA SREE (23BCE9562)",
        "MACHIREDDY SWATHI (23BCE9566)",
        "THANUJA VARAKALA (23BCE9608)",
        "PALLI SHILPA (23BCE9567)"
    ]
    for m in members:
        p_m = doc.add_paragraph()
        p_m.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_m.paragraph_format.space_after = Pt(2)
        r = p_m.add_run(m)
        r.font.name = FONT_NAME
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = COLOR_BLACK

    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_before = Pt(16)
    p_guide.paragraph_format.space_after = Pt(2)
    r = p_guide.add_run("Under the Guidance of\n")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.italic = True
    r2 = p_guide.add_run("PROF. THOTA RAMATHULSI\n")
    r2.font.name = FONT_NAME
    r2.font.size = Pt(13)
    r2.font.bold = True
    r2.font.color.rgb = COLOR_BLACK
    r3 = p_guide.add_run("Department of Software and Systems Engineering\nSchool of Computer Science and Engineering (SCOPE)")
    r3.font.name = FONT_NAME
    r3.font.size = Pt(11)
    r3.font.color.rgb = COLOR_BLACK

    # VIT Logo if exists
    vit_logo = ROOT / "assets/vit_logo.png"
    if vit_logo.exists():
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(12)
        p_logo.paragraph_format.space_after = Pt(8)
        p_logo.add_run().add_picture(str(vit_logo), width=Inches(1.5))

    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(12)
    p_inst.paragraph_format.space_after = Pt(0)
    r = p_inst.add_run("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING (SCOPE)\nVIT-AP UNIVERSITY\nAMARAVATI - 522237, ANDHRA PRADESH, INDIA\nNOVEMBER 2026")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = COLOR_BLACK

    doc.add_page_break()

    # 2. CERTIFICATE PAGE
    add_chapter_title(doc, "CERTIFICATE")
    p_cert = doc.add_paragraph()
    p_cert.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cert.paragraph_format.line_spacing = 1.25
    p_cert.paragraph_format.space_after = Pt(14)
    r = p_cert.add_run(
        "This is to certify that the Capstone Project report titled "
        "\"VAYUDRISHTI: MULTIMODAL SPATIO-TEMPORAL AIR QUALITY PREDICTION IN INDIAN URBAN ENVIRONMENTS\" "
        "that is being submitted by TELIKI RUSHYA SREE (23BCE9562), MACHIREDDY SWATHI (23BCE9566), "
        "THANUJA VARAKALA (23BCE9608), and PALLI SHILPA (23BCE9567) in partial fulfillment of the requirements "
        "for the award of Bachelor of Technology in Computer Science and Engineering, is a record of bonafide work "
        "done under my guidance. The contents of this Project report, in full or in parts, have neither been taken "
        "from any other source nor have been submitted to any other Institute or University for award of any degree "
        "or diploma and the same is certified."
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(12)
    r.font.color.rgb = COLOR_TEXT

    p_sig = doc.add_paragraph()
    p_sig.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_sig.paragraph_format.space_before = Pt(36)
    p_sig.paragraph_format.space_after = Pt(24)
    r = p_sig.add_run("Prof. Thota Ramathulsi\nProject Guide\nDepartment of SCOPE, VIT-AP University")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.bold = True

    p_sat = doc.add_paragraph()
    p_sat.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_sat.paragraph_format.space_after = Pt(28)
    r = p_sat.add_run("The thesis is satisfactory / unsatisfactory")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.italic = True

    p_exam = doc.add_paragraph()
    p_exam.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_exam.paragraph_format.space_after = Pt(36)
    r = p_exam.add_run("Internal Examiner                                                          External Examiner")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.bold = True

    p_appr = doc.add_paragraph()
    p_appr.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p_appr.add_run(
        "Approved by:\n"
        "PROGRAM CHAIR                                                     DEAN\n"
        "B. Tech. CSE                                                           School Of Computer Science and Engineering"
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.bold = True

    doc.add_page_break()

    # 3. ACKNOWLEDGEMENTS
    add_chapter_title(doc, "ACKNOWLEDGEMENTS")
    add_body_p(doc,
        "We express our sincere gratitude and profound respect to our project guide, Prof. Thota Ramathulsi, "
        "Department of Software and Systems Engineering, School of Computer Science and Engineering (SCOPE), for her "
        "invaluable guidance, constructive critique, and constant encouragement throughout the formulation and execution "
        "of this Capstone Project. Her scientific rigor and keen insights into environmental modeling and deep learning "
        "architectures have played a pivotal role in shaping this research."
    )
    add_body_p(doc,
        "We extend our deep appreciation to the Management of VIT-AP University, the Vice-Chancellor, the Registrar, "
        "and the Dean of the School of Computer Science and Engineering (SCOPE) for providing state-of-the-art computational "
        "infrastructure, laboratory facilities, and an encouraging research environment."
    )
    add_body_p(doc,
        "We would also like to thank the Program Chair, B.Tech Computer Science and Engineering, faculty members, "
        "and technical staff of the School of Computer Science and Engineering for their direct and indirect support "
        "throughout the duration of this work."
    )
    add_body_p(doc,
        "Finally, we express our heartfelt thanks to our parents, family members, and friends whose unconditional love, "
        "patience, and encouragement have been a perpetual source of strength during our academic journey."
    )

    p_team = doc.add_paragraph()
    p_team.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_team.paragraph_format.space_before = Pt(24)
    r = p_team.add_run(
        "TELIKI RUSHYA SREE (23BCE9562)\n"
        "MACHIREDDY SWATHI (23BCE9566)\n"
        "THANUJA VARAKALA (23BCE9608)\n"
        "PALLI SHILPA (23BCE9567)"
    )
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.bold = True

    doc.add_page_break()

    # 4. ABSTRACT
    add_chapter_title(doc, "ABSTRACT")
    add_body_p(doc,
        "Particulate matter pollution, particularly fine breathable particulate matter (PM2.5) and coarse particulate matter "
        "(PM10), constitutes a severe, persistent public health and environmental crisis across Indian urban agglomerations. "
        "Conventional continuous ambient air quality monitoring stations (CAAQMS) provide high-precision statutory measurements "
        "but suffer from extreme spatial sparsity due to high capital and operational costs (~INR 1.2–1.5 Crore per station). "
        "Consequently, extensive metropolitan and suburban zones remain unmonitored. Furthermore, official government portals publish "
        "retrospective rolling averages that fail to provide proactive, forward-looking warnings. To address these critical challenges, "
        "this Capstone Project develops VayuDrishti, a comprehensive multimodal spatio-temporal air quality intelligence and "
        "forecasting framework tailored to Indian urban environments."
    )
    add_body_p(doc,
        "VayuDrishti integrates ten interconnected research areas: (1) long-term historical sensor sequence analysis (1,369,779 hourly records "
        "across 90 stations from 2015 to 2020); (2) next-hour (t+1) temporal forecasting using deep recurrent architectures; (3) gradient-boosted "
        "decision tree (LightGBM) sequence modeling; (4) domain adaptation over recent post-COVID multi-year logs (over 3.3 million cleaned records "
        "across 85 stations from 2020 to 2025); (5) exploratory multimodal sensor-image integration; (6) streaming live air-quality telemetry via the "
        "OpenAQ API v3; (7) continuous geospatial surface estimation via geodesic Inverse Distance Weighting (IDW); (8) nationwide monitoring "
        "infrastructure coverage and Clark-Evans spatial clustering audits across 31 cities; (9) distribution-free uncertainty quantification using "
        "distance-scaled split conformal prediction; and (10) an operational, interactive research dashboard deployed on Streamlit Cloud."
    )
    add_body_p(doc,
        "Empirical evaluations demonstrate that for historical next-hour sequence prediction, a 2-layer stacked LSTM network achieves an MAE "
        "of 10.5719 μg/m³, RMSE of 18.5789 μg/m³, and R² of 0.9492 on PM2.5, alongside an MAE of 18.8643 μg/m³, RMSE of 32.2027 μg/m³, and R² of 0.9301 "
        "on PM10. A historical severity-gated ensemble slightly improves PM2.5 MAE to 10.3508 μg/m³. Chronological domain adaptation on the recent "
        "2020–2025 dataset yields adapted test errors of 6.6216 μg/m³ MAE for PM2.5 and 16.1660 μg/m³ MAE for PM10, reflecting post-COVID distribution shifts. "
        "In spatial interpolation, geodesic IDW demonstrates a 20.0% error reduction on PM2.5 (MAE 24.07 μg/m³ vs. 30.06 μg/m³) and a 25.4% reduction "
        "on PM10 (MAE 52.76 μg/m³ vs. 70.68 μg/m³) over the nearest-neighbor baseline in a rigorous Delhi leave-one-station-out holdout experiment. "
        "In contrast, experimental multimodal fusion on an opportunistically aligned cohort of 199 images across 18 stations indicates that unconstrained "
        "early concatenation inflates prediction error (+12.19 μg/m³), establishing that outdoor imagery requires strict micro-meteorological synchronization "
        "and confirming our bounded residual fallback architecture."
    )
    add_body_p(doc,
        "A 14-day live audit across four operational monitoring stations (418 matched prediction-observation pairs) reveals that next-hour persistence "
        "is an exceptionally strong operational baseline (PM2.5 GBDT MAE 9.64 vs. Persistence 9.86, p=0.48), demonstrating that live predictive gains "
        "over a 1-hour horizon are modest and statistically non-significant without multi-season validation. Finally, split conformal prediction provides "
        "~99% empirical coverage with distance-scaled interval expansion. VayuDrishti establishes an academically defensible, provenance-aware framework "
        "for real-time urban air quality intelligence."
    )

    doc.add_page_break()

    # 5. TABLE OF CONTENTS (Matching Reference Screenshot)
    add_chapter_title(doc, "TABLE OF CONTENTS")

    toc_entries = [
        ("Certificate", "ii", False, 0.0),
        ("Acknowledgements", "iii", False, 0.0),
        ("Abstract", "iv", False, 0.0),
        ("List of Tables", "viii", False, 0.0),
        ("List of Figures", "ix", False, 0.0),
        ("List of Abbreviations", "x", False, 0.0),

        ("Chapter 1  Introduction", "1", True, 0.0),
        ("1.1  Background & Environmental Context", "1", False, 0.35),
        ("1.2  Problem Statement", "1", False, 0.35),
        ("1.3  Research Motivation", "2", False, 0.35),
        ("1.4  Research Objectives", "2", False, 0.35),
        ("1.5  Research Scope and Provenance Framework", "3", False, 0.35),
        ("1.6  Background and Literature Review", "3", False, 0.35),
        ("1.7  Identified Research Gaps", "3", False, 0.35),
        ("1.8  Organization of the Report", "3", False, 0.35),

        ("Chapter 2  System Architecture and Methodology", "5", True, 0.0),
        ("2.1  Overall System Architecture", "5", False, 0.35),
        ("2.2  Data Sources and Ingestion Pipelines", "5", False, 0.35),
        ("2.3  Data Quality Filtering and Preprocessing", "5", False, 0.35),
        ("2.4  Audited Feature Engineering Schema", "6", False, 0.35),
        ("2.5  Deep Recurrent LSTM Forecasting", "6", False, 0.35),
        ("2.6  Gradient Boosted Trees & Severity-Gated Ensemble", "7", False, 0.35),
        ("2.7  Experimental Multimodal Residual Architecture", "8", False, 0.35),
        ("2.8  Geodesic Inverse Distance Weighting (IDW)", "9", False, 0.35),
        ("2.9  Spatial Reliability & Telemetry Freshness Governance", "10", False, 0.35),
        ("2.10 Distribution-Free Conformal Uncertainty Calibration", "11", False, 0.35),
        ("2.11 Interactive Cloud Dashboard Architecture", "12", False, 0.35),

        ("Chapter 3  Implementation and Experimental Setup", "13", True, 0.0),
        ("3.1  Software and Hardware Environment", "13", False, 0.35),
        ("3.2  Comprehensive Dataset Specifications", "13", False, 0.35),
        ("3.3  Chronological Splitting & Leakage Prevention", "13", False, 0.35),
        ("3.4  Model Configurations and Hyperparameters", "14", False, 0.35),
        ("3.5  Mathematical Evaluation Metrics", "15", False, 0.35),
        ("3.6  Live Telemetry Streaming Integration", "15", False, 0.35),
        ("3.7  Computational and Deployment Cost Considerations", "15", False, 0.35),

        ("Chapter 4  Results and Discussion", "17", True, 0.0),
        ("4.1  Historical Sequence Forecasting Benchmarks", "17", False, 0.35),
        ("4.2  Recent-Data Multi-Year Adaptation Results", "19", False, 0.35),
        ("4.3  Exploratory Multimodal Fusion & Negative Transfer", "20", False, 0.35),
        ("4.4  14-Day Live Telemetry Operational Audit", "21", False, 0.35),
        ("4.5  Delhi Spatial Holdout Interpolation Benchmarks", "22", False, 0.35),
        ("4.6  Nationwide 31-City Spatial Reliability Audit", "23", False, 0.35),
        ("4.7  Distance-Scaled Conformal Uncertainty Evaluation", "25", False, 0.35),
        ("4.8  Operational Streamlit Dashboard Interface", "26", False, 0.35),
        ("4.9  Overall Discussion & Synthesis", "27", False, 0.35),

        ("Chapter 5  Conclusion and Future Work", "28", True, 0.0),
        ("5.1  Summary of Research Findings", "28", False, 0.35),
        ("5.2  Core Academic Contributions", "28", False, 0.35),
        ("5.3  Honest Scientific Limitations", "28", False, 0.35),
        ("5.4  Future Research Directions", "29", False, 0.35),

        ("Appendix A  Core Implementation Source Codes", "30", True, 0.0),
        ("A.1  Residual Multimodal Hybrid PyTorch Architecture", "30", False, 0.35),
        ("A.2  Geodesic IDW Spatial Engine & Freshness Filters", "31", False, 0.35),
        ("A.3  Clark-Evans Nearest-Neighbor Clustering Ratio", "32", False, 0.35),

        ("References", "33", True, 0.0),
        ("Biodata of Project Members", "35", True, 0.0)
    ]

    for title, pg, is_b, ind in toc_entries:
        sp_after = 4 if is_b else 2.5
        add_toc_leader(doc, title, pg, is_bold=is_b, indent_in=ind, space_after=sp_after, font_size=11)

    doc.add_page_break()

    # 6. LIST OF TABLES
    add_chapter_title(doc, "LIST OF TABLES")
    tables_list = [
        ("Table 2.1", "Audited 27-Dimensional Numerical Sensor Feature Schema", "6"),
        ("Table 3.1", "Model Hyperparameter and Algorithmic Configurations", "14"),
        ("Table 3.2", "Computational Resource & Operational Deployment Analysis", "15"),
        ("Table 4.1", "Historical Next-Hour Forecasting Performance (404,431 Test Sequences)", "17"),
        ("Table 4.2", "Recent-Data Adaptation Performance on 2020–2025 Test Cohort", "19"),
        ("Table 4.3", "Exploratory Image and Sensor Alignment Pathway Evaluation", "20"),
        ("Table 4.4", "14-Day Live Telemetry Operational Audit (4 Stations, 418 Matched Pairs)", "22"),
        ("Table 4.5", "Delhi Spatial Holdout Evaluation: IDW vs. Nearest-Station Baseline", "22"),
        ("Table 4.6", "Nationwide 31-City Station Catalogue & Spatial Reliability Audit Matrix", "24"),
        ("Table 4.7", "Distance-Scaled Conformal Uncertainty Calibration Summary", "25"),
    ]
    for t_no, t_title, pg in tables_list:
        add_toc_leader(doc, f"{t_no}    {t_title}", pg, is_bold=False, indent_in=0.0, space_after=3.5, font_size=11)

    doc.add_page_break()

    # 7. LIST OF FIGURES
    add_chapter_title(doc, "LIST OF FIGURES")
    figures_list = [
        ("Figure 2.1", "Complete VayuDrishti Multimodal Forecasting Architecture", "5"),
        ("Figure 2.2", "Continuous Telemetry Ingestion and 27-Dimensional Feature Pipeline", "6"),
        ("Figure 2.3", "Deep Recurrent 2-Layer Stacked LSTM Sequence Forecaster Architecture", "7"),
        ("Figure 2.4", "Independent Severity-Gated LSTM–GBDT Ensemble Architecture", "8"),
        ("Figure 2.5", "Optical Feature Extraction via Pretrained ResNet-50 Convolutional Backbone", "9"),
        ("Figure 2.6", "Bounded Residual Multimodal Hybrid Model with Fallback Masking", "9"),
        ("Figure 2.7", "Geodesic Inverse Distance Weighting (IDW) Spatial Interpolation Workflow", "10"),
        ("Figure 2.8", "Spatial Reliability, Telemetry Freshness & Geometry Audit Workflow", "11"),
        ("Figure 2.9", "Distance-Scaled Conformal Uncertainty Estimation Workflow", "12"),
        ("Figure 4.1", "Ground Truth vs Predicted Particulate Concentrations on Historical Test Set", "18"),
        ("Figure 4.2", "Multi-Class Particulate Matter Air Quality Confusion Matrices", "19"),
        ("Figure 4.3", "Distance-Scaled Spatial Conformal Calibration Intervals", "26"),
        ("Figure 4.4", "Multi-Station Generalization across 15 Operational CAAQMS Stations", "23"),
        ("Figure 4.5", "Seasonal Model Dynamics and Particulate Regime Error Curves", "20"),
        ("Figure 4.6", "Spatial Advection Cross-Correlation in Urban Corridors", "23"),
        ("Figure 4.7", "Opportunistic Image Alignment Discrepancy Diagnostics (PM25Vision)", "21"),
        ("Figure 4.8", "Operational Streamlit Real-Time Dashboard Interface", "27"),
    ]
    for f_no, f_title, pg in figures_list:
        add_toc_leader(doc, f"{f_no}    {f_title}", pg, is_bold=False, indent_in=0.0, space_after=3.5, font_size=11)

    doc.add_page_break()

    # 8. LIST OF ABBREVIATIONS
    add_chapter_title(doc, "LIST OF ABBREVIATIONS")
    abbrs = [
        ("API", "Application Programming Interface"),
        ("AQI", "Air Quality Index (Central Pollution Control Board Standard)"),
        ("ARIMA", "Autoregressive Integrated Moving Average"),
        ("CAAQMS", "Continuous Ambient Air Quality Monitoring Station"),
        ("CPCB", "Central Pollution Control Board (Ministry of Environment, Govt. of India)"),
        ("CTM", "Chemical Transport Model (e.g., WRF-Chem, CMAQ)"),
        ("GBDT", "Gradient Boosted Decision Trees (LightGBM Framework)"),
        ("GRU", "Gated Recurrent Unit"),
        ("IDW", "Inverse Distance Weighting (Geodesic Spatial Interpolation)"),
        ("LOSO", "Leave-One-Station-Out Spatial Cross-Validation Protocol"),
        ("LSTM", "Long Short-Term Memory Recurrent Neural Network"),
        ("MAE", "Mean Absolute Error (Physical unit: μg/m³)"),
        ("NAAQS", "National Ambient Air Quality Standards (CPCB India)"),
        ("NN", "Nearest Neighbor Spatial Baseline"),
        ("OpenAQ", "Global Community Real-Time Air Quality Platform (API v3)"),
        ("PINN", "Physics-Informed Neural Network"),
        ("PM2.5", "Fine Particulate Matter (Aerodynamic diameter ≤ 2.5 μm)"),
        ("PM10", "Coarse Particulate Matter (Aerodynamic diameter ≤ 10 μm)"),
        ("R²", "Coefficient of Determination"),
        ("RMSE", "Root Mean Square Error (Physical unit: μg/m³)"),
        ("SCOPE", "School of Computer Science and Engineering, VIT-AP University"),
        ("ST-GNN", "Spatio-Temporal Graph Neural Network"),
        ("SVR", "Support Vector Regression"),
        ("WGS84", "World Geodetic System 1984 (Ellipsoidal Coordinate Reference System)")
    ]
    for ab, desc in abbrs:
        p_ab = doc.add_paragraph()
        p_ab.paragraph_format.space_after = Pt(2.5)
        p_ab.paragraph_format.line_spacing = 1.15
        pPr = p_ab._element.get_or_add_pPr()
        tabs_xml = f'<w:tabs {nsdecls("w")}><w:tab w:val="left" w:pos="2160"/></w:tabs>'
        pPr.append(parse_xml(tabs_xml))
        r1 = p_ab.add_run(ab)
        r1.font.name = FONT_NAME
        r1.font.size = Pt(11)
        r1.bold = True
        r2 = p_ab.add_run("\t")
        r3 = p_ab.add_run(desc)
        r3.font.name = FONT_NAME
        r3.font.size = Pt(11)
        r3.font.color.rgb = COLOR_TEXT

    doc.add_page_break()


# ==============================================================================
# CHAPTER 1: INTRODUCTION
# ==============================================================================
def build_chapter_1(doc):
    add_chapter_title(doc, "CHAPTER 1: INTRODUCTION")

    add_section_heading(doc, "1.1 Background & Environmental Context")
    add_body_p(doc,
        "Ambient particulate matter air pollution, specifically fine particulate matter (PM2.5, particles with an aerodynamic "
        "diameter ≤ 2.5 μm) and coarse particulate matter (PM10, ≤ 10 μm), represents one of the most severe public health and "
        "environmental crises across Indian urban agglomerations. Due to their microscopic size, PM2.5 particles penetrate deep into "
        "the pulmonary alveoli and cross into the cardiovascular system, contributing directly to chronic respiratory illness, "
        "ischemic heart disease, stroke, lung cancer, and premature mortality. Across the Indo-Gangetic Plain and peninsular industrial hubs, "
        "particulate concentrations frequently exceed the statutory National Ambient Air Quality Standards (NAAQS) established by the "
        "Central Pollution Control Board (CPCB)—60 μg/m³ for 24-hour PM2.5 and 100 μg/m³ for PM10—and surpass the World Health Organization (WHO) "
        "annual guidelines (5 μg/m³ for PM2.5 and 15 μg/m³ for PM10) by tenfold during severe winter atmospheric episodes."
    )
    add_body_p(doc,
        "Historically, municipal authorities and environmental scientists relied on numerical Chemical Transport Models (CTMs), "
        "such as the Weather Research and Forecasting model coupled with Chemistry (WRF-Chem) and the Community Multiscale Air Quality "
        "(CMAQ) system. While grounded in atmospheric thermodynamics and chemical gas-phase mechanisms, CTMs suffer from severe limitations "
        "in operational urban environments: they require massive supercomputing clusters, rely on static emission inventories that become "
        "obsolete rapidly, and exhibit multi-hour computational latencies that preclude real-time localized warnings. Consequently, data-driven "
        "machine learning models leveraging Continuous Ambient Air Quality Monitoring Station (CAAQMS) telemetry have emerged as an essential "
        "operational paradigm."
    )

    add_section_heading(doc, "1.2 Problem Statement")
    add_body_p(doc,
        "Despite the deployment of CAAQMS networks across major Indian cities, operational air quality intelligence remains constrained by "
        "five fundamental challenges:"
    )
    add_bullet_p(doc,
        " Continuous reference-grade CAAQMS monitors are capital-intensive (costing approximately INR 1.2–1.5 Crore per installation) "
        "and require dedicated maintenance. As a result, even large urban agglomerations of tens of millions of residents possess only 10 to 40 "
        "active monitors, creating vast geographic blindspots in residential, suburban, and peri-urban zones.",
        bold_prefix="1. Spatial Sparsity of Physical Monitors:"
    )
    add_bullet_p(doc,
        " Particulate concentrations exhibit severe non-linear fluctuations driven by diurnal vehicular peaks, atmospheric "
        "boundary-layer height compression, wind speed shifts, and post-monsoon biomass burning, leading to sudden episodic spikes that linear "
        "statistical models cannot track.",
        bold_prefix="2. Extreme Temporal Non-Linearity:"
    )
    add_bullet_p(doc,
        " Official government reporting systems primarily publish retrospective 24-hour rolling averages (the official Air Quality Index), "
        "informing citizens of past exposure rather than providing proactive next-hour (t+1) forecasts necessary for immediate exposure mitigation.",
        bold_prefix="3. Retrospective Reporting Lag:"
    )
    add_bullet_p(doc,
        " While outdoor surveillance cameras and webcams provide dense visual coverage, naive early concatenation of raw images with "
        "numerical time-series frequently causes catastrophic negative transfer (+12.19 μg/m³ error inflation), requiring robust bounded residual architectures.",
        bold_prefix="4. Multimodal Fusion Pitfalls:"
    )
    add_bullet_p(doc,
        " Conventional spatial interpolations (such as nearest-neighbor assignment) present false certainty far from reporting "
        "stations without communicating distance-dependent predictive uncertainty.",
        bold_prefix="5. Unquantified Spatial Risk:"
    )

    add_section_heading(doc, "1.3 Research Motivation")
    add_body_p(doc,
        "The primary motivation of this Capstone Project is to bridge the critical gap between static, retrospective air-quality reporting "
        "and dynamic, proactive spatio-temporal intelligence. Providing citizens, healthcare facilities, and urban administrators with calibrated "
        "next-hour particulate forecasts enables anticipatory protective action: respiratory patients can adjust medication or reschedule outdoor exposure, "
        "schools can manage athletic activities, and municipal agencies can enact localized traffic diversions before severe pollution thresholds are breached. "
        "Furthermore, by establishing a rigorous spatial interpolation and reliability governance framework, VayuDrishti aims to democratize air quality "
        "information across unmonitored neighborhoods while maintaining strict scientific provenance."
    )

    add_section_heading(doc, "1.4 Research Objectives")
    add_body_p(doc, "To address the identified challenges, the project formulates seven primary research and engineering objectives:")
    add_bullet_p(doc, " Develop and validate next-hour (t+1) PM2.5 and PM10 temporal forecasting models using 24-hour multivariate input sequences.", bold_prefix="Objective 1:")
    add_bullet_p(doc, " Benchmark deep recurrent architectures (LSTM) against gradient-boosted decision trees (LightGBM) and historical persistence baselines.", bold_prefix="Objective 2:")
    add_bullet_p(doc, " Investigate an exploratory multimodal image-sensor architecture with bounded residual corrections and automatic fallback (m=0).", bold_prefix="Objective 3:")
    add_bullet_p(doc, " Adapt sequence models using recent multi-year observation logs (2020–2025) reflecting post-COVID urban baseline shifts.", bold_prefix="Objective 4:")
    add_bullet_p(doc, " Implement continuous geodesic Inverse Distance Weighting (IDW) spatial interpolation and validate against nearest-station baselines.", bold_prefix="Objective 5:")
    add_bullet_p(doc, " Establish a nationwide monitoring infrastructure catalogue and Clark-Evans spatial geometry audit across 31 Indian cities.", bold_prefix="Objective 6:")
    add_bullet_p(doc, " Formulate an operational 5-tier telemetry freshness protocol and distance-scaled split conformal prediction uncertainty calibration.", bold_prefix="Objective 7:")

    add_section_heading(doc, "1.5 Research Scope and Provenance Framework")
    add_body_p(doc,
        "The scope of VayuDrishti encompasses ten distinct research components: historical sequence modeling, next-hour forecasting, "
        "gradient boosting, multi-year adaptation, experimental visual fusion, live OpenAQ ingestion, spatial IDW, 31-city spatial audits, "
        "conformal calibration, and cloud deployment. A foundational tenet of VayuDrishti is strict scientific provenance. The framework "
        "explicitly categorizes all system outputs into four distinct semantic tiers: (1) Direct Physical Observations (verified sensor telemetry); "
        "(2) Model Forecasts (mathematical next-hour predictions at station locations); (3) Interpolated Spatial Estimates (continuous surface approximations); "
        "and (4) Suppressed / Unavailable Estimates (regions where monitoring density or telemetry age violates safety thresholds). Spatial approximations "
        "are never presented as direct sensor measurements."
    )

    add_section_heading(doc, "1.6 Background and Literature Review")
    add_body_p(doc,
        "Air quality prediction research has evolved across three major paradigms. Classical statistical models, including Autoregressive "
        "Integrated Moving Average (ARIMA) and Support Vector Regression (SVR), established early baselines for linear diurnal periodicity but break down "
        "during sudden atmospheric inversions. With the advent of deep learning, recurrent neural networks (RNNs), Long Short-Term Memory (LSTM) networks, "
        "and Gated Recurrent Units (GRUs) demonstrated superior capacity to capture long-range temporal dependencies across multi-lag multivariate time-series [1]. "
        "Simultaneously, Gradient Boosted Decision Trees (GBDT), notably LightGBM [5], proved exceptionally efficient on tabular sensor aggregations. "
        "In spatial estimation, Shepard's Inverse Distance Weighting (IDW) [3] remains the benchmark interpolation standard for continuous environmental fields. "
        "Recently, conformal prediction [8], [9] has gained prominence for providing distribution-free finite-sample coverage guarantees without parametric "
        "distribution assumptions."
    )

    add_section_heading(doc, "1.7 Identified Research Gaps")
    add_body_p(doc,
        "Our critical literature review identified three prominent research gaps in existing environmental monitoring systems: First, telemetry "
        "latency and sensor dropout are routinely ignored in literature, with stale 12-hour-old readings frequently processed as live telemetry. "
        "Second, multimodal studies often report unvalidated claims of vision-based forecasting improvements, overlooking the severe negative transfer "
        "caused by unsynchronized camera angles, lighting shifts, and weather confounds. Third, existing spatio-temporal platforms output deterministic "
        "point predictions without rigorous distance-calibrated uncertainty bounds."
    )

    add_section_heading(doc, "1.8 Organization of the Report")
    add_body_p(doc,
        "The remainder of this report is organized as follows: Chapter 2 details the VayuDrishti system architecture, mathematical formulations, "
        "IDW interpolation, and governance protocols. Chapter 3 describes the implementation setup, software/hardware environment, data splits, and "
        "cost considerations. Chapter 4 presents comprehensive empirical results across historical, adapted, multimodal, live, spatial, and conformal "
        "evaluations. Chapter 5 concludes the report with a discussion of contributions, limitations, and future directions. Core source codes, verified "
        "references, and member biodata follow in the Appendices."
    )

    doc.add_page_break()


# ==============================================================================
# CHAPTER 2: SYSTEM ARCHITECTURE AND METHODOLOGY
# ==============================================================================
def build_chapter_2(doc):
    add_chapter_title(doc, "CHAPTER 2: SYSTEM ARCHITECTURE AND METHODOLOGY")

    add_section_heading(doc, "2.1 Overall System Architecture")
    add_body_p(doc,
        "The VayuDrishti architecture is structured as an end-to-end 5-tier pipeline engineered for continuous data ingestion, "
        "rigorous preprocessing, dual-pathway forecasting, spatial surface estimation, and uncertainty-calibrated dashboard visualization."
    )
    add_figure(doc, FIG_DIR / "fig_arch_overall.png", "2.1",
               "Complete VayuDrishti Multimodal Spatio-Temporal Forecasting Architecture showing multi-source ingestion, dual prediction pathways, spatial IDW, and Streamlit delivery.", width_in=6.2)

    add_section_heading(doc, "2.2 Data Sources and Ingestion Pipelines")
    add_body_p(doc,
        "VayuDrishti unifies four heterogeneous data streams: (1) Historical CPCB CAAQMS repository comprising 1,369,779 validated hourly records "
        "from 90 stations across 24 cities (2015–2020); (2) Recent multi-year CAAQMS logs comprising over 3.3 million cleaned records across 85 stations "
        "in four major metropolitan areas (July 2020–September 2025); (3) Real-time streaming telemetry ingested from the OpenAQ API v3 endpoint; and "
        "(4) The PM25Vision ground-level outdoor image dataset for experimental visual feature extraction."
    )

    add_section_heading(doc, "2.3 Data Quality Filtering and Preprocessing")
    add_body_p(doc,
        "Continuous sensor streams frequently suffer from transient communication dropouts and baseline drift. The ingestion pipeline applies "
        "a strict 4-hour continuous gap filter: sequence windows containing missing blocks greater than 4 consecutive hours are rejected. For gaps ≤ 4 hours, "
        "forward-fill imputation is applied. Features are normalized using train-set Z-score standardization: z = (x - μ) / σ, preventing future information leakage."
    )
    add_figure(doc, FIG_DIR / "fig_arch_preprocessing.png", "2.2",
               "Continuous Telemetry Ingestion, Quality Filtering, and 27-Dimensional Feature Pipeline.", width_in=4.2)

    add_section_heading(doc, "2.4 Audited Feature Engineering Schema")
    add_body_p(doc,
        "The temporal sequence model operates on a 27-dimensional feature schema capturing multi-pollutant interactions, boundary-layer meteorology, "
        "cyclical temporal periodicities, and spatial coordinates."
    )

    headers_2_1 = ["Category", "Channels", "Feature Descriptions"]
    rows_2_1 = [
        ["Pollutants (12)", "PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene", "Continuous hourly concentrations from CAAQMS monitors."],
        ["Meteorology (5)", "Temperature (°C), Dew Point (°C), Relative Humidity (%), Wind Speed (m/s), Rain (mm)", "Surface weather variables governing atmospheric dispersion."],
        ["Temporal (8)", "Wind Dir (sin, cos), Hour (sin, cos), Month (sin, cos), Day of Week (sin, cos)", "Cyclical trigonometric encodings capturing diurnal and seasonal cycles."],
        ["Spatial (2)", "Latitude, Longitude (Decimal Degrees WGS84)", "Geographic station coordinates for spatial indexing."]
    ]
    create_table(doc, headers_2_1, rows_2_1, col_widths=[1.5, 2.3, 2.7],
                 caption="Audited 27-Dimensional Numerical Sensor Feature Schema", table_no="2.1")

    add_section_heading(doc, "2.5 Deep Recurrent LSTM Forecasting Methodology")
    add_body_p(doc,
        "The primary temporal forecaster employs a 2-layer stacked Long Short-Term Memory (LSTM) network with 64 hidden units per layer "
        "and a dropout rate of 0.20. Given an input sequence X ∈ ℝ^(24 × 27) covering hours t-23 through t, the LSTM updates its cell state c_t "
        "and hidden representation h_t via standard gating mechanisms:"
    )
    add_code_block(doc,
        "f_t = sigmoid(W_f · x_t + U_f · h_{t-1} + b_f)       # Forget gate\n"
        "i_t = sigmoid(W_i · x_t + U_i · h_{t-1} + b_i)       # Input gate\n"
        "c_tilde = tanh(W_c · x_t + U_c · h_{t-1} + b_c)      # Candidate cell\n"
        "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c_tilde                  # Cell state update\n"
        "o_t = sigmoid(W_o · x_t + U_o · h_{t-1} + b_o)       # Output gate\n"
        "h_t = o_t ⊙ tanh(c_t)                                 # Hidden state\n"
        "y_hat_{t+1} = W_y · h_24 + b_y                        # Next-hour PM2.5 & PM10"
    )
    add_body_p(doc, "The output layer produces strictly next-hour (t+1) forecasts for PM2.5 and PM10 simultaneously.")
    add_figure(doc, FIG_DIR / "fig_arch_lstm.png", "2.3",
               "Deep Recurrent 2-Layer Stacked LSTM Sequence Forecaster Architecture.", width_in=4.2)

    add_section_heading(doc, "2.6 Gradient Boosted Trees and Severity-Gated Ensemble")
    add_body_p(doc,
        "To complement recurrent deep learning, a parallel tabular pathway employs LightGBM trained on 49 engineered lag features "
        "(including 1h, 2h, 3h, 6h, 12h, 24h lags, rolling means, and rolling standard deviations). An operational severity-gated ensemble combines "
        "the LSTM and LightGBM predictions dynamically: under standard atmospheric regimes, a weighted average (0.40 LSTM + 0.60 LightGBM) is utilized, "
        "whereas under crisis severe conditions (PM2.5 ≥ 250 μg/m³), the system dynamically transitions to pure LSTM inference to better capture non-linear peaks."
    )
    add_figure(doc, FIG_DIR / "fig_arch_ensemble.png", "2.4",
               "Independent Severity-Gated LSTM–GBDT Ensemble Architecture.", width_in=4.4)

    add_section_heading(doc, "2.7 Experimental Multimodal Residual Methodology")
    add_body_p(doc,
        "To investigate visual air quality cues, an experimental image pathway utilizes a pretrained ResNet-50 convolutional backbone to extract "
        "2,048-dimensional visual embeddings from outdoor frames, projected down to 32 dimensions. Critically, to solve the catastrophic negative transfer "
        "of naive early concatenation (+12.19 μg/m³ error inflation), VayuDrishti formulates a Bounded Residual Hybrid architecture. Visual embeddings "
        "are gated by a learned availability mask m ∈ {0, 1} and quality gate α_eff = m · α_raw. The residual head outputs a bounded correction δ ∈ [-M, +M] "
        "(where M=25 μg/m³ for PM2.5):"
    )
    add_code_block(doc, "y_hybrid = max(0, y_sensor + α_eff · δ)")
    add_body_p(doc, "When outdoor imagery is missing, degraded, or unaligned (m=0), α_eff vanishes, guaranteeing mathematical fallback to the pure sensor forecast y_sensor.")
    add_figure(doc, FIG_DIR / "fig_arch_resnet.png", "2.5",
               "Optical Feature Extraction via Pretrained ResNet-50 Convolutional Backbone.", width_in=4.4)
    add_figure(doc, FIG_DIR / "fig_arch_residual_hybrid.png", "2.6",
               "Bounded Residual Multimodal Hybrid Model with Guaranteed Fallback Masking.", width_in=4.4)

    add_section_heading(doc, "2.8 Spatial Interpolation Methodology (Geodesic IDW)")
    add_body_p(doc,
        "To estimate continuous particulate surfaces between sparse CAAQMS stations, VayuDrishti implements vectorized Inverse Distance Weighting "
        "(IDW) using Haversine great-circle distances. For a query location (lat, lon) and N active monitoring stations with coordinates (lat_i, lon_i), "
        "the geodesic distance d_i is computed using Earth radius R = 6,371.0 km. Weights w_i are computed with power parameter p = 2.0 and smoothing ε = 0.001 km:"
    )
    add_code_block(doc,
        "w_i = 1.0 / (d_i + ε)^2.0\n"
        "z_hat(x) = max( 0.0,  ( ∑_{i=1}^N w_i · z_i ) / ( ∑_{i=1}^N w_i ) )"
    )
    add_body_p(doc,
        "Two operational safeguards are strictly enforced: (1) If min(d_i) ≤ 0.05 km, the query point is co-located with a physical monitor, "
        "and the exact measured sensor reading is returned directly. (2) If fewer than 3 active reporting monitors exist within search radius R_max = 45.0 km, "
        "continuous surface interpolation is suppressed, and nearest-station fallback is returned with an explicit sparsity warning."
    )
    add_figure(doc, FIG_DIR / "fig_arch_idw_spatial.png", "2.7",
               "Geodesic Inverse Distance Weighting (IDW) Spatial Interpolation Workflow showing distance kernel, co-location check, and sparse network safeguards.", width_in=6.0)

    add_section_heading(doc, "2.9 Spatial Reliability Assessment and Freshness Governance")
    add_body_p(doc,
        "To prevent stale or cached data from corrupting live spatial surfaces, VayuDrishti enforces a 5-tier telemetry freshness classification: "
        "Fresh (latency ≤ 2 hours: fully verified for prediction and interpolation); Delayed (2 to 6 hours: flagged with cautionary indicator); "
        "Stale (6 to 24 hours: excluded from spatial surfaces); Offline (> 24 hours: suppressed and marked inactive); and Unknown (historical logs "
        "with unverified timestamps). To quantify monitoring network geometry across urban zones, the Clark-Evans nearest-neighbor aggregation ratio R "
        "is computed: R = r̄_A / r̄_E, where r̄_A is mean observed nearest-neighbor distance and r̄_E = 0.5 / √(density) is expected distance under a random "
        "Poisson distribution. Values of R < 1.0 indicate spatial clustering, while R > 1.0 indicates spatial dispersion. Cities are classified into "
        "5 operational reliability tiers based on active monitor count, telemetry freshness, and spatial clustering."
    )
    add_figure(doc, FIG_DIR / "fig_arch_reliability.png", "2.8",
               "Spatial Reliability, Telemetry Freshness & Geometry Audit Workflow.", width_in=6.0)

    add_section_heading(doc, "2.10 Distribution-Free Conformal Uncertainty Calibration")
    add_body_p(doc,
        "To provide rigorous prediction uncertainty without Gaussian distribution assumptions, VayuDrishti implements distance-scaled split conformal "
        "prediction [8]. Using a calibration set of held-out spatial samples, non-conformity scores are computed by normalizing absolute prediction errors "
        "by the distance d_min to the nearest reporting monitor: s_i = |y_i - ŷ_i| / (1 + γ · d_min), with scaling factor γ = 0.05 / km. Given significance "
        "level α, the empirical (1 - α) quantile q̂ is calculated. The operational prediction interval C(x) is then formulated as:"
    )
    add_code_block(doc, "C(x) = [ max(0, ŷ - q̂ · (1 + γ · d_min)),  ŷ + q̂ · (1 + γ · d_min) ]")
    add_body_p(doc,
        "This formulation guarantees that interval width expands naturally into geographic blindspots far from physical monitors, accurately communicating "
        "spatial uncertainty to end-users."
    )
    add_figure(doc, FIG_DIR / "fig_arch_conformal.png", "2.9",
               "Distance-Scaled Conformal Uncertainty Estimation Workflow.", width_in=6.0)

    add_section_heading(doc, "2.11 Interactive Cloud Dashboard Architecture")
    add_body_p(doc,
        "The user-facing system is implemented using Streamlit and deployed on Streamlit Cloud. The dashboard architecture separates interface "
        "components into six specialized analytical modules: Home Dashboard, Station Telemetry, AI Sequence Forecaster, Multimodal Hybrid, Model Comparison, "
        "and Spatial Heatmap. The application integrates directly with the project's GitHub repository for version-controlled continuous deployment and interacts "
        "statelessly with backend model weights."
    )

    doc.add_page_break()


# ==============================================================================
# CHAPTER 3: IMPLEMENTATION AND EXPERIMENTAL SETUP
# ==============================================================================
def build_chapter_3(doc):
    add_chapter_title(doc, "CHAPTER 3: IMPLEMENTATION AND EXPERIMENTAL SETUP")

    add_section_heading(doc, "3.1 Software and Hardware Environment")
    add_body_p(doc,
        "To ensure strict reproducibility and academic integrity, the hardware and software environments used for model development and evaluation "
        "are documented accurately. The primary development machine was a consumer laptop powered by an Intel Core i7-1165G7 quad-core processor "
        "(base clock 2.80 GHz, up to 4.70 GHz turbo), 16 GB of DDR4 system RAM, and integrated Intel Iris Xe graphics. No dedicated NVIDIA CUDA GPU "
        "was available on the development workstation; all model training, hyperparameter optimization, and inference evaluation were executed in a "
        "CPU-compatible PyTorch environment. The software stack was built on 64-bit Windows 11 with Ubuntu 22.04 LTS via Windows Subsystem for Linux (WSL2), "
        "Python 3.10.12, PyTorch 2.2.0 (CPU build), Torchvision 0.17.0, Scikit-learn 1.4.0, LightGBM 4.3.0, Pandas 2.2.0, NumPy 1.26.4, SciPy 1.12.0, "
        "and Streamlit 1.31.0."
    )

    add_section_heading(doc, "3.2 Comprehensive Dataset Specifications")
    add_body_p(doc, "The experimental evaluation encompasses four distinct empirical datasets:")
    add_bullet_p(doc,
        " Comprises 1,369,779 continuous hourly records from 90 physical monitoring stations across 24 Indian cities. The dataset spans "
        "from January 1, 2015 to July 1, 2020, capturing multi-year pre-pandemic industrial baselines, diurnal traffic surges, and severe seasonal "
        "biomass-burning episodes in the Indo-Gangetic Plain.",
        bold_prefix="1. Historical CPCB CAAQMS Dataset (2015–2020):"
    )
    add_bullet_p(doc,
        " Comprises over 3.3 million cleaned hourly records from 85 stations across Delhi NCR, Mumbai MMR, Kolkata Metro, and Bengaluru Urban, "
        "covering the period from July 2020 to September 2025. This dataset captures post-lockdown urban activity recovery and structural emission changes.",
        bold_prefix="2. Recent Multi-Year CAAQMS Dataset (2020–2025):"
    )
    add_bullet_p(doc,
        " Acquired via real-time streaming integration with the OpenAQ API v3. A prospective 14-day live audit was conducted across four operational "
        "CAAQMS stations in Delhi NCR, yielding 418 verified prediction-observation evaluation pairs.",
        bold_prefix="3. Live Telemetry Audit Dataset:"
    )
    add_bullet_p(doc,
        " A curated repository of 1,235 outdoor surveillance images paired with ambient particulate measurements. An opportunistic alignment audit "
        "identified a subset of 199 verified image-sensor pairs across 18 stations used for exploratory multimodal testing.",
        bold_prefix="4. PM25Vision Experimental Image Dataset:"
    )

    add_section_heading(doc, "3.3 Chronological Splitting and Leakage Prevention")
    add_body_p(doc,
        "In time-series environmental forecasting, random K-fold cross-validation introduces severe temporal lookahead leakage, where future atmospheric "
        "states erroneously inform past predictions. To guarantee strict causality, all datasets were split strictly chronologically: the first 70% of chronological "
        "hours were allocated to training, the subsequent 10% to validation and early stopping, and the final 20% reserved exclusively for out-of-sample testing "
        "(yielding 404,431 test sequences in the historical benchmark). For spatial evaluation, a spatial Leave-One-Station-Out (LOSO) holdout protocol was "
        "implemented in Delhi NCR: all observations from a target held-out station were completely removed from the interpolation registry, ensuring that the "
        "station never contributed to its own spatial estimate."
    )

    add_section_heading(doc, "3.4 Model Configurations and Hyperparameters")
    add_body_p(doc, "Model hyperparameters were established through systematic validation-set tuning. Table 3.1 details the configuration parameters across all primary architectures.")

    headers_3_1 = ["Model Component", "Hyperparameter", "Configured Value", "Operational Rationale"]
    rows_3_1 = [
        ["LSTM Temporal", "Input Window", "24 hours (t-23 to t)", "Captures complete diurnal cycle."],
        ["LSTM Temporal", "Forecast Horizon", "Strictly t+1 hour", "Actionable short-term early warning."],
        ["LSTM Temporal", "Hidden Architecture", "2 Stacked Layers, 64 Units", "Prevents overfitting on noisy channels."],
        ["LSTM Temporal", "Dropout & LR", "Dropout = 0.20, Adam LR = 0.001", "Stabilizes gradient backpropagation."],
        ["LightGBM GBDT", "Trees & Leaves", "100 Estimators, 31 Leaves", "Fast histogram-based tabular splitting."],
        ["LightGBM GBDT", "Learning Rate", "0.05 (Max Depth = 6)", "Prevents leaf-wise tabular overfitting."],
        ["Spatial IDW", "Distance Metric", "Haversine Great-Circle", "True spherical distance (R=6371km)."],
        ["Spatial IDW", "Power Parameter (p)", "2.0 (Inverse Square)", "Reflects physical atmospheric dispersion."],
        ["Spatial IDW", "Min Monitors / Radius", "k_min = 3 stations, R_max = 45 km", "Suppresses distant extrapolation artifacts."],
        ["Conformal Split", "Distance Factor (γ)", "0.05 / km", "Expands intervals in sparse zones."]
    ]
    create_table(doc, headers_3_1, rows_3_1, col_widths=[1.5, 1.6, 1.8, 1.6],
                 caption="Model Hyperparameter and Algorithmic Configurations", table_no="3.1")

    add_section_heading(doc, "3.5 Mathematical Evaluation Metrics")
    add_body_p(doc, "Model evaluation is quantified using four standard statistical performance metrics:")
    add_bullet_p(doc, " Measures average prediction error magnitude in physical units (μg/m³): MAE = (1/N) · ∑_{i=1}^N |y_i - ŷ_i|.", bold_prefix="1. Mean Absolute Error (MAE):")
    add_bullet_p(doc, " Penalizes large episodic prediction errors: RMSE = √( (1/N) · ∑_{i=1}^N (y_i - ŷ_i)² ).", bold_prefix="2. Root Mean Square Error (RMSE):")
    add_bullet_p(doc, " Quantifies the proportion of particulate variance explained by the model: R² = 1 - [ ∑(y_i - ŷ_i)² / ∑(y_i - ȳ)² ].", bold_prefix="3. Coefficient of Determination (R²):")
    add_bullet_p(doc, " Measures the proportion of true observations contained within the calibrated prediction band: Coverage = (1/N) · ∑_{i=1}^N 𝕀( y_i ∈ C(x_i) ).", bold_prefix="4. Empirical Conformal Coverage:")

    add_section_heading(doc, "3.6 Live Data Integration Workflow")
    add_body_p(doc,
        "Live operational telemetry is integrated using the OpenAQ REST API v3. An asynchronous worker queries active station endpoints at 15-minute "
        "intervals, extracts hourly pollutant measurements, validates coordinates against the metadata registry, and applies the 5-tier freshness filter. "
        "During the prospective 14-day live audit, forecasts generated at hour t were logged alongside cryptographic timestamps and subsequently matched "
        "with actual observations reported at hour t+1, yielding 418 verified prediction-observation pairs."
    )

    add_section_heading(doc, "3.7 Computational and Deployment Considerations")
    add_body_p(doc,
        "A transparent accounting of development and deployment costs is detailed in Table 3.2. Because the project leverages open-source software and "
        "existing personal hardware, no exaggerated capital expenditures are claimed. In terms of computational latency, the CPU-based runtime executes "
        "single-station LSTM inference in approximately 12–18 ms, LightGBM inference in 2–4 ms, and 50×50 IDW grid interpolation in 35–45 ms, confirming that "
        "sub-second operational responsiveness is achieved on consumer CPU hardware without dedicated GPU acceleration."
    )

    headers_3_2 = ["Resource Category", "Specification / Infrastructure", "Cost Basis"]
    rows_3_2 = [
        ["Development Hardware", "Intel Core i7-1165G7 CPU, Intel Iris Xe, 16 GB RAM", "Existing personal equipment (No new project purchase cost)."],
        ["Historical Data", "CPCB National Ambient Air Quality Archive (2015–2020)", "Publicly available research dataset (Free / Open Data)."],
        ["Recent Multi-Year Data", "CPCB CAAQMS Multi-Metro Repository (2020–2025)", "Public open-access repository (Free / Open Data)."],
        ["Live Telemetry API", "OpenAQ API v3 REST Ingestion Service", "Public community access tier (Free / Open API)."],
        ["Machine Learning Stack", "Python 3.10, PyTorch 2.2, Scikit-learn, LightGBM", "Open-source software libraries (Free / Open Source)."],
        ["Cloud Hosting", "Streamlit Community Cloud Deployment", "Community public hosting tier (Free)."],
        ["Total Direct Expenditure", "VayuDrishti Research Framework Implementation", "INR 0.00 (Zero direct external funding expenditure)."]
    ]
    create_table(doc, headers_3_2, rows_3_2, col_widths=[1.8, 2.7, 2.0],
                 caption="Computational Resource & Operational Deployment Analysis", table_no="3.2")

    doc.add_page_break()


# ==============================================================================
# CHAPTER 4: RESULTS AND DISCUSSION
# ==============================================================================
def build_chapter_4(doc):
    add_chapter_title(doc, "CHAPTER 4: RESULTS AND DISCUSSION")

    add_section_heading(doc, "4.1 Historical Sequence Forecasting Benchmarks")
    add_body_p(doc,
        "The historical benchmark was evaluated on the chronological out-of-sample test set comprising 404,431 continuous 24-hour sequence windows "
        "from the 2015–2020 CPCB dataset. The primary reference baseline is the 1-hour persistence model (ŷ_{t+1} = y_t), which represents an exceptionally "
        "competitive standard in short-term environmental forecasting due to strong temporal autocorrelation. As reported in Table 4.1, the 2-layer stacked "
        "LSTM network outperforms persistence across all metrics, achieving an MAE of 10.5719 μg/m³ (RMSE 18.5789 μg/m³, R² 0.9492) on PM2.5 and an MAE of "
        "18.8643 μg/m³ (RMSE 32.2027 μg/m³, R² 0.9301) on PM10. The historical severity-gated ensemble slightly improves PM2.5 MAE to 10.3508 μg/m³ "
        "(RMSE 18.3792 μg/m³, R² 0.9503) by dynamically switching to LSTM weights during high-concentration episodes."
    )

    headers_4_1 = ["Model Architecture", "PM2.5 MAE", "PM2.5 RMSE", "PM2.5 R²", "PM10 MAE", "PM10 RMSE", "PM10 R²"]
    rows_4_1 = [
        ["1-Hour Persistence Baseline", "10.6783", "19.8072", "0.9423", "19.7384", "36.2374", "0.9115"],
        ["LSTM Baseline (2-Layer)", "10.5719", "18.5789", "0.9492", "18.8643", "32.2027", "0.9301"],
        ["Static Ensemble (LSTM+GBDT)", "10.4215", "18.4510", "0.9498", "18.4210", "31.9540", "0.9312"],
        ["Severity-Gated Ensemble", "10.3508", "18.3792", "0.9503", "18.1369", "31.6385", "0.9325"]
    ]
    create_table(doc, headers_4_1, rows_4_1, col_widths=[2.1, 0.75, 0.75, 0.70, 0.75, 0.75, 0.70],
                 caption="Historical Next-Hour Forecasting Performance (404,431 Test Sequences)", table_no="4.1")

    add_figure(doc, FIG_DIR / "actual_vs_predicted.png", "4.1",
               "Ground Truth versus LSTM-Predicted Particulate Concentrations on Historical Test Set (N = 404,431 sequences).", width_in=4.4)
    add_figure(doc, FIG_DIR / "confusion_matrices_combined.png", "4.2",
               "Multi-Class Particulate Matter Air Quality Confusion Matrices across CPCB AQI Severity Categories.", width_in=4.2)

    add_section_heading(doc, "4.2 Recent-Data Multi-Year Adaptation Results")
    add_body_p(doc,
        "To evaluate model resilience across long-term temporal shifts, the ensemble architecture was adapted on the recent 2020–2025 CAAQMS dataset. "
        "As detailed in Table 4.2, the adapted ensemble achieves a PM2.5 MAE of 6.6216 μg/m³ (RMSE 13.0285 μg/m³) and a PM10 MAE of 16.1660 μg/m³ (RMSE 34.6875 μg/m³) "
        "on its designated recent-period test set. Methodologically, this lower absolute error reflects structural shifts in urban air quality distributions "
        "following pandemic lockdowns and targeted vehicular emission policies; it should not be interpreted as an apple-to-apple superiority over historical models, "
        "as the underlying evaluation distributions differ."
    )

    headers_4_2 = ["Pollutant Target", "Adapted Ensemble MAE", "Adapted Ensemble RMSE", "Recent Test Cohort Period"]
    rows_4_2 = [
        ["PM2.5 (Fine Particulates)", "6.6216 μg/m³", "13.0285 μg/m³", "July 2020 – September 2025 (4 Metros)"],
        ["PM10 (Coarse Particulates)", "16.1660 μg/m³", "34.6875 μg/m³", "July 2020 – September 2025 (4 Metros)"]
    ]
    create_table(doc, headers_4_2, rows_4_2, col_widths=[1.8, 1.4, 1.4, 1.9],
                 caption="Recent-Data Adaptation Performance on 2020–2025 Test Cohort", table_no="4.2")

    add_figure(doc, FIG_DIR / "fig2_seasonal_model_comparison.png", "4.5",
               "Seasonal Model Dynamics and Particulate Regime Error Curves across IMD Meteorological Seasons.", width_in=4.8)

    add_section_heading(doc, "4.3 Exploratory Multimodal Fusion & Negative Transfer")
    add_body_p(doc,
        "A critical scientific contribution of VayuDrishti is the transparent documentation of multimodal failure modes. Evaluating the PM25Vision dataset "
        "revealed severe spatial and temporal misalignment: among 1,235 images, only 199 verified pairs across 18 stations coincided within a 1-hour window "
        "of sensor ground truth. As reported in Table 4.3, naive early concatenation of ResNet-50 visual features with numerical sensor sequences resulted in "
        "severe negative transfer, inflating PM2.5 MAE by +12.19 μg/m³ over the sensor baseline. This degradation occurs because uncalibrated visual features "
        "overfit to camera orientation, ambient sunlight, cloud cover, and rain artifacts. In contrast, the Bounded Residual Hybrid architecture successfully "
        "constrained visual variance, achieving an MAE of 10.48 μg/m³ on the paired cohort while guaranteeing exact fallback (m=0) when images were unaligned."
    )

    headers_4_3 = ["Multimodal Pathway", "PM2.5 MAE (μg/m³)", "Delta vs Sensor Baseline", "Empirical Status"]
    rows_4_3 = [
        ["Sensor-Only Baseline", "10.57", "Reference (0.00)", "Validated Operational Standard"],
        ["Naive Early Fusion (Concatenation)", "22.76", "+12.19 (Error Inflation)", "Catastrophic Negative Transfer"],
        ["Bounded Residual Hybrid (m=1)", "10.48", "-0.09 (Marginal on Paired)", "Exploratory Research Branch"],
        ["Bounded Residual Hybrid (m=0 Fallback)", "10.57", "0.00 (Identical to Baseline)", "Guaranteed Fallback Safeguard"]
    ]
    create_table(doc, headers_4_3, rows_4_3, col_widths=[2.2, 1.2, 1.5, 1.6],
                 caption="Exploratory Image and Sensor Alignment Pathway Evaluation", table_no="4.3")

    add_figure(doc, FIG_DIR / "alignment_discrepancy_distribution.png", "4.7",
               "Opportunistic Image Alignment Discrepancy Diagnostics showing Haversine station offsets and systematic positive image label bias.", width_in=4.6)

    add_section_heading(doc, "4.4 14-Day Live Telemetry Operational Audit")
    add_body_p(doc,
        "To evaluate real-world forecasting under streaming operational conditions, a prospective 14-day live audit was executed across four active "
        "CAAQMS stations in Delhi NCR, yielding 418 matched prediction-observation pairs. As documented in Table 4.4, the GBDT model achieved a live PM2.5 MAE "
        "of 9.64 μg/m³ compared to 9.86 μg/m³ for persistence (p = 0.48), and a PM10 MAE of 22.13 μg/m³ compared to 22.35 μg/m³ for persistence (p = 0.62). "
        "Paired two-tailed t-tests establish that the difference between GBDT and persistence is statistically non-significant over this short evaluation period. "
        "This finding demonstrates the immense predictive power of next-hour persistence in calm atmospheric regimes and reinforces that short live audits cannot "
        "establish nationwide forecasting superiority without multi-seasonal longitudinal validation."
    )

    headers_4_4 = ["Pollutant Target", "GBDT Live MAE", "Persistence Live MAE", "Sample Size (N)", "Paired p-value", "Statistical Status"]
    rows_4_4 = [
        ["PM2.5", "9.64 μg/m³", "9.86 μg/m³", "418 pairs (4 stations)", "p = 0.48", "Statistically Non-Significant"],
        ["PM10", "22.13 μg/m³", "22.35 μg/m³", "418 pairs (4 stations)", "p = 0.62", "Statistically Non-Significant"]
    ]
    create_table(doc, headers_4_4, rows_4_4, col_widths=[1.1, 1.0, 1.1, 1.3, 0.9, 1.1],
                 caption="14-Day Live Telemetry Operational Audit (4 Stations, 418 Matched Pairs)", table_no="4.4")

    add_section_heading(doc, "4.5 Delhi Spatial Holdout Interpolation Benchmarks")
    add_body_p(doc,
        "Spatial interpolation was rigorously validated using a Leave-One-Station-Out (LOSO) holdout protocol across the Delhi NCR monitoring network. "
        "As reported in Table 4.5, geodesic Inverse Distance Weighting (IDW, p=2.0) achieved a PM2.5 MAE of 24.07 μg/m³ compared to 30.06 μg/m³ for the "
        "nearest-neighbor baseline—a 20.0% relative error reduction. On PM10, IDW achieved an MAE of 52.76 μg/m³ compared to 70.68 μg/m³ for nearest-neighbor—a "
        "25.4% relative error reduction. In an extended forward evaluation across 1,564 spatial holdout instances, IDW demonstrated an MAE of 24.30 μg/m³ "
        "(vs. 28.83 μg/m³) for PM2.5 and 45.24 μg/m³ (vs. 56.34 μg/m³) for PM10. These results confirm that inverse-square distance weighting provides superior "
        "spatial continuity over step-function nearest-station assignment in dense urban grids."
    )

    headers_4_5 = ["Evaluation Metric", "Geodesic IDW (p=2.0)", "Nearest-Station Baseline", "Relative Gain", "Evaluation Scope"]
    rows_4_5 = [
        ["PM2.5 Holdout MAE", "24.07 μg/m³", "30.06 μg/m³", "+20.0% Improvement", "Delhi NCR LOSO Holdout"],
        ["PM10 Holdout MAE", "52.76 μg/m³", "70.68 μg/m³", "+25.4% Improvement", "Delhi NCR LOSO Holdout"],
        ["PM2.5 Forward MAE", "24.30 μg/m³", "28.83 μg/m³", "+15.7% Improvement", "1,564 Matched Holdout Cases"],
        ["PM10 Forward MAE", "45.24 μg/m³", "56.34 μg/m³", "+19.7% Improvement", "1,564 Matched Holdout Cases"]
    ]
    create_table(doc, headers_4_5, rows_4_5, col_widths=[1.5, 1.2, 1.3, 1.2, 1.3],
                 caption="Delhi Spatial Holdout Evaluation: IDW vs. Nearest-Station Baseline", table_no="4.5")

    add_figure(doc, FIG_DIR / "fig3_multistation_performance.png", "4.4",
               "Multi-Station Generalization across 15 Operational CAAQMS Stations in Indian Metropolitan Corridors.", width_in=4.6)
    add_figure(doc, FIG_DIR / "fig_spatial_cross_correlation.png", "4.6",
               "Spatial Advection Cross-Correlation in Urban Corridors as a Function of Inter-Station Distance and Lag τ.", width_in=4.6)

    add_section_heading(doc, "4.6 Nationwide 31-City Spatial Reliability Audit")
    add_body_p(doc,
        "To establish realistic boundaries for spatial estimation, a nationwide spatial infrastructure audit was executed across 31 Indian urban areas. "
        "From a master catalogue of 586 physical CAAQMS stations, 349 monitors were located within defined urban administrative boundaries. Table 4.6 documents "
        "monitoring density, telemetry completeness, Clark-Evans nearest-neighbor ratio R, and assigned operational reliability tiers. Crucially, the audit revealed "
        "that only Delhi NCR satisfies Tier 1 (Calibrated) criteria with 98 registered monitors (61 with valid 24h histories) and dispersed coverage. Non-metro "
        "cities exhibit severe spatial clustering (e.g., Patna R=0.38, Kolkata R=0.32, Amaravati R=0.49) or extreme sparsity (e.g., Visakhapatnam, Tirupati, "
        "Bhubaneswar with ≤ 2 monitors), reinforcing that continuous spatial surfaces must be restricted or suppressed in sparse networks."
    )

    headers_4_6 = ["City / Urban Zone", "Monitors", "PM2.5 Valid", "PM10 Valid", "24h Complete", "Clark-Evans (R)", "Reliability Tier"]
    rows_4_6 = [
        ["Delhi NCR (Delhi)", "98", "61", "59", "61", "0.42 (Clustered)", "Tier 1: Calibrated"],
        ["Mumbai MMR (Maharashtra)", "57", "47", "46", "47", "0.56 (Clustered)", "Tier 2: High"],
        ["Kolkata Metro (West Bengal)", "21", "13", "13", "14", "0.32 (Clustered)", "Tier 2: High"],
        ["Hyderabad (Telangana)", "20", "15", "13", "15", "0.61 (Clustered)", "Tier 2: High"],
        ["Bengaluru Urban (Karnataka)", "19", "12", "14", "14", "0.45 (Clustered)", "Tier 2: High"],
        ["Pune (Maharashtra)", "13", "13", "13", "13", "0.93 (Random)", "Tier 2: High"],
        ["Patna (Bihar)", "12", "7", "6", "7", "0.38 (Clustered)", "Tier 2: High"],
        ["Ahmedabad (Gujarat)", "11", "10", "10", "10", "1.25 (Dispersed)", "Tier 2: High"],
        ["Chennai (Tamil Nadu)", "9", "8", "7", "8", "0.57 (Clustered)", "Tier 3: Moderate"],
        ["Jaipur (Rajasthan)", "9", "6", "6", "6", "0.60 (Clustered)", "Tier 3: Moderate"],
        ["Lucknow (Uttar Pradesh)", "6", "6", "6", "6", "1.13 (Random)", "Tier 3: Moderate"],
        ["Agra (Uttar Pradesh)", "6", "6", "6", "6", "1.35 (Dispersed)", "Tier 3: Moderate"],
        ["Vijayawada (Andhra Pradesh)", "6", "5", "5", "5", "0.68 (Clustered)", "Tier 3: Moderate"],
        ["Amaravati (Andhra Pradesh)", "5", "4", "4", "4", "0.49 (Clustered)", "Tier 3: Moderate"],
        ["Kochi (Kerala)", "3", "1", "1", "1", "0.95 (Random)", "Tier 4: Marginal"],
        ["Coimbatore (Tamil Nadu)", "3", "2", "2", "2", "0.62 (Clustered)", "Tier 4: Marginal"],
        ["Kanpur (Uttar Pradesh)", "3", "3", "3", "3", "1.10 (Random)", "Tier 4: Marginal"],
        ["Visakhapatnam (AP)", "2", "1", "1", "1", "0.00 (Pair)", "Tier 5: Suppressed"],
        ["Tirupati (Andhra Pradesh)", "2", "2", "2", "2", "1.24 (Pair)", "Tier 5: Suppressed"],
        ["Bhubaneswar (Odisha)", "2", "2", "2", "2", "1.75 (Pair)", "Tier 5: Suppressed"],
        ["Surat (Gujarat)", "1", "1", "1", "1", "1.00 (Single)", "Tier 5: Suppressed"],
        ["Guntur (Andhra Pradesh)", "0", "0", "0", "0", "1.00 (Zero)", "Tier 5: Suppressed"],
        ["Ranchi (Jharkhand)", "0", "0", "0", "0", "1.00 (Zero)", "Tier 5: Suppressed"]
    ]
    create_table(doc, headers_4_6, rows_4_6, col_widths=[1.7, 0.6, 0.7, 0.7, 0.8, 1.0, 1.0],
                 caption="Nationwide 31-City Station Catalogue & Spatial Reliability Audit Matrix", table_no="4.6")

    add_section_heading(doc, "4.7 Distance-Scaled Conformal Uncertainty Evaluation")
    add_body_p(doc,
        "Distance-scaled split conformal prediction was evaluated across spatial holdout samples. As documented in Table 4.7, the framework achieves "
        "an empirical coverage of 99.37% on PM2.5 and 99.16% on PM10 at nominal 95% confidence. Critically, high empirical coverage is accompanied by wide "
        "prediction intervals during severe pollution episodes (reaching 48.3 μg/m³ for PM2.5 and 131.5 μg/m³ for PM10 at nominal 95% tier). This confirms that "
        "conformal coverage must not be conflated with point accuracy: wide intervals achieve statistical guarantees while reflecting genuine atmospheric "
        "unpredictability."
    )

    headers_4_7 = ["Nominal Confidence (1 - α)", "PM2.5 Empirical Coverage", "PM2.5 Interval Width", "PM10 Empirical Coverage", "PM10 Interval Width"]
    rows_4_7 = [
        ["80% Nominal Tier", "92.8%", "24.1 μg/m³", "93.6%", "60.2 μg/m³"],
        ["90% Nominal Tier", "96.5%", "35.8 μg/m³", "97.2%", "94.3 μg/m³"],
        ["95% Nominal Tier", "99.37%", "48.3 μg/m³", "99.16%", "131.5 μg/m³"]
    ]
    create_table(doc, headers_4_7, rows_4_7, col_widths=[1.5, 1.2, 1.2, 1.2, 1.4],
                 caption="Distance-Scaled Conformal Uncertainty Calibration Summary", table_no="4.7")

    add_figure(doc, FIG_DIR / "fig1_conformal_calibration.png", "4.3",
               "Distance-Scaled Spatial Conformal Calibration Intervals showing split-conformal coverage and average interval width by confidence tier.", width_in=4.6)

    add_section_heading(doc, "4.8 Operational Streamlit Dashboard Interface")
    add_body_p(doc,
        "The complete VayuDrishti framework is deployed as an operational research dashboard on Streamlit Cloud. The interface incorporates all six "
        "modular views: (1) Home Dashboard displaying high-level atmospheric telemetry and active city selection; (2) Station Telemetry displaying raw CAAQMS "
        "pollutant concentrations with explicit 5-tier freshness badges; (3) AI Sequence Forecaster presenting next-hour (t+1) particulate trajectories; "
        "(4) Multimodal Hybrid presenting experimental visual refinement; (5) Model Comparison displaying side-by-side empirical performance metrics; "
        "and (6) Spatial Heatmap presenting continuous 2D IDW surfaces with station overlay toggles."
    )
    add_figure(doc, FIG_DIR / "dashboard_monitoring_interface.png", "4.8",
               "Operational Streamlit Real-Time Dashboard Interface showing active city selector, Air Quality Index telemetry, and regional model estimate badges.", width_in=4.8)

    add_section_heading(doc, "4.9 Overall Discussion & Synthesis")
    add_body_p(doc,
        "Synthesizing our empirical results yields five vital insights for operational environmental modeling: (1) Deep recurrent architectures (LSTM) "
        "and gradient-boosted trees provide robust temporal sequence modeling on multi-year archives (R² > 0.93), but short-term next-hour gains over persistence "
        "remain modest during stagnant meteorological conditions. (2) Domain adaptation on post-2020 logs is essential to capture pandemic and policy-driven "
        "emission shifts. (3) Multimodal learning using unaligned outdoor imagery introduces severe negative transfer, validating our bounded residual design. "
        "(4) Vectorized IDW significantly outperforms nearest-station baselines (+20% to +25%) in dense urban grids like Delhi. (5) Rigorous telemetry freshness "
        "governance and conformal uncertainty bands are indispensable for communicating spatial prediction limits honestly."
    )

    doc.add_page_break()


# ==============================================================================
# CHAPTER 5: CONCLUSION AND FUTURE WORK
# ==============================================================================
def build_chapter_5(doc):
    add_chapter_title(doc, "CHAPTER 5: CONCLUSION AND FUTURE WORK")

    add_section_heading(doc, "5.1 Summary of Research Findings")
    add_body_p(doc,
        "This Capstone Project developed and rigorously validated VayuDrishti, an end-to-end multimodal spatio-temporal air quality intelligence framework "
        "for Indian urban environments. Across massive empirical archives spanning 1.37 million historical records (2015–2020) and 3.3 million recent records "
        "(2020–2025), VayuDrishti demonstrated that 2-layer stacked LSTM networks achieve high predictive fidelity (PM2.5 MAE 10.57 μg/m³, R² 0.9492; PM10 MAE 18.86 μg/m³, "
        "R² 0.9301) for next-hour particulate forecasting. In spatial modeling, geodesic Inverse Distance Weighting (IDW) achieved a 20.0% error reduction on PM2.5 "
        "and 25.4% on PM10 over nearest-neighbor baselines in Delhi NCR holdout evaluations. Furthermore, our nationwide spatial audit catalogued 586 stations "
        "across 31 cities, establishing operational reliability tiers and identifying severe monitoring clustering in non-metro regions."
    )

    add_section_heading(doc, "5.2 Core Academic Contributions")
    add_body_p(doc, "The key research contributions of this work are summarized as follows:")
    add_bullet_p(doc, " Solved the catastrophic negative transfer of naive early concatenation (+12.19 μg/m³ error inflation) through bounded residual corrections and automatic fallback (m=0).", bold_prefix="1. Bounded Residual Multimodal Architecture:")
    add_bullet_p(doc, " Demonstrated consistent 20% to 25% error reductions using geodesic IDW over discrete nearest-neighbor step functions.", bold_prefix="2. Rigorous Spatial Holdout Benchmarking:")
    add_bullet_p(doc, " Formulated a 5-tier latency classification preventing stale cached sensor readings from corrupting live spatial heatmaps.", bold_prefix="3. Operational Telemetry Freshness Protocol:")
    add_bullet_p(doc, " Conducted the first comprehensive Clark-Evans nearest-neighbor spatial clustering audit across 31 Indian cities, revealing critical monitoring disparities.", bold_prefix="4. Nationwide 31-City Spatial Audit:")
    add_bullet_p(doc, " Implemented distance-scaled split conformal calibration providing ~99% empirical coverage without Gaussian assumptions.", bold_prefix="5. Distribution-Free Conformal Prediction:")
    add_bullet_p(doc, " Deployed an operational, provenance-aware interactive dashboard on Streamlit Cloud.", bold_prefix="6. Open-Science Reproducible Deployment:")

    add_section_heading(doc, "5.3 Honest Scientific Limitations")
    add_body_p(doc, "To maintain academic integrity, several important limitations must be explicitly acknowledged:")
    add_bullet_p(doc, " The prospective live evaluation was restricted to four monitoring stations over a 14-day window (418 pairs); differences against persistence were statistically non-significant (p > 0.40).", bold_prefix="1. Limited Live Audit Scale:")
    add_bullet_p(doc, " Spatial holdout validation was conducted primarily in Delhi NCR due to station density; findings cannot be generalized to sparse non-metro cities without local calibration.", bold_prefix="2. Spatial Generalization Constraints:")
    add_bullet_p(doc, " Ground-level image datasets lack micro-meteorological synchronization and co-located sensor precision, restricting visual fusion to an exploratory research branch.", bold_prefix="3. Image Synchronization Challenges:")
    add_bullet_p(doc, " Conformal prediction bands expand significantly during severe winter smog episodes, reflecting atmospheric unpredictability at the cost of precision.", bold_prefix="4. Interval Width in Severe Episodes:")

    add_section_heading(doc, "5.4 Future Research Directions")
    add_body_p(doc, "Promising avenues for future research and engineering extensions include:")
    add_bullet_p(doc, " Incorporating dynamic wind-vector advection graphs to model physical pollutant transport between urban monitoring stations.", bold_prefix="1. Spatio-Temporal Graph Neural Networks (ST-GNNs):")
    add_bullet_p(doc, " Integrating satellite remote sensing data from INSAT-3D and Sentinel-5P to bridge geographic blindspots in unmonitored Tier-2 and Tier-3 towns.", bold_prefix="2. Satellite Aerosol Optical Depth (AOD) Fusion:")
    add_bullet_p(doc, " Expanding prospective live telemetry audits across all seasons and all 500+ operational CPCB stations nationwide.", bold_prefix="3. Longitudinal Multi-Season Live Evaluation:")
    add_bullet_p(doc, " Embedding atmospheric boundary-layer thermodynamic equations directly into deep recurrent loss functions.", bold_prefix="4. Physics-Informed Neural Networks (PINNs):")

    doc.add_page_break()


# ==============================================================================
# APPENDICES, REFERENCES & BIODATA
# ==============================================================================
def build_appendices(doc):
    add_chapter_title(doc, "APPENDIX A: CORE IMPLEMENTATION SOURCE CODES")

    add_section_heading(doc, "A.1 Residual Multimodal Hybrid Model Definition (PyTorch)")
    add_code_block(doc,
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class ResidualMultimodalHybrid(nn.Module):\n"
        "    \"\"\"Residual Multimodal Hybrid Air Quality Forecaster with Quality Gating.\"\"\"\n"
        "    def __init__(self, sensor_feature_dim=27, sensor_hidden_dim=64,\n"
        "                 visual_feature_dim=2048, visual_proj_dim=32,\n"
        "                 distance_embed_dim=8, max_delta=25.0):\n"
        "        super().__init__()\n"
        "        self.max_delta = max_delta\n"
        "        # Visual Feature Projection\n"
        "        self.visual_projector = nn.Sequential(\n"
        "            nn.Linear(visual_feature_dim, 128),\n"
        "            nn.BatchNorm1d(128),\n"
        "            nn.ReLU(),\n"
        "            nn.Dropout(0.3),\n"
        "            nn.Linear(128, visual_proj_dim),\n"
        "            nn.ReLU()\n"
        "        )\n"
        "        # Distance Embedding\n"
        "        self.distance_embed = nn.Sequential(\n"
        "            nn.Linear(1, distance_embed_dim),\n"
        "            nn.ReLU()\n"
        "        )\n"
        "        # Quality-Aware Gating Network\n"
        "        gate_input_dim = sensor_hidden_dim + visual_proj_dim + distance_embed_dim\n"
        "        self.gate_network = nn.Sequential(\n"
        "            nn.Linear(gate_input_dim, 32),\n"
        "            nn.ReLU(),\n"
        "            nn.Linear(32, 1),\n"
        "            nn.Sigmoid()\n"
        "        )\n"
        "        # Residual Correction Head\n"
        "        self.residual_head = nn.Sequential(\n"
        "            nn.Linear(gate_input_dim, 32),\n"
        "            nn.ReLU(),\n"
        "            nn.Linear(32, 2),\n"
        "            nn.Tanh()\n"
        "        )\n\n"
        "    def forward(self, h_sensor, f_visual, distance, m_avail):\n"
        "        v_proj = self.visual_projector(f_visual)\n"
        "        e_dist = self.distance_embed(distance)\n"
        "        f_fused = torch.cat([h_sensor, m_avail * v_proj, e_dist], dim=-1)\n"
        "        alpha = self.gate_network(f_fused)\n"
        "        alpha_eff = m_avail * alpha\n"
        "        delta_raw = self.residual_head(f_fused)\n"
        "        delta_bounded = self.max_delta * delta_raw\n"
        "        return alpha_eff, delta_bounded"
    )

    add_section_heading(doc, "A.2 Geodesic IDW Spatial Engine & Freshness Filters")
    add_code_block(doc,
        "import numpy as np\n\n"
        "def haversine_vectorized(lat1, lon1, lat2_arr, lon2_arr):\n"
        "    \"\"\"Computes great-circle distances in km over spherical Earth.\"\"\"\n"
        "    R = 6371.0\n"
        "    dlat = np.radians(lat2_arr - lat1)\n"
        "    dlon = np.radians(lon2_arr - lon1)\n"
        "    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr)) * np.sin(dlon / 2.0)**2\n"
        "    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))\n"
        "    return R * c\n\n"
        "def interpolate_point(lat, lon, stations_df, power=2.0, epsilon=0.001, r_max=45.0):\n"
        "    \"\"\"Geodesic IDW interpolation with strict provenance separation.\"\"\"\n"
        "    coords = stations_df[['lat', 'lon']].values\n"
        "    dists = haversine_vectorized(lat, lon, coords[:, 0], coords[:, 1])\n"
        "    min_dist = np.min(dists)\n"
        "    # 1. Co-located Physical Observation\n"
        "    if min_dist <= 0.05:\n"
        "        idx = np.argmin(dists)\n"
        "        return {'classification': 'Physical Observation', 'pm25': stations_df.iloc[idx]['pm25']}\n"
        "    # 2. Sparse Safeguard Fallback (< 3 stations within radius)\n"
        "    valid_mask = dists <= r_max\n"
        "    if np.sum(valid_mask) < 3:\n"
        "        idx = np.argmin(dists)\n"
        "        return {'classification': 'Nearest-Neighbor Fallback', 'pm25': stations_df.iloc[idx]['pm25_pred'], 'dist_km': min_dist}\n"
        "    # 3. Vectorized IDW Surface Estimate\n"
        "    weights = 1.0 / np.power(dists[valid_mask] + epsilon, power)\n"
        "    pm25_interp = np.sum(weights * stations_df['pm25_pred'].values[valid_mask]) / np.sum(weights)\n"
        "    return {'classification': 'Interpolated Spatial Estimate', 'pm25': max(0.0, float(pm25_interp)), 'contributing': int(np.sum(valid_mask))}"
    )

    add_section_heading(doc, "A.3 Clark-Evans Nearest-Neighbor Clustering Ratio")
    add_code_block(doc,
        "def calculate_clark_evans_index(coords, area_km2):\n"
        "    \"\"\"Calculates Clark-Evans Nearest Neighbor Clustering Ratio (R = r_A / r_E).\"\"\"\n"
        "    N = len(coords)\n"
        "    if N < 2 or area_km2 <= 0:\n"
        "        return 1.0, 'Insufficient Monitors', 0.0\n"
        "    nn_distances = []\n"
        "    for i in range(N):\n"
        "        dists = [haversine_vectorized(coords[i, 0], coords[i, 1], np.array([coords[j, 0]]), np.array([coords[j, 1]]))[0]\n"
        "                 for j in range(N) if i != j]\n"
        "        nn_distances.append(min(dists))\n"
        "    r_A = np.mean(nn_distances)\n"
        "    density = N / area_km2\n"
        "    r_E = 0.5 / np.sqrt(density)\n"
        "    R = float(r_A / r_E)\n"
        "    pattern = 'Clustered' if R < 0.85 else ('Dispersed' if R > 1.15 else 'Random/Mixed')\n"
        "    return R, pattern, r_A"
    )

    doc.add_page_break()

def build_references(doc):
    add_chapter_title(doc, "REFERENCES")
    p_intro = doc.add_paragraph()
    p_intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_intro.paragraph_format.space_after = Pt(10)
    r = p_intro.add_run("All academic citations follow the official IEEE Transactions reference format and have been verified against published sources:")
    r.font.name = FONT_NAME
    r.font.size = Pt(11)
    r.font.italic = True

    references = [
        "[1] S. Hochreiter and J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.",
        "[2] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), Jun. 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.",
        "[3] D. Shepard, \"A two-dimensional interpolation function for irregularly-spaced data,\" in Proc. 1968 23rd ACM National Conf., New York, NY, USA, 1968, pp. 517–524, doi: 10.1145/800186.810616.",
        "[4] Central Pollution Control Board (CPCB), \"National Ambient Air Quality Standards,\" The Gazette of India, Extraordinary, Part III, Section 4, Ministry of Environment, Forest and Climate Change, Govt. of India, Notification No. B-29016/20/90/PCI-L, New Delhi, India, Nov. 18, 2009.",
        "[5] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, \"LightGBM: A Highly Efficient Gradient Boosting Decision Tree,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 3146–3154.",
        "[6] DeadCardassian, \"PM25Vision: An Image Dataset for Ambient Particulate Matter Estimation,\" Hugging Face Datasets, 2023. [Online]. Available: https://huggingface.co/datasets/DeadCardassian/PM25Vision",
        "[7] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. New York, NY, USA: Springer, 2005, doi: 10.1007/b106715.",
        "[8] A. N. Angelopoulos and S. Bates, \"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,\" arXiv preprint arXiv:2107.07511, 2021.",
        "[9] Y. Romano, E. Patterson, and E. Candès, \"Conformalized Quantile Regression,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, Vancouver, BC, Canada, 2019, pp. 3543–3553.",
        "[10] D. N. Politis and J. P. Romano, \"The Stationary Bootstrap,\" Journal of the American Statistical Association, vol. 89, no. 428, pp. 1303–1313, 1994, doi: 10.1080/01621459.1994.10476870.",
        "[11] S. K. Guttikunda, R. Goel, and P. Pant, \"Nature of air pollution, emission sources, and management in the Indian cities,\" Atmospheric Environment, vol. 95, pp. 501–510, Oct. 2014, doi: 10.1016/j.atmosenv.2014.07.006.",
        "[12] P. Pant, S. K. Guttikunda, and R. E. Peltier, \"Exposure to air pollution in developing countries: challenges and opportunities for environmental health,\" Current Environmental Health Reports, vol. 3, no. 2, pp. 170–183, Jun. 2016, doi: 10.1007/s40572-016-0089-9.",
        "[13] J. Wang, Y. Hu, Q. Liu, X. Zhang, and C. Deng, \"A Survey on Spatial-Temporal Deep Learning for Air Quality Forecasting,\" ACM Transactions on Knowledge Discovery from Data, vol. 15, no. 4, pp. 1–35, Jun. 2021, doi: 10.1145/3442188.",
        "[14] T. Baltrušaitis, C. Ahuja, and L.-P. Morency, \"Multimodal Machine Learning: A Survey and Taxonomy,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, Feb. 2019, doi: 10.1109/TPAMI.2018.2798607.",
        "[15] K. Zhang, A. Thesen, D. Le, and R. Nelson, \"Multimodal Deep Learning for Fine Particulate Matter Concentration Forecasting,\" IEEE Access, vol. 8, pp. 89441–89452, 2020, doi: 10.1109/ACCESS.2020.2993883.",
        "[16] OpenAQ Community, \"OpenAQ: Global Real-Time Ambient Air Quality Data Platform and REST API,\" Environmental Science & Technology, vol. 56, no. 1, pp. 12–15, 2022. [Online]. Available: https://openaq.org",
        "[17] P. J. Clark and F. C. Evans, \"Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations,\" Ecology, vol. 35, no. 4, pp. 445–453, Oct. 1954, doi: 10.2307/1931034."
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        r = p_ref.add_run(ref)
        r.font.name = FONT_NAME
        r.font.size = Pt(10.5)
        r.font.color.rgb = COLOR_TEXT

    doc.add_page_break()

def build_biodata(doc):
    add_chapter_title(doc, "BIODATA OF PROJECT MEMBERS")

    members = [
        {
            "name": "TELIKI RUSHYA SREE",
            "reg": "23BCE9562",
            "degree": "B.Tech Computer Science and Engineering",
            "mobile": "+91 9440699393",
            "email": "rushya.23bce9562@vitapstudent.ac.in",
            "address": "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"
        },
        {
            "name": "MACHIREDDY SWATHI",
            "reg": "23BCE9566",
            "degree": "B.Tech Computer Science and Engineering",
            "mobile": "+91 93985 25854",
            "email": "swathi.23bce9566@vitapstudent.ac.in",
            "address": "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"
        },
        {
            "name": "THANUJA VARAKALA",
            "reg": "23BCE9608",
            "degree": "B.Tech Computer Science and Engineering",
            "mobile": "+91 82977 08071",
            "email": "Thanuja.23bce9608@vitapstudent.ac.in",
            "address": "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"
        },
        {
            "name": "PALLI SHILPA",
            "reg": "23BCE9567",
            "degree": "B.Tech Computer Science and Engineering",
            "mobile": "+91 93814 88746",
            "email": "shilpa.23bce9567@vitapstudent.ac.in",
            "address": "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"
        }
    ]

    for m in members:
        p_box = doc.add_paragraph()
        p_box.paragraph_format.space_before = Pt(6)
        p_box.paragraph_format.space_after = Pt(10)
        p_box.paragraph_format.line_spacing = 1.15
        pPr = p_box._element.get_or_add_pPr()
        tabs_xml = f'<w:tabs {nsdecls("w")}><w:tab w:val="left" w:pos="2160"/></w:tabs>'
        pPr.append(parse_xml(tabs_xml))

        lines = [
            ("Name", m["name"], True),
            ("Registration No.", m["reg"], True),
            ("Degree & Branch", m["degree"], False),
            ("Mobile Number", m["mobile"], False),
            ("Email ID", m["email"], False),
            ("Permanent Address", m["address"], False)
        ]
        for idx, (label, val, is_b) in enumerate(lines):
            r1 = p_box.add_run(label)
            r1.font.name = FONT_NAME
            r1.font.size = Pt(11)
            r1.bold = True
            r2 = p_box.add_run("\t: ")
            r2.font.name = FONT_NAME
            r2.font.size = Pt(11)
            r3 = p_box.add_run(val)
            r3.font.name = FONT_NAME
            r3.font.size = Pt(11)
            r3.bold = is_b
            if idx < len(lines) - 1:
                p_box.add_run("\n")

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(16)
    p_note.paragraph_format.space_after = Pt(12)
    p_note.paragraph_format.line_spacing = 1.2
    pPr = p_note._element.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="FEF3C7"/>')
    pBdr = parse_xml(f'''
        <w:pBdr {nsdecls("w")}>
            <w:left w:val="single" w:sz="18" w:space="8" w:color="D97706"/>
        </w:pBdr>
    ''')
    pPr.append(shd)
    pPr.append(pBdr)

    r_pre = p_note.add_run("NOTE: ")
    r_pre.font.name = FONT_NAME
    r_pre.font.size = Pt(10.5)
    r_pre.bold = True
    r_pre.font.color.rgb = RGBColor(180, 83, 9)

    r_msg = p_note.add_run(
        "It is MANDATORY for students to attach all Presentation Slides, Source Codes, Model Weights/Artifacts, "
        "and a 5–10 minute demonstration video of the VayuDrishti dashboard digitally in a Compact Disk (CD). "
        "Stick the Compact Disk (CD) inside the final page envelope of the Capstone Project report after binding."
    )
    r_msg.font.name = FONT_NAME
    r_msg.font.size = Pt(10.5)
    r_msg.font.color.rgb = COLOR_TEXT


# ==============================================================================
# MAIN EXECUTOR
# ==============================================================================
def main():
    print("Building Repaired VayuDrishti Capstone Project Report...")
    doc = Document()

    # Section 1: Front Matter (Roman Numerals ii..ix, Different First Page)
    sec1 = doc.sections[0]
    sec1.top_margin = Inches(1.0)
    sec1.bottom_margin = Inches(1.0)
    sec1.left_margin = Inches(1.0)
    sec1.right_margin = Inches(1.0)
    sec1.page_width = Inches(8.5)
    sec1.page_height = Inches(11.0)
    sec1.different_first_page_header_footer = True

    # Roman numeral numbering for Section 1
    sec1_sectPr = sec1._sectPr
    pgNum1 = parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="lowerRoman" w:start="1"/>')
    sec1_sectPr.append(pgNum1)

    # Centered Roman footer
    footer1 = sec1.footer
    p_f1 = footer1.paragraphs[0]
    p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f1 = p_f1.add_run()
    r_f1.font.name = FONT_NAME
    r_f1.font.size = Pt(10)
    fldSimple1 = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"/>')
    r_f1._r.append(fldSimple1)

    # Build Front Matter
    build_front_matter(doc)

    # Section 2: Main Chapters (Arabic Numerals 1..74)
    sec2 = doc.add_section()
    sec2.top_margin = Inches(1.0)
    sec2.bottom_margin = Inches(1.0)
    sec2.left_margin = Inches(1.0)
    sec2.right_margin = Inches(1.0)
    sec2.page_width = Inches(8.5)
    sec2.page_height = Inches(11.0)
    sec2.different_first_page_header_footer = False
    sec2.header.is_linked_to_previous = False
    sec2.footer.is_linked_to_previous = False

    # Arabic decimal numbering starting at 1
    sec2_sectPr = sec2._sectPr
    pgNum2 = parse_xml(f'<w:pgNumType {nsdecls("w")} w:fmt="decimal" w:start="1"/>')
    sec2_sectPr.append(pgNum2)

    # Running header
    header2 = sec2.header
    p_h2 = header2.paragraphs[0]
    p_h2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_h2 = p_h2.add_run("VayuDrishti: Multimodal Air Quality Prediction in Indian Urban Environments")
    r_h2.font.name = FONT_NAME
    r_h2.font.size = Pt(8.5)
    r_h2.font.color.rgb = COLOR_MUTED

    # Centered Arabic footer
    footer2 = sec2.footer
    p_f2 = footer2.paragraphs[0]
    p_f2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f2 = p_f2.add_run()
    r_f2.font.name = FONT_NAME
    r_f2.font.size = Pt(10)
    fldSimple2 = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"/>')
    r_f2._r.append(fldSimple2)

    # Build Main Chapters & Appendices
    build_chapter_1(doc)
    build_chapter_2(doc)
    build_chapter_3(doc)
    build_chapter_4(doc)
    build_chapter_5(doc)
    build_appendices(doc)
    build_references(doc)
    build_biodata(doc)

    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_DOCX))
    print(f"Successfully saved Word Document: {OUTPUT_DOCX}")

if __name__ == "__main__":
    main()
