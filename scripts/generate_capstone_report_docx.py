"""
scripts/generate_capstone_report_docx.py

Generates the official VIT-AP University Capstone Project Report Word Document (.docx)
and compiles it to PDF (.pdf) for VayuDrishti.

Follows the exact institutional structure:
- Title Page
- Certificate Page (Guide, Examiners, Program Chair, Dean)
- Acknowledgements
- Abstract
- Table of Contents
- List of Tables & List of Figures
- Chapter 1: Introduction (Objectives, Literature Survey, Report Organization)
- Chapter 2: System Architecture & Methodology (Hardware, Software, Models, Standards)
- Chapter 3: Cost & Resource Analysis
- Chapter 4: Results and Discussion (Benchmarks, Conformal Calibration, Spatial Audit across 31 Cities)
- Chapter 5: Conclusion & Future Works
- Chapter 6: Appendix (Key Source Codes)
- Chapter 7: References
- Biodata of Project Members
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DOCX = ROOT / "docs/VayuDrishti_Capstone_Project_Report.docx"
OUTPUT_PDF = ROOT / "docs/VayuDrishti_Capstone_Project_Report.pdf"
FIG_DIR = ROOT / "paper/figures"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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

def add_title(doc, text, size=18, bold=True, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(15, 23, 42)
    return p

def add_heading_1(doc, text, space_before=16, space_after=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(15, 23, 42)
    return p

def add_heading_2(doc, text, space_before=12, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_heading_3(doc, text, space_before=8, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11.5)
    run.font.color.rgb = RGBColor(51, 65, 85)
    return p

def add_body_p(doc, text, indent=True, bold_prefix="", space_after=6, line_spacing=1.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = line_spacing
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.3)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(11)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_bullet_p(doc, text, bold_prefix=""):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Times New Roman"
        r_pre.font.size = Pt(11)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_figure(doc, img_path, fig_no, caption, width_in=5.8):
    if not os.path.exists(img_path):
        print(f"Warning: Figure {img_path} not found.")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=Inches(width_in))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    r_no = p_cap.add_run(f"Figure {fig_no}: ")
    r_no.bold = True
    r_no.font.name = "Times New Roman"
    r_no.font.size = Pt(10)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(10)
    r_cap.italic = True

def add_code_block(doc, code_text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(code_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(30, 41, 59)

def generate_capstone_report():
    print("Building VIT-AP Capstone Project Report Word Document...")
    doc = Document()

    # Standard 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.2)
        s.right_margin = Inches(1.0)

    # --------------------------------------------------------------------------
    # 1. TITLE PAGE
    # --------------------------------------------------------------------------
    add_title(doc, "VAYUDRISHTI: A RESIDUAL MULTIMODAL HYBRID FRAMEWORK FOR AIR QUALITY FORECASTING", size=16, bold=True, space_after=14)
    add_title(doc, "A CAPSTONE PROJECT REPORT", size=13, bold=True, space_after=10)
    add_title(doc, "Submitted in partial fulfillment of the requirement for the award of the Degree of", size=11, bold=False, space_after=4)
    add_title(doc, "BACHELOR OF TECHNOLOGY", size=13, bold=True, space_after=2)
    add_title(doc, "IN", size=11, bold=True, space_after=2)
    add_title(doc, "COMPUTER SCIENCE AND ENGINEERING", size=13, bold=True, space_after=12)
    add_title(doc, "by", size=11, bold=False, space_after=6)

    # Authors Block
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.line_spacing = 1.2
    p_auth.paragraph_format.space_after = Pt(14)
    authors = [
        ("TELIKI RUSHYA SREE", "23BCE9562"),
        ("MACHIREDDY SWATHI", "23BCE9566"),
        ("THANUJA VARAKALA", "23BCE9608"),
        ("PALLI SHILPA", "23BCE9567")
    ]
    for name, reg in authors:
        r1 = p_auth.add_run(f"{name} ({reg})\n")
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(11)

    add_title(doc, "Under the Guidance of", size=11, bold=False, space_after=2)
    add_title(doc, "PROF. THOTA RAMATHULSI", size=13, bold=True, space_after=2)
    add_title(doc, "Department of SCOPE", size=11, bold=False, space_after=16)

    add_title(doc, "SCHOOL OF COMPUTER SCIENCE AND ENGINEERING (SCOPE)", size=12, bold=True, space_after=2)
    add_title(doc, "VIT-AP UNIVERSITY", size=14, bold=True, space_after=2)
    add_title(doc, "AMARAVATI - 522237, ANDHRA PRADESH, INDIA", size=11, bold=False, space_after=4)
    add_title(doc, "NOVEMBER 2026", size=12, bold=True, space_after=0)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 2. CERTIFICATE PAGE
    # --------------------------------------------------------------------------
    add_title(doc, "CERTIFICATE", size=16, bold=True, space_after=18)
    
    add_body_p(doc, 
        "This is to certify that the Capstone Project report titled “VAYUDRISHTI: A RESIDUAL MULTIMODAL HYBRID FRAMEWORK FOR AIR QUALITY FORECASTING” "
        "that is being submitted by TELIKI RUSHYA SREE (23BCE9562), MACHIREDDY SWATHI (23BCE9566), THANUJA VARAKALA (23BCE9608), and PALLI SHILPA (23BCE9567) "
        "in partial fulfillment of the requirements for the award of Bachelor of Technology in Computer Science and Engineering, is a record of bonafide work done under my guidance. "
        "The contents of this Project report, in full or in parts, have neither been taken from any other source nor have been submitted to any other Institute or University for award of any degree or diploma and the same is certified.",
        indent=True, line_spacing=1.5
    )

    p_guide = doc.add_paragraph()
    p_guide.paragraph_format.space_before = Pt(36)
    p_guide.paragraph_format.space_after = Pt(24)
    r_g1 = p_guide.add_run("Prof. Thota Ramathulsi\n")
    r_g1.bold = True
    r_g1.font.name = "Times New Roman"
    r_g1.font.size = Pt(11)
    r_g2 = p_guide.add_run("Project Guide\nDepartment of SCOPE, VIT-AP University")
    r_g2.font.name = "Times New Roman"
    r_g2.font.size = Pt(11)

    p_eval = doc.add_paragraph()
    p_eval.paragraph_format.space_before = Pt(12)
    p_eval.paragraph_format.space_after = Pt(28)
    r_ev = p_eval.add_run("The thesis is satisfactory / unsatisfactory\n\n\n\nInternal Examiner                                                          External Examiner")
    r_ev.font.name = "Times New Roman"
    r_ev.font.size = Pt(11)

    p_app = doc.add_paragraph()
    p_app.paragraph_format.space_before = Pt(20)
    p_app.paragraph_format.space_after = Pt(0)
    r_ap = p_app.add_run("Approved by:\n\n\n\nPROGRAM CHAIR                                                     DEAN\nB. Tech. CSE                                                           School Of Computer Science and Engineering")
    r_ap.bold = True
    r_ap.font.name = "Times New Roman"
    r_ap.font.size = Pt(11)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 3. ACKNOWLEDGEMENTS
    # --------------------------------------------------------------------------
    add_title(doc, "ACKNOWLEDGEMENTS", size=16, bold=True, space_after=16)

    add_body_p(doc, "We express our sincere gratitude and profound respect to our project guide, Prof. Thota Ramathulsi, Department of SCOPE, for her invaluable guidance, constructive critique, and constant encouragement throughout the formulation and execution of this Capstone Project. Her scientific rigor and keen insights into environmental modeling and deep learning architectures have played a pivotal role in shaping this research.")
    add_body_p(doc, "We extend our deep appreciation to the Management of VIT-AP University, the Vice-Chancellor, the Registrar, and the Dean of the School of Computer Science and Engineering (SCOPE) for providing state-of-the-art computational infrastructure, laboratory facilities, and an encouraging research environment.")
    add_body_p(doc, "We would also like to thank the Program Chair, B.Tech Computer Science and Engineering, faculty members, and technical staff of the School of Computer Science and Engineering for their direct and indirect support throughout the duration of this work.")
    add_body_p(doc, "Finally, we express our heartfelt thanks to our parents, family members, and friends whose unconditional love, patience, and encouragement have been a perpetual source of strength during our academic journey.")

    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.space_before = Pt(24)
    r_sig = p_sig.add_run("TELIKI RUSHYA SREE (23BCE9562)\nMACHIREDDY SWATHI (23BCE9566)\nTHANUJA VARAKALA (23BCE9608)\nPALLI SHILPA (23BCE9567)")
    r_sig.bold = True
    r_sig.font.name = "Times New Roman"
    r_sig.font.size = Pt(11)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 4. ABSTRACT
    # --------------------------------------------------------------------------
    add_title(doc, "ABSTRACT", size=16, bold=True, space_after=16)

    add_body_p(doc, "Ambient particulate matter pollution (PM2.5 and PM10) across developing urban agglomerations poses an acute hazard to public health and economic productivity. However, operational forecasting systems face frequent sensor dropouts, non-linear meteorological volatility, optical ambiguity in outdoor surveillance imagery, and spatial sensor sparsity. This Capstone Project presents VayuDrishti, an operational residual multimodal hybrid framework for next-hour particulate matter forecasting and nationwide spatial reliability auditing.")
    add_body_p(doc, "Rather than relying on unconstrained joint feature concatenation, VayuDrishti structures visual features as a bounded residual modifier over a verified recurrent sensor baseline (2-layer LSTM), regulated by a learned quality-aware gating mechanism with guaranteed mathematical fallback when imagery is corrupted or unavailable (m = 0). Across 404,431 continuous hourly sequences from 90 continuous stations, the sensor baseline achieves a Mean Absolute Error (MAE) of 10.57 ug/m3 (R2 = 0.9492) for PM2.5 and 18.86 ug/m3 (R2 = 0.9301) for PM10. To extend predictions across unmonitored coordinates, an integrated geostatistical spatial prediction layer using modified Geodesic Inverse Distance Weighting achieves a 15.71% (PM2.5) and 19.71% (PM10) MAE reduction over nearest-neighbor baselines across 1,564 Leave-One-Station-Out evaluations (block bootstrap p < 0.001).")
    add_body_p(doc, "Furthermore, distance-scaled split-conformal calibration delivers finite-sample coverage guarantees (>= 99.1% empirical coverage across urban distance bands). The project expands spatial modeling across 31 Indian urban agglomerations indexing 586 physical stations nationwide, conducts an empirical spatial reliability audit using the Clark-Evans nearest neighbor ratio (R) to diagnose urban core clustering (R = 0.32-0.61 in major metros), and enforces a 5-tier operational reliability framework with automated safeguards suppressing continuous surfaces in sensor-sparse environments (N < 3). The complete framework is operationalized through an interactive 6-view real-time Streamlit deployment architecture with audited forecast provenance.")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 5. TABLE OF CONTENTS
    # --------------------------------------------------------------------------
    add_title(doc, "TABLE OF CONTENTS", size=16, bold=True, space_after=16)

    toc_table = doc.add_table(rows=1, cols=4)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = toc_table.rows[0].cells
    hdr_cells[0].text = "S.No."
    hdr_cells[1].text = "Chapter"
    hdr_cells[2].text = "Title"
    hdr_cells[3].text = "Page No."
    for cell in hdr_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)

    toc_data = [
        ("1.", "", "Certificate", "ii"),
        ("2.", "", "Acknowledgements", "iii"),
        ("3.", "", "Abstract", "iv"),
        ("4.", "", "List of Tables", "vi"),
        ("5.", "", "List of Figures", "vii"),
        ("6.", "1", "Introduction\n1.1 Objectives\n1.2 Background and Literature Survey\n1.3 Organization of the Report", "1\n2\n3\n5"),
        ("7.", "2", "VayuDrishti Multimodal Forecasting System\n2.1 Proposed System Architecture\n2.2 Working Methodology & Dual Pathways\n2.3 Standards, Protocols, & Provenance Safeguards\n2.4 System Details\n   2.4.1 Software & Machine Learning Pipelines\n   2.4.2 Hardware & Cloud Deployment Infrastructure", "6\n6\n8\n10\n12\n12\n19"),
        ("8.", "3", "Computational & Deployment Cost Analysis\n3.1 Resource & Component Cost Breakdown\n3.2 Operational Cloud & Latency Overhead", "22\n22\n23"),
        ("9.", "4", "Results and Discussion\n4.1 Sensor Baseline LSTM & GBDT Ensemble Results\n4.2 Residual Multimodal Hybrid Evaluation\n4.3 Distribution-Free Conformal Uncertainty Calibration\n4.4 Spatial Prediction Layer & Temporal Bootstrapping\n4.5 31-City Spatial Reliability & Telemetry Audit\n4.6 Operational Streamlit Dashboard Interface", "25\n25\n28\n30\n32\n34\n39"),
        ("10.", "5", "Conclusion and Future Works\n5.1 Summary of Findings\n5.2 Future Directions", "42\n42\n43"),
        ("11.", "6", "Appendix: Core Implementation Codes", "45"),
        ("12.", "7", "References", "55"),
        ("13.", "", "Biodata of Project Members", "58"),
    ]

    for sno, chap, title, page in toc_data:
        row_cells = toc_table.add_row().cells
        row_cells[0].text = sno
        row_cells[1].text = chap
        row_cells[2].text = title
        row_cells[3].text = page
        for c in row_cells:
            set_cell_margins(c, top=60, bottom=60, left=80, right=80)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 6. LIST OF TABLES & LIST OF FIGURES
    # --------------------------------------------------------------------------
    add_title(doc, "LIST OF TABLES", size=14, bold=True, space_after=12)
    tables_list = [
        ("Table 2.1", "Audited 27-Dimensional Numerical Sensor Feature Schema", "13"),
        ("Table 3.1", "Computational Infrastructure & Operational Deployment Cost Analysis", "23"),
        ("Table 4.1", "Overall Historical Benchmark Performance on 404,431 Continuous Sequences", "26"),
        ("Table 4.2", "Paired Model Comparison on Synchronous Multimodal Cohort (N = 97)", "29"),
        ("Table 4.3", "Distribution-Free Conformal Calibration across Severity Regimes", "31"),
        ("Table 4.4", "Leave-One-Station-Out Spatial Prediction & Bootstrap Significance", "33"),
        ("Table 4.5", "Nationwide 31-City Station Infrastructure & Spatial Reliability Audit Matrix", "36"),
    ]
    t_tbl = doc.add_table(rows=1, cols=3)
    t_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_tbl.rows[0].cells[0].text = "Table No."
    t_tbl.rows[0].cells[1].text = "Title"
    t_tbl.rows[0].cells[2].text = "Page No."
    for cell in t_tbl.rows[0].cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "F1F5F9")
    for no, title, pg in tables_list:
        row = t_tbl.add_row().cells
        row[0].text = no
        row[1].text = title
        row[2].text = pg
        for c in row:
            set_cell_margins(c, top=50, bottom=50, left=60, right=60)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(18)

    add_title(doc, "LIST OF FIGURES", size=14, bold=True, space_after=12)
    figures_list = [
        ("Figure 2.1", "Complete VayuDrishti Multimodal Forecasting Architecture", "7"),
        ("Figure 2.2", "Continuous Telemetry Ingestion and 27-Dimensional Feature Pipeline", "14"),
        ("Figure 2.3", "Deep Recurrent 2-Layer LSTM Sequence Forecaster Architecture", "15"),
        ("Figure 2.4", "Optical Feature Extraction via Pretrained ResNet-50 Backbone", "16"),
        ("Figure 2.5", "Learned Quality-Aware Gating and Feature Fusion Mechanism", "17"),
        ("Figure 2.6", "Bounded Residual Multimodal Hybrid Model with Fallback Masking", "18"),
        ("Figure 2.7", "Independent Severity-Gated LSTM–GBDT Ensemble Architecture", "19"),
        ("Figure 4.1", "Ground Truth vs Predicted Particulate Concentrations (N = 404,431)", "27"),
        ("Figure 4.2", "Multi-Class Particulate Matter Air Quality Confusion Matrices", "28"),
        ("Figure 4.3", "Distance-Scaled Spatial Conformal Calibration Intervals", "31"),
        ("Figure 4.4", "Multi-Station Generalization across 15 Operational CAAQMS Stations", "32"),
        ("Figure 4.5", "Seasonal Model Dynamics and Particulate Regime Error Curves", "34"),
        ("Figure 4.6", "Spatial Advection Cross-Correlation in Urban Corridors", "35"),
        ("Figure 4.7", "Opportunistic Image Alignment Discrepancy Diagnostics (PM25Vision)", "38"),
        ("Figure 4.8", "Operational Streamlit Real-Time Dashboard Interface", "40"),
    ]
    f_tbl = doc.add_table(rows=1, cols=3)
    f_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    f_tbl.rows[0].cells[0].text = "Figure No."
    f_tbl.rows[0].cells[1].text = "Title"
    f_tbl.rows[0].cells[2].text = "Page No."
    for cell in f_tbl.rows[0].cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "F1F5F9")
    for no, title, pg in figures_list:
        row = f_tbl.add_row().cells
        row[0].text = no
        row[1].text = title
        row[2].text = pg
        for c in row:
            set_cell_margins(c, top=50, bottom=50, left=60, right=60)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 1: INTRODUCTION
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 1\nINTRODUCTION", size=16, bold=True, space_after=16)

    add_body_p(doc, "Ambient particulate matter pollution, specifically breathable fine particles (PM2.5, aerodynamic diameter <= 2.5 um) and coarse particles (PM10, <= 10 um), represents an environmental and public health crisis across rapidly industrializing developing nations. In India, metropolitan agglomerations spanning the Indo-Gangetic Plain and the peninsular plateau routinely endure catastrophic air pollution events during post-monsoon agricultural biomass burning and winter atmospheric temperature inversions. Providing municipal authorities, healthcare providers, and citizens with highly accurate, short-term (t + 1 hour) predictive warnings is essential for exposure mitigation, hospital surge readiness, and dynamic vehicular traffic intervention.")
    add_body_p(doc, "Traditional air pollution forecasting has historically depended on numerical chemical transport models (CTMs), including WRF-Chem and CMAQ. While grounded in the first principles of atmospheric physics and gas-phase chemical kinetics, CTMs require massive high-performance computing clusters, stationary emission inventories that become obsolete rapidly, and suffer from high operational turnaround latency. Consequently, data-driven machine learning models leveraging continuous ambient air quality monitoring station (CAAQMS) sensor streams have become the standard methodology for rapid operational forecasting.")

    add_heading_2(doc, "1.1 Objectives")
    add_body_p(doc, "The primary objectives of this Capstone Project are as follows:")
    add_bullet_p(doc, "Design and implement a robust recurrent deep learning baseline (2-layer LSTM) trained on continuous 27-dimensional telemetry to forecast next-hour PM2.5 and PM10 concentrations.", bold_prefix="1. Deep Recurrent Sensor Modeling: ")
    add_bullet_p(doc, "Formulate a novel Residual Multimodal Hybrid framework that fuses numerical telemetry with outdoor surveillance images as a bounded physical correction under learned quality-aware gating.", bold_prefix="2. Bounded Multimodal Residual Fusion: ")
    add_bullet_p(doc, "Implement a mathematically guaranteed fallback mechanism (m = 0 => alpha_eff = 0) ensuring that when imagery is corrupted, occluded, or missing, the hybrid model output identically collapses to the verified sensor baseline.", bold_prefix="3. Guaranteed Fallback Mechanism: ")
    add_bullet_p(doc, "Develop an independent severity-gated LSTM–GBDT ensemble that dynamically decouples gradient-boosted decision tree gains in moderate regimes from recurrent neural network handling during hazardous crisis regimes.", bold_prefix="4. Severity-Gated Ensemble: ")
    add_bullet_p(doc, "Incorporate distribution-free split-conformal prediction intervals delivering finite-sample coverage guarantees across complex urban pollution regimes.", bold_prefix="5. Conformal Uncertainty Quantification: ")
    add_bullet_p(doc, "Extend point forecasts to unmonitored coordinates using modified Geodesic Inverse Distance Weighting (IDW) across 31 registered Indian urban agglomerations, supported by automated safeguards that suppress continuous surfaces in sensor-sparse environments (N < 3).", bold_prefix="6. Multi-City Spatial Modeling & Safeguards: ")
    add_bullet_p(doc, "Conduct an empirical spatial reliability audit using the Clark-Evans nearest neighbor ratio (R) to quantify intra-urban clustering and establish a 5-tier operational reliability framework.", bold_prefix="7. Spatial Reliability Audit: ")
    add_bullet_p(doc, "Deploy a high-responsiveness, 6-view interactive dashboard with strict telemetry freshness state machines, 24-hour sequence completeness requirements, and zero-silent-substitution provenance guarantees.", bold_prefix="8. Production Deployment & Provenance: ")

    add_heading_2(doc, "1.2 Background and Literature Survey")
    add_body_p(doc, "Air quality prediction has evolved across three major technological epochs:")
    add_body_p(doc, "Early research utilized classical statistical time-series methods such as Autoregressive Integrated Moving Average (ARIMA) and Vector Autoregression (VAR). While computationally lightweight, these methods assume linear dependencies and Gaussian error structures, rendering them incapable of capturing abrupt non-linear transitions during pollution spikes. Shallow machine learning models, including Support Vector Regression (SVR) and Random Forests, demonstrated improved capacity to model non-linear pollutant interactions. More recently, Gradient Boosted Decision Trees (GBDT), particularly LightGBM, have proven exceptionally capable on tabular air quality data due to their histogram-based split optimization and native handling of missing trace gas channels.", bold_prefix="Sensor-Based Machine Learning: ")
    add_body_p(doc, "Because atmospheric dispersion exhibits temporal memory and diurnal cyclicity, Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRU), have become the dominant deep learning paradigm. LSTMs maintain internal memory cells regulated by input, forget, and output gates, effectively mitigating vanishing gradient phenomena over multi-hour lookback horizons.", bold_prefix="Recurrent Neural Networks: ")
    add_body_p(doc, "To complement sparse physical monitoring networks, recent works have explored computer vision techniques to estimate particulate concentrations from outdoor surveillance cameras and smartphone photographs. Particulate matter scatters and absorbs visible light according to Koschmieder's atmospheric attenuation law, degrading image contrast, shifting chromatic distributions, and attenuating high-frequency edges. Deep Convolutional Neural Networks (CNNs), notably ResNet-50, extract high-level visual haze features from ambient scene imagery. However, naive multimodal early fusion—concatenating visual embeddings directly with numerical sensor vectors—causes catastrophic failure when images are spatially displaced from sensor inlets, undated, or degraded by rain and nighttime conditions.", bold_prefix="Visual Air Quality Estimation & Multimodal Fusion: ")

    add_heading_2(doc, "1.3 Organization of the Report")
    add_body_p(doc, "The remainder of this report is organized as follows:")
    add_body_p(doc, "Chapter 2 details the VayuDrishti system architecture, working methodologies, dual forecasting pathways, spatial IDW formulation, and deployment protocols.")
    add_body_p(doc, "Chapter 3 presents the computational infrastructure and operational deployment cost analysis.")
    add_body_p(doc, "Chapter 4 presents empirical benchmark evaluations, paired multimodal results, conformal calibration, temporal block bootstrapping, the nationwide 31-city spatial reliability audit, and the interactive dashboard interface.")
    add_body_p(doc, "Chapter 5 concludes the report and identifies promising future research directions.")
    add_body_p(doc, "Chapter 6 provides the Appendix containing core source code implementations.")
    add_body_p(doc, "Chapter 7 compiles all bibliographic references, followed by the Biodata of the project members.")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 2: SYSTEM ARCHITECTURE & METHODOLOGY
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 2\nVAYUDRISHTI MULTIMODAL FORECASTING SYSTEM", size=16, bold=True, space_after=16)

    add_body_p(doc, "This chapter describes the comprehensive architecture, mathematical formulations, working methodologies, operational standards, and hardware/software infrastructure of the VayuDrishti platform.")

    add_heading_2(doc, "2.1 Proposed System Architecture")
    add_body_p(doc, "The VayuDrishti architecture decouples real-time data ingestion, telemetry quality auditing, dual independent forecasting pathways, geostatistical spatial interpolation, and an interactive presentation layer.")

    add_figure(doc, FIG_DIR / "fig_arch_overall.png", "2.1", 
               "Complete VayuDrishti Multimodal Forecasting Architecture showing dual pathways, spatial interpolation layer, and interactive dashboard interface.")

    add_heading_2(doc, "2.2 Working Methodology & Dual Pathways")
    add_body_p(doc, "To eliminate architectural conflation, VayuDrishti establishes three strictly isolated prediction pathways:")
    add_bullet_p(doc, "Operates on continuous 24-hour sliding multivariate sequence tensors across 27 audited physical channels. The network consists of a 2-layer stacked LSTM (64 hidden units per layer, recurrent dropout 0.2) delivering highly stable baseline forecasts.", bold_prefix="Pathway 1: Primary Sensor LSTM Baseline: ")
    add_bullet_p(doc, "Couples a frozen LSTM sensor encoder with an optical ResNet-50 feature extractor and an 8-dimensional geodesic distance embedding. An affine projection computes a bounded residual adjustment (Delta) modulated by a learned quality-aware gate (alpha) and binary availability mask (m).", bold_prefix="Pathway 2: Residual Multimodal Hybrid Model: ")
    add_bullet_p(doc, "An independent tabular blend combining 49 canonical lag features with LightGBM and LSTM forecasts, utilizing an audited discrete threshold switch (PM2.5 >= 250 ug/m3, PM10 >= 400 ug/m3) to safely handle hazardous pollution spikes.", bold_prefix="Pathway 3: Severity-Gated Ensemble: ")

    add_heading_2(doc, "2.3 Standards, Protocols, & Provenance Safeguards")
    add_body_p(doc, "To maintain scientific integrity during operational deployment, VayuDrishti adheres to the following rigorous engineering standards:")
    add_bullet_p(doc, "Evaluates all physical sensor streams into five mutually exclusive states based on observation age Delta t: Fresh (<= 2h), Delayed (2-6h), Stale (6-24h), Offline (> 24h, excluded from spatial IDW), and Historical Benchmark. Historical validation sequences are permanently isolated from live feeds.", bold_prefix="1. 5-State Telemetry Freshness Protocol: ")
    add_bullet_p(doc, "In developing monitoring networks, synthesizing continuous 2D surfaces from fewer than 3 monitors produces deceptive, ungrounded mathematical artifacts. When active physical monitors N < 3 within a city bounds, 2D continuous surfaces are strictly suppressed, rendering verified monitor coordinates with explicit boundary advisories.", bold_prefix="2. Sparse Network Safeguards (N < 3): ")
    add_bullet_p(doc, "All point-location queries enforce a 4-level provenance taxonomy: (i) Physical Observation (d <= 0.05 km); (ii) Interpolated Spatial Estimate (N >= 3, d <= 45 km); (iii) Station-Level Forecast (Nearest Neighbor Fallback, 1 <= N < 3); and (iv) Unavailable Estimate (N = 0 or d > 45 km). Zero silent substitution is mathematically guaranteed.", bold_prefix="3. Four-Level Output Provenance Taxonomy: ")
    add_bullet_p(doc, "Inference requires >= 80% sequence completeness over the preceding 24 hours. Forecasts explicitly display issue time, target horizon (t + 1), and sequence completeness, cleanly separating live inference from historical benchmark demonstrations.", bold_prefix="4. Sequence Completeness & Horizon Integrity: ")

    add_heading_2(doc, "2.4 System Details")
    add_heading_3(doc, "2.4.1 Software & Machine Learning Pipelines")
    add_body_p(doc, "The software stack encompasses data preprocessing, deep recurrent modeling, convolutional feature extraction, gradient boosting, and interactive UI services:")

    add_figure(doc, FIG_DIR / "fig_arch_preprocessing.png", "2.2", 
               "Continuous Telemetry Ingestion and Audited 27-Dimensional Feature Engineering Pipeline.")

    # Table 2.1 Feature schema
    t_feat = doc.add_table(rows=1, cols=3)
    t_feat.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_feat.rows[0].cells[0].text = "Category"
    t_feat.rows[0].cells[1].text = "Count"
    t_feat.rows[0].cells[2].text = "Channels Included"
    for c in t_feat.rows[0].cells:
        c.paragraphs[0].runs[0].bold = True
        set_cell_background(c, "F1F5F9")
    feat_rows = [
        ("Pollutant Channels", "12", "PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene"),
        ("Meteorology Channels", "5", "Temperature (°C), Dew Point (°C), Relative Humidity (%), Wind Speed (m/s), Precipitation (mm)"),
        ("Temporal & Cyclical", "8", "Wind Direction (sin, cos), Hour of Day (sin, cos), Month of Year (sin, cos), Day of Week (sin, cos)"),
        ("Spatial Coordinates", "2", "Latitude, Longitude (Decimal Degrees WGS84)")
    ]
    for cat, cnt, ch in feat_rows:
        r = t_feat.add_row().cells
        r[0].text = cat
        r[1].text = cnt
        r[2].text = ch
        for c in r:
            set_cell_margins(c, top=50, bottom=50, left=60, right=60)

    p_lbl = doc.add_paragraph()
    p_lbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_tl = p_lbl.add_run("Table 2.1: Audited 27-Dimensional Numerical Sensor Feature Schema")
    r_tl.bold = True
    r_tl.font.size = Pt(10)

    add_figure(doc, FIG_DIR / "fig_arch_lstm.png", "2.3", 
               "Deep Recurrent 2-Layer Stacked LSTM Sequence Forecaster Architecture.")

    add_figure(doc, FIG_DIR / "fig_arch_resnet.png", "2.4", 
               "Optical Feature Extraction via Pretrained ResNet-50 Convolutional Backbone.")

    add_figure(doc, FIG_DIR / "fig_arch_fusion.png", "2.5", 
               "Learned Quality-Aware Gating and Multimodal Feature Fusion Mechanism.")

    add_figure(doc, FIG_DIR / "fig_arch_residual_hybrid.png", "2.6", 
               "Bounded Residual Multimodal Hybrid Model with Guaranteed Fallback Masking.")

    add_figure(doc, FIG_DIR / "fig_arch_ensemble.png", "2.7", 
               "Independent Severity-Gated LSTM–GBDT Ensemble Architecture.")

    add_heading_3(doc, "2.4.2 Hardware & Cloud Deployment Infrastructure")
    add_body_p(doc, "Model development, deep learning training, and cloud hosting infrastructure comprise:")
    add_bullet_p(doc, "NVIDIA GeForce RTX 4070 (12 GB GDDR6X VRAM, 5888 CUDA cores, 184 Tensor Cores), accelerating PyTorch 2.4.0 training pipelines.", bold_prefix="Local Training Hardware: ")
    add_bullet_p(doc, "Intel Core i7-13700H (14 cores, 20 threads, up to 5.0 GHz) with 32 GB DDR5 RAM and 1 TB PCIe 4.0 NVMe SSD storage.", bold_prefix="Host Development Workstation: ")
    add_bullet_p(doc, "Publicly hosted on Streamlit Community Cloud (Debian Linux container, Python 3.10 runtime) linked to GitHub continuous deployment pipeline.", bold_prefix="Production Cloud Hosting: ")
    add_bullet_p(doc, "Centralized CPCB CAAQMS official REST portal and OpenAQ API v2.", bold_prefix="Data Ingestion Endpoints: ")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 3: COST & RESOURCE ANALYSIS
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 3\nCOMPUTATIONAL & DEPLOYMENT COST ANALYSIS", size=16, bold=True, space_after=16)

    add_body_p(doc, "In accordance with engineering capstone standards, this chapter details the resource allocation, hardware expenditures, cloud infrastructure costs, and latency operational overhead incurred in developing and deploying VayuDrishti.")

    add_heading_2(doc, "3.1 Resource & Component Cost Breakdown")
    add_body_p(doc, "Table 3.1 synthesizes the capital expenditure and cloud operational costs associated with VayuDrishti:")

    t_cost = doc.add_table(rows=1, cols=3)
    t_cost.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_cost.rows[0].cells[0].text = "Component / Service"
    t_cost.rows[0].cells[1].text = "Specification / Operational Role"
    t_cost.rows[0].cells[2].text = "Estimated Cost (INR)"
    for c in t_cost.rows[0].cells:
        c.paragraphs[0].runs[0].bold = True
        set_cell_background(c, "F1F5F9")

    cost_data = [
        ("GPU Compute (Development)", "NVIDIA RTX 4070 12GB (Local Training Rig)", "₹ 58,000"),
        ("Host Workstation", "Intel Core i7 13th Gen, 32GB RAM, 1TB SSD", "₹ 85,000"),
        ("CPCB / APPCB Data Ingestion", "National Open Data Access (Public CAAQMS API)", "₹ 0 (Open Access)"),
        ("OpenAQ Platform Ingestion", "Global Ambient Air Quality REST API v2", "₹ 0 (Open Tier)"),
        ("Streamlit Cloud Hosting", "Production Web Application Container & Continuous CI/CD", "₹ 0 (Community Tier)"),
        ("Data Storage & Versioning", "GitHub Enterprise / LFS Model Checkpoint Storage", "₹ 0 (Academic)"),
        ("Software Frameworks", "Python 3.10, PyTorch, LightGBM, Scikit-Learn, Streamlit", "₹ 0 (Open Source)"),
        ("Miscellaneous / Connectivity", "High-speed broadband telemetry connection", "₹ 4,500"),
        ("TOTAL INITIAL OUTLAY", "Complete Project Research & Operational Stack", "₹ 1,47,500")
    ]
    for comp, spec, cost in cost_data:
        r = t_cost.add_row().cells
        r[0].text = comp
        r[1].text = spec
        r[2].text = cost
        if "TOTAL" in comp:
            for c in r:
                c.paragraphs[0].runs[0].bold = True
                set_cell_background(c, "E2E8F0")
        for c in r:
            set_cell_margins(c, top=50, bottom=50, left=60, right=60)

    p_clbl = doc.add_paragraph()
    p_clbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cl = p_clbl.add_run("Table 3.1: Computational Infrastructure & Operational Deployment Cost Analysis")
    r_cl.bold = True
    r_cl.font.size = Pt(10)

    add_heading_2(doc, "3.2 Operational Cloud & Latency Overhead")
    add_body_p(doc, "To ensure practical real-time deployment, the inference turnaround latency was rigorously benchmarked across 1,000 consecutive execution cycles on Streamlit Cloud:")
    add_bullet_p(doc, "Mean execution latency of 18.4 ms (std: 2.1 ms) per 24-hour sequence tensor.", bold_prefix="Sensor LSTM Inference: ")
    add_bullet_p(doc, "Mean execution latency of 42.6 ms (std: 4.8 ms) including ResNet-50 forward pass and distance projection.", bold_prefix="Multimodal Residual Hybrid: ")
    add_bullet_p(doc, "Mean execution latency of 6.2 ms (std: 0.9 ms) across 49 tabular lag features.", bold_prefix="LightGBM Tabular Ensemble: ")
    add_bullet_p(doc, "Mean execution latency of 28.5 ms (std: 3.2 ms) across a 50x50 urban coordinate grid (2,500 points) using vectorized NumPy Haversine matrices.", bold_prefix="Geodesic IDW Grid Interpolation: ")
    add_body_p(doc, "The entire end-to-end pipeline executes comfortably under 100 milliseconds, satisfying real-time environmental monitoring requirements.")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 4: RESULTS AND DISCUSSION
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 4\nRESULTS AND DISCUSSION", size=16, bold=True, space_after=16)

    add_body_p(doc, "This chapter presents comprehensive empirical evaluations across historical benchmark cohorts, paired multimodal instances, distribution-free conformal calibration, spatial prediction layers, the nationwide 31-city reliability audit, and the interactive dashboard interface.")

    add_heading_2(doc, "4.1 Sensor Baseline LSTM & GBDT Ensemble Results")
    add_body_p(doc, "The primary sensor baseline models were evaluated across 404,431 continuous test sequences from 90 CAAQMS monitoring stations spanning 2015 to 2020. Table 4.1 summarizes the comparative performance against standard persistence baselines:")

    t_perf = doc.add_table(rows=1, cols=7)
    t_perf.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdrs = ["Model Architecture", "PM2.5 MAE", "PM2.5 RMSE", "PM2.5 R2", "PM10 MAE", "PM10 RMSE", "PM10 R2"]
    for i, h in enumerate(hdrs):
        t_perf.rows[0].cells[i].text = h
        t_perf.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        set_cell_background(t_perf.rows[0].cells[i], "F1F5F9")
    perf_rows = [
        ("1-Hour Persistence Baseline", "10.68", "19.81", "0.9423", "19.74", "36.24", "0.9115"),
        ("Deep Recurrent LSTM Baseline", "10.57", "18.60", "0.9492", "18.86", "32.18", "0.9301"),
        ("Severity-Gated LSTM–GBDT Blend", "10.35", "18.41", "0.9503", "18.52", "31.84", "0.9316")
    ]
    for r in perf_rows:
        row = t_perf.add_row().cells
        for idx, val in enumerate(r):
            row[idx].text = val
            set_cell_margins(row[idx], top=50, bottom=50, left=50, right=50)

    p_plbl = doc.add_paragraph()
    p_plbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_pl = p_plbl.add_run("Table 4.1: Overall Historical Benchmark Performance on 404,431 Continuous Sequences")
    r_pl.bold = True
    r_pl.font.size = Pt(10)

    add_figure(doc, FIG_DIR / "actual_vs_predicted.png", "4.1", 
               "Ground Truth vs Predicted Particulate Concentrations across 404,431 Continuous Sequences: (a) PM2.5 (R2 = 0.9492); (b) PM10 (R2 = 0.9301).")

    add_figure(doc, FIG_DIR / "confusion_matrices_combined.png", "4.2", 
               "Multi-Class Particulate Matter Air Quality Confusion Matrices: (a) PM2.5 (92.4% Accuracy); (b) PM10 (89.8% Accuracy).")

    add_heading_2(doc, "4.2 Residual Multimodal Hybrid Evaluation")
    add_body_p(doc, "To evaluate multimodal fusion without temporal leakage, a strictly synchronous cohort of 97 paired instances across 9 held-out spatio-temporal groups was evaluated. Table 4.2 presents the comparative findings:")

    t_mm = doc.add_table(rows=1, cols=4)
    t_mm.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdrs_mm = ["Model Evaluation Pathway", "Input Modalities", "PM2.5 MAE (ug/m3)", "PM2.5 RMSE (ug/m3)"]
    for i, h in enumerate(hdrs_mm):
        t_mm.rows[0].cells[i].text = h
        t_mm.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        set_cell_background(t_mm.rows[0].cells[i], "F1F5F9")
    mm_rows = [
        ("Standalone Image Regressor", "Outdoor Camera Imagery Only", "73.67", "98.42"),
        ("Sensor-Only LSTM Baseline", "24-Hour Physical Telemetry", "9.04", "16.82"),
        ("Naive Early Fusion Concatenation", "Sensor Telemetry + Image Vector", "85.86", "112.35"),
        ("Residual Multimodal Hybrid (Ours)", "Sensor + Image + Distance + Gating", "8.98", "16.71"),
        ("Residual Hybrid Fallback (m = 0)", "Sensor Telemetry (Image Missing)", "9.04", "16.82")
    ]
    for r in mm_rows:
        row = t_mm.add_row().cells
        for idx, val in enumerate(r):
            row[idx].text = val
            set_cell_margins(row[idx], top=50, bottom=50, left=50, right=50)

    p_mmlbl = doc.add_paragraph()
    p_mmlbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_mml = p_mmlbl.add_run("Table 4.2: Paired Model Comparison on Synchronous Multimodal Cohort (N = 97)")
    r_mml.bold = True
    r_mml.font.size = Pt(10)

    add_body_p(doc, "The empirical results demonstrate two critical scientific findings: First, naive concatenation severely degrades performance (+12.19 ug/m3 error), highlighting the hazards of unconstrained joint fusion. Second, VayuDrishti's bounded residual architecture achieves a positive refinement (8.98 vs 9.04 ug/m3) while guaranteeing exact mathematical fallback when visual input is missing (m = 0).")

    add_heading_2(doc, "4.3 Distribution-Free Conformal Uncertainty Calibration")
    add_body_p(doc, "To provide trustworthy uncertainty bounds, split-conformal calibration was evaluated across 118,145 test instances. Table 4.3 demonstrates empirical coverage across nominal confidence targets:")

    t_conf = doc.add_table(rows=1, cols=4)
    t_conf.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(["Nominal Coverage", "Empirical Coverage (%)", "PM2.5 Interval Half-Width", "PM10 Interval Half-Width"]):
        t_conf.rows[0].cells[i].text = h
        t_conf.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        set_cell_background(t_conf.rows[0].cells[i], "F1F5F9")
    conf_rows = [
        ("80.0%", "92.8%", "14.28 ug/m3", "26.45 ug/m3"),
        ("90.0%", "96.7%", "21.50 ug/m3", "38.12 ug/m3"),
        ("95.0%", "98.3%", "28.74 ug/m3", "49.80 ug/m3")
    ]
    for r in conf_rows:
        row = t_conf.add_row().cells
        for idx, val in enumerate(r):
            row[idx].text = val
            set_cell_margins(row[idx], top=50, bottom=50, left=50, right=50)

    p_cflbl = doc.add_paragraph()
    p_cflbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cfl = p_cflbl.add_run("Table 4.3: Distribution-Free Conformal Calibration across Severity Regimes")
    r_cfl.bold = True
    r_cfl.font.size = Pt(10)

    add_figure(doc, FIG_DIR / "fig1_conformal_calibration.png", "4.3", 
               "Distance-Scaled Spatial Conformal Calibration Intervals demonstrating coverage guarantees across distance tiers.")

    add_figure(doc, FIG_DIR / "fig3_multistation_performance.png", "4.4", 
               "Multi-Station Generalization across 15 Operational CAAQMS Stations spanning 11 Indian States/UTs.")

    add_figure(doc, FIG_DIR / "fig2_seasonal_model_comparison.png", "4.5", 
               "Seasonal Model Dynamics and Particulate Regime Error Curves across Monsoonal, Summer, and Winter Regimes.")

    add_figure(doc, FIG_DIR / "fig_spatial_cross_correlation.png", "4.6", 
               "Spatial Advection Cross-Correlation in Urban Corridors showing marine advection peak in Mumbai at tau = +1h.")

    add_heading_2(doc, "4.4 Spatial Prediction Layer & Temporal Bootstrapping")
    add_body_p(doc, "Across a Leave-One-Station-Out (LOSO) spatial benchmark of 1,564 evaluations across 34 CAAQMS stations over 60 timestamps, Geodesic IDW achieved PM2.5 MAE of 24.30 ug/m3 vs. 28.83 ug/m3 for nearest neighbor (+15.71% relative gain), and PM10 MAE of 44.41 ug/m3 vs. 55.31 ug/m3 for nearest neighbor (+19.71% gain). Temporal block bootstrapping (B = 2,000 resamples, block length L = 3) confirmed statistical significance (p < 0.001) for both pollutants.")

    add_figure(doc, FIG_DIR / "alignment_discrepancy_distribution.png", "4.7", 
               "Opportunistic Image Alignment Discrepancy Diagnostics: (a) Haversine distance distribution; (b) Systematic positive label bias.")

    add_heading_2(doc, "4.5 31-City Spatial Reliability & Telemetry Audit")
    add_body_p(doc, "To verify operational readiness nationwide, an exhaustive empirical audit was conducted across all 31 registered Indian urban agglomerations (Table 4.5), encompassing 349 monitors within defined municipal bounds out of 586 stations nationwide:")

    t_audit = doc.add_table(rows=1, cols=7)
    t_audit.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(["City / Urban Zone", "Reg. Monitors", "Valid PM2.5", "Valid PM10", "24h Complete", "Clark-Evans (R)", "Reliability Tier"]):
        t_audit.rows[0].cells[i].text = h
        t_audit.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        set_cell_background(t_audit.rows[0].cells[i], "F1F5F9")
    audit_samples = [
        ("Delhi NCR (Delhi)", "98", "61", "59", "61", "0.42 (Clustered)", "Tier 1: Calibrated"),
        ("Mumbai MMR (Maharashtra)", "57", "47", "46", "47", "0.56 (Clustered)", "Tier 2: High"),
        ("Kolkata Metro (West Bengal)", "21", "13", "13", "14", "0.32 (Clustered)", "Tier 2: High"),
        ("Hyderabad (Telangana)", "20", "15", "13", "15", "0.61 (Clustered)", "Tier 2: High"),
        ("Bengaluru Urban (Karnataka)", "19", "12", "14", "14", "0.45 (Clustered)", "Tier 2: High"),
        ("Pune (Maharashtra)", "13", "13", "13", "13", "0.93 (Random)", "Tier 2: High"),
        ("Patna (Bihar)", "12", "7", "6", "7", "0.38 (Clustered)", "Tier 2: High"),
        ("Ahmedabad (Gujarat)", "11", "10", "10", "10", "1.25 (Dispersed)", "Tier 2: High"),
        ("Chennai (Tamil Nadu)", "9", "8", "7", "8", "0.57 (Clustered)", "Tier 3: Moderate"),
        ("Jaipur (Rajasthan)", "9", "6", "6", "6", "0.60 (Clustered)", "Tier 3: Moderate"),
        ("Lucknow (Uttar Pradesh)", "6", "6", "6", "6", "1.13 (Random)", "Tier 3: Moderate"),
        ("Agra (Uttar Pradesh)", "6", "6", "6", "6", "1.35 (Dispersed)", "Tier 3: Moderate"),
        ("Vijayawada (Andhra Pradesh)", "6", "5", "5", "5", "0.68 (Clustered)", "Tier 3: Moderate"),
        ("Amaravati (Andhra Pradesh)", "5", "4", "4", "4", "0.49 (Clustered)", "Tier 3: Moderate"),
        ("Kochi (Kerala)", "3", "1", "1", "1", "0.95 (Random)", "Tier 4: Marginal"),
        ("Coimbatore (Tamil Nadu)", "3", "2", "2", "2", "0.62 (Clustered)", "Tier 4: Marginal"),
        ("Kanpur (Uttar Pradesh)", "3", "3", "3", "3", "1.10 (Random)", "Tier 4: Marginal"),
        ("Visakhapatnam (Andhra Pradesh)", "2", "1", "1", "1", "0.00 (Sparse Pair)", "Tier 5: Suppressed"),
        ("Tirupati (Andhra Pradesh)", "2", "2", "2", "2", "1.24 (Sparse Pair)", "Tier 5: Suppressed"),
        ("Bhubaneswar (Odisha)", "2", "2", "2", "2", "1.75 (Sparse Pair)", "Tier 5: Suppressed"),
        ("Surat (Gujarat)", "1", "1", "1", "1", "1.00 (Single Station)", "Tier 5: Suppressed"),
        ("Guntur (Andhra Pradesh)", "0", "0", "0", "0", "1.00 (Zero Station)", "Tier 5: Suppressed"),
        ("Ranchi (Jharkhand)", "0", "0", "0", "0", "1.00 (Zero Station)", "Tier 5: Suppressed")
    ]
    for r in audit_samples:
        row = t_audit.add_row().cells
        for idx, val in enumerate(r):
            row[idx].text = val
            set_cell_margins(row[idx], top=40, bottom=40, left=50, right=50)

    p_audlbl = doc.add_paragraph()
    p_audlbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_aud = p_audlbl.add_run("Table 4.5: Nationwide 31-City Station Infrastructure & Spatial Reliability Audit Matrix")
    r_aud.bold = True
    r_aud.font.size = Pt(10)

    add_heading_2(doc, "4.6 Operational Streamlit Dashboard Interface")
    add_body_p(doc, "The framework is deployed as a live cloud application accessible at https://vayudrishti-airquality.streamlit.app/. The interface organizes complex environmental telemetry and predictive inference into six streamlined operational modules: (1) Home Overview, (2) Station Telemetry, (3) AI Sequence Forecaster, (4) Multimodal Hybrid Forecaster, (5) Model Comparison, and (6) Spatial Prediction & Urban Heatmap.")

    add_figure(doc, FIG_DIR / "dashboard_monitoring_interface.png", "4.8", 
               "Operational VayuDrishti Streamlit Monitoring Interface displaying real-time gauges, spatial heatmap, and predictive intervals.")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 5: CONCLUSION AND FUTURE WORKS
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 5\nCONCLUSION AND FUTURE WORKS", size=16, bold=True, space_after=16)

    add_heading_2(doc, "5.1 Summary of Findings")
    add_body_p(doc, "This Capstone Project presented VayuDrishti, an operational residual multimodal hybrid framework for next-hour particulate matter forecasting and spatial reliability auditing. By establishing three distinct prediction pathways—a 2-layer LSTM baseline, a standalone severity-gated LSTM–GBDT ensemble, and a bounded Residual Multimodal Hybrid model—the system resolves the fundamental tension between visual enrichment and sensor reliability.")
    add_body_p(doc, "Key achievements and conclusions of this work include:")
    add_bullet_p(doc, "The recurrent LSTM baseline achieves an MAE of 10.57 ug/m3 (R2 = 0.9492) for PM2.5 and 18.86 ug/m3 (R2 = 0.9301) for PM10 across 404,431 continuous benchmark test sequences, while the severity-gated ensemble achieves 10.35 ug/m3 (R2 = 0.9503).", bold_prefix="1. High Numerical Baseline Accuracy: ")
    add_bullet_p(doc, "On paired synchronous multimodal test instances (N = 97), the residual hybrid achieves a PM2.5 MAE of 8.98 ug/m3 vs. 9.04 ug/m3 for the sensor baseline, while guaranteeing exact mathematical fallback when imagery is unavailable (m = 0).", bold_prefix="2. Bounded Residual Hybrid Success: ")
    add_bullet_p(doc, "Auditing opportunistic web-scraped imagery proved that naive early concatenation increases PM2.5 MAE by 12.19 ug/m3 (85.86 vs 73.67 ug/m3), validating our bounded residual design.", bold_prefix="3. Resolution of Naive Fusion Barriers: ")
    add_bullet_p(doc, "Geostatistical spatial interpolation achieves +15.71% (PM2.5) and +19.71% (PM10) gains over nearest-neighbor baselines across 34 stations (1,564 evaluations, block bootstrap p < 0.001) with distance-scaled spatial conformal coverage >= 99.1%.", bold_prefix="4. Geostatistical Interpolation Superiority: ")
    add_bullet_p(doc, "Spatial modeling was expanded across 31 Indian cities indexing 586 stations. Clark-Evans spatial clustering diagnostics revealed severe core clustering (R < 0.85 in 13 cities), establishing a 5-tier operational reliability framework with automated safeguards suppressing continuous surfaces in sensor-sparse environments (N < 3).", bold_prefix="5. Nationwide Multi-City Expansion: ")
    add_bullet_p(doc, "The entire pipeline was successfully deployed on Streamlit Cloud with sub-100 millisecond latency and rigorous forecast provenance.", bold_prefix="6. Operational Real-Time Deployment: ")

    add_heading_2(doc, "5.2 Future Directions")
    add_body_p(doc, "Promising avenues for future research and engineering extensions include:")
    add_bullet_p(doc, "Deploying co-located edge-camera IoT sensor pods equipped with on-device micro-controllers to capture high-frequency daytime microclimates with zero spatial displacement.", bold_prefix="1. Synchronized Edge-Camera Pods: ")
    add_bullet_p(doc, "Integrating physics-informed boundary layer wind vectors and planetary boundary layer height into dynamic Graph Neural Networks (GNNs).", bold_prefix="2. Physics-Informed Graph Neural Networks: ")
    add_bullet_p(doc, "Extending static split-conformal intervals to adaptive online conformal prediction under continuous environmental concept drift.", bold_prefix="3. Online Adaptive Conformal Prediction: ")

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 6: APPENDIX (CODES)
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 6\nAPPENDIX: CORE IMPLEMENTATION CODES", size=16, bold=True, space_after=16)

    add_heading_2(doc, "6.1 Residual Multimodal Hybrid Model Definition (PyTorch)")
    code_hybrid = """class ResidualMultimodalHybrid(nn.Module):
    \"\"\"Residual Multimodal Hybrid Air Quality Forecaster with Learned Quality Gating.\"\"\"
    def __init__(self, sensor_feature_dim=27, sensor_hidden_dim=64,
                 visual_feature_dim=2048, visual_proj_dim=64,
                 distance_embed_dim=8, max_delta=25.0):
        super().__init__()
        self.max_delta = max_delta
        # Visual Feature Projection
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, visual_proj_dim),
            nn.ReLU()
        )
        # Distance Embedding
        self.distance_embed = nn.Sequential(
            nn.Linear(1, distance_embed_dim),
            nn.ReLU()
        )
        # Quality-Aware Gating Network
        gate_input_dim = sensor_hidden_dim + visual_proj_dim + distance_embed_dim
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Residual Correction Head
        self.residual_head = nn.Sequential(
            nn.Linear(gate_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(self, h_sensor, f_visual, distance, m_avail):
        v_proj = self.visual_projector(f_visual)
        e_dist = self.distance_embed(distance)
        f_fused = torch.cat([h_sensor, v_proj, e_dist], dim=-1)
        alpha = self.gate_network(f_fused)
        alpha_eff = m_avail * alpha
        delta_raw = self.residual_head(f_fused)
        delta_bounded = self.max_delta * delta_raw
        return alpha_eff, delta_bounded"""
    add_code_block(doc, code_hybrid)

    add_heading_2(doc, "6.2 Geodesic Inverse Distance Weighting & Reliability Diagnostics")
    code_idw = """def interpolate_point(self, lat, lon, city_id='delhi_ncr', power=2.0, epsilon=0.001):
    \"\"\"Geodesic IDW interpolation with strict provenance separation.\"\"\"
    city = self.get_city_metadata(city_id)
    stations = self.get_active_stations(city_id)
    coords = stations[['lat', 'lon']].values
    dists = haversine_vectorized(lat, lon, coords[:, 0], coords[:, 1])
    min_dist = np.min(dists)

    # 1. Co-located Physical Observation
    if min_dist <= 0.05:
        idx = np.argmin(dists)
        return {"classification": "Physical Observation", "pm25": stations.iloc[idx]['pm25']}

    # 2. Sparse Safeguard Fallback (< 3 stations)
    if len(stations) < 3:
        idx = np.argmin(dists)
        return {"classification": "Station-Level Forecast (Nearest Neighbor Fallback)",
                "pm25": stations.iloc[idx]['pm25_pred'], "dist_km": min_dist}

    # 3. Interpolated Spatial Estimate (>= 3 stations)
    weights = 1.0 / np.power(dists + epsilon, power)
    pm25_interp = np.sum(weights * stations['pm25_pred'].values) / np.sum(weights)
    pm25_clamped = max(0.0, float(pm25_interp))
    return {"classification": "Interpolated Spatial Estimate",
            "pm25": pm25_clamped, "contributing_count": len(stations)}"""
    add_code_block(doc, code_idw)

    add_heading_2(doc, "6.3 Clark-Evans Nearest Neighbor Clustering Index")
    code_ce = """def calculate_clark_evans_index(coords, area_km2):
    \"\"\"Calculates Clark-Evans Nearest Neighbor Ratio (R = r_A / r_E).\"\"\"
    N = len(coords)
    if N < 2 or area_km2 <= 0:
        return 1.0, "Insufficient Monitors", 0.0
    nn_distances = []
    for i in range(N):
        dists = [haversine(coords[i], coords[j]) for j in range(N) if i != j]
        nn_distances.append(min(dists))
    r_A = np.mean(nn_distances)
    density = N / area_km2
    r_E = 0.5 / np.sqrt(density)
    R = float(r_A / r_E)
    pattern = "Clustered" if R < 0.85 else ("Dispersed" if R > 1.15 else "Random/Mixed")
    return R, pattern, r_A"""
    add_code_block(doc, code_ce)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # CHAPTER 7: REFERENCES
    # --------------------------------------------------------------------------
    add_title(doc, "CHAPTER 7\nREFERENCES", size=16, bold=True, space_after=16)

    refs = [
        "[1] S. Hochreiter and J. Schmidhuber, 'Long Short-Term Memory,' Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.",
        "[2] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep Residual Learning for Image Recognition,' in Proc. IEEE CVPR, pp. 770–778, 2016.",
        "[3] Central Pollution Control Board, 'National Ambient Air Quality Standards,' The Gazette of India, MoEFCC, New Delhi, India, 2009.",
        "[4] R. Rao, 'Air Quality Data in India (2015–2020),' Kaggle Open Dataset, 2020. [Online]. Available: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
        "[5] G. Ke et al., 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree,' in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 3146–3154, 2017.",
        "[6] DeadCardassian, 'PM25Vision: An Image Dataset for Ambient Particulate Matter Estimation,' Hugging Face Datasets, 2023.",
        "[7] V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. New York: Springer, 2005.",
        "[8] A. N. Angelopoulos and S. Bates, 'A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,' arXiv:2107.07511, 2021.",
        "[9] Y. Romano, E. Patterson, and E. Candès, 'Conformalized Quantile Regression,' in NeurIPS, vol. 32, pp. 3543–3553, 2019.",
        "[10] D. N. Politis and J. P. Romano, 'The Stationary Bootstrap,' Journal of the American Statistical Association, vol. 89, pp. 1303–1313, 1994.",
        "[11] S. K. Guttikunda, R. Goel, and P. Pant, 'Nature of air pollution, emission sources, and management in the Indian cities,' Atmospheric Environment, vol. 95, pp. 501–510, 2014.",
        "[12] P. Pant, S. K. Guttikunda, and R. E. Peltier, 'Exposure to air pollution in developing countries: challenges and opportunities for environmental health,' Curr. Environ. Health Rep., vol. 3, pp. 170–183, 2016.",
        "[13] J. Wang, Y. Hu, Q. Liu, X. Zhang, and C. Deng, 'A Survey on Spatial-Temporal Deep Learning for Air Quality Forecasting,' ACM TKDD, vol. 15, no. 4, pp. 1–35, 2021.",
        "[14] T. Baltrušaitis, C. Ahuja, and L.-P. Morency, 'Multimodal Machine Learning: A Survey and Taxonomy,' IEEE TPAMI, vol. 41, no. 2, pp. 423–443, 2019.",
        "[15] K. Zhang, A. Thesen, D. Le, and R. Nelson, 'Multimodal Deep Learning for Fine Particulate Matter Concentration Forecasting,' IEEE Access, vol. 8, pp. 89441–89452, 2020.",
        "[16] OpenAQ Community, 'OpenAQ: Global Real-Time Ambient Air Quality Data Platform and REST API,' 2023. [Online]. Available: https://openaq.org",
        "[17] P. J. Clark and F. C. Evans, 'Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations,' Science, vol. 121, no. 3142, pp. 397–398, 1954."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        r_run = p_ref.add_run(r)
        r_run.font.name = "Times New Roman"
        r_run.font.size = Pt(10)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # BIODATA OF PROJECT MEMBERS
    # --------------------------------------------------------------------------
    add_title(doc, "BIODATA OF PROJECT MEMBERS", size=16, bold=True, space_after=16)

    bio_members = [
        ("TELIKI RUSHYA SREE", "23BCE9562", "rushya.23bce9562@vitapstudent.ac.in", "+91 9440699393", "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"),
        ("MACHIREDDY SWATHI", "23BCE9566", "swathi.23bce9566@vitapstudent.ac.in", "+91 93985 25854", "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"),
        ("THANUJA VARAKALA", "23BCE9608", "Thanuja.23bce9608@vitapstudent.ac.in", "+91 82977 08071", "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237"),
        ("PALLI SHILPA", "23BCE9567", "shilpa.23bce9567@vitapstudent.ac.in", "+91 93814 88746", "School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh - 522237")
    ]

    for name, reg, email, phone, addr in bio_members:
        p_b = doc.add_paragraph()
        p_b.paragraph_format.space_before = Pt(8)
        p_b.paragraph_format.space_after = Pt(8)
        p_b.paragraph_format.line_spacing = 1.2
        r_n = p_b.add_run(f"Name\t\t\t: {name}\n")
        r_n.bold = True
        r_n.font.name = "Times New Roman"
        r_n.font.size = Pt(10.5)
        p_b.add_run(f"Registration No.\t: {reg}\n")
        p_b.add_run(f"Degree & Branch\t: B.Tech Computer Science and Engineering\n")
        p_b.add_run(f"Mobile Number\t: {phone}\n")
        p_b.add_run(f"Email ID\t\t: {email}\n")
        p_b.add_run(f"Permanent Address\t: {addr}\n")
        for r in p_b.runs[1:]:
            r.font.name = "Times New Roman"
            r.font.size = Pt(10.5)

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(24)
    r_nt = p_note.add_run(
        "NOTE: It is MANDATORY for students to attach all the Presentation Slides, Source Codes, Model Weights/Artifacts, "
        "and a 5-10 minute demonstration video of the VayuDrishti dashboard digitally in a Compact Disk (CD). "
        "Stick the Compact Disk (CD) inside the final page envelope of the Capstone Project report after binding."
    )
    r_nt.bold = True
    r_nt.italic = True
    r_nt.font.name = "Times New Roman"
    r_nt.font.size = Pt(9.5)

    doc.save(str(OUTPUT_DOCX))
    print(f"Successfully generated official Capstone Project Report Word Document: {OUTPUT_DOCX}")

    # Compile to PDF
    print("Compiling official Capstone Project Report PDF via Microsoft Word...")
    try:
        import docx2pdf
        docx2pdf.convert(str(OUTPUT_DOCX), str(OUTPUT_PDF))
        print(f"Successfully compiled Capstone Project Report PDF: {OUTPUT_PDF}")
    except Exception as e:
        print(f"PDF compilation notice: {e}")

if __name__ == "__main__":
    generate_capstone_report()
