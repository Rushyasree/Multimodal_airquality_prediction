"""
scripts/build_final_capstone_presentation.py

Generates the fully audited, academically verified, and professionally formatted
PowerPoint presentation:
`presentation/VayuDrishti_Final_Capstone_Review.pptx`
(and mirrors to `docs/VayuDrishti_Final_Capstone_Review.pptx`)

Preserves the original 16-slide university template structure while incorporating all
technical corrections from the Master Prompt:
- Correct official title: "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments"
- Verified metadata (Prof. Thota Ramathulsi, SCOPE, 4 students with 23BCE IDs)
- Strict distinction between LSTM (27 features) and adapted GBDT (49 features)
- 24h lookback predicting strictly next hour (t+1)
- ResNet-50 visual feature extractor and bounded residual hybrid with fallback (m=0)
- Accurate Clark-Evans values from Phase 16 audit (Delhi R=0.42, Mumbai R=0.56, Kolkata R=0.32, etc.)
- Statistical non-significance of 14-day live audit (p=0.48, p=0.62)
- Geodesic IDW (p=2.0) vs 1-NN holdout in Delhi (-20.0% PM2.5, -25.4% PM10)
- Natural spoken speaker scripts and comprehensive viva defense notes
"""

import os
import shutil
import sys
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PPTX = ROOT / "presentation/Capstone_PPT_Template.pptx"
OUT_PPTX_PRES = ROOT / "presentation/VayuDrishti_Final_Capstone_Review.pptx"
OUT_PPTX_DOCS = ROOT / "docs/VayuDrishti_Final_Capstone_Review.pptx"
FIG_DIR = ROOT / "paper/figures"

# Colors
C_NAVY = RGBColor(16, 44, 87)       # Deep Navy #102C57
C_DARK = RGBColor(24, 43, 73)       # Dark Slate #182B49
C_TEAL = RGBColor(14, 116, 144)     # Teal #0E7490
C_LIGHT_BG = RGBColor(248, 250, 252)# Slate 50 #F8FAFC
C_CARD_BG = RGBColor(241, 245, 249) # Slate 100 #F1F5F9
C_CARD_BORDER = RGBColor(203, 213, 225) # Slate 300 #CBD5E1
C_TEXT_MAIN = RGBColor(15, 23, 42)  # Slate 900 #0F172A
C_TEXT_MUTED = RGBColor(71, 85, 105)# Slate 600 #475569
C_ACCENT_BLUE = RGBColor(37, 99, 235) # Blue 600 #2563EB
C_MAROON = RGBColor(87, 35, 20)     # University Header Maroon #572314
C_GREEN = RGBColor(22, 101, 52)     # Emerald 800 #166534
C_RED = RGBColor(185, 28, 28)       # Red 700 #B91C1C
C_WHITE = RGBColor(255, 255, 255)
C_LIGHT_BLUE = RGBColor(239, 246, 255) # Blue 50 #EFF6FF
C_AMBER_BG = RGBColor(254, 243, 199) # Amber 100 #FEF3C7
C_AMBER_BORDER = RGBColor(245, 158, 11) # Amber 500 #F59E0B

FONT_HEADING = "Times New Roman"
FONT_SUBHEADING = "Gill Sans MT"
FONT_BODY = "Calibri"

def set_speaker_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text.strip()

def clear_body_shapes(slide, keep_names=('TextShape 1', 'TextShape 2', 'TextShape 3', 'TextShape 5', 'Picture 2', 'Picture 3', 'Picture 4', 'Title 1', 'Subtitle 2')):
    """Removes temporary shapes leaving headers, footers, and logos."""
    shapes_to_remove = []
    for s in slide.shapes:
        if s.name not in keep_names:
            shapes_to_remove.append(s)
    for s in shapes_to_remove:
        sp = s._element
        sp.getparent().remove(sp)

def setup_header(slide, section_text, topic_text):
    for shape in slide.shapes:
        if shape.name == 'TextShape 1':
            shape.text_frame.word_wrap = True
            shape.text_frame.margin_left = Inches(0.1)
            shape.text_frame.margin_top = Inches(0.04)
            shape.left = Inches(0.2)
            shape.width = Inches(8.0)
            p = shape.text_frame.paragraphs[0]
            p.text = section_text
            p.font.name = FONT_HEADING
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = C_MAROON
        elif shape.name == 'TextShape 2':
            shape.text_frame.word_wrap = True
            shape.text_frame.margin_left = Inches(0.1)
            shape.text_frame.margin_top = Inches(0.0)
            shape.left = Inches(0.2)
            shape.top = Inches(0.52)
            shape.width = Inches(8.0)
            shape.height = Inches(0.42)
            p = shape.text_frame.paragraphs[0]
            p.text = topic_text
            p.font.name = FONT_SUBHEADING
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = C_DARK
            # Remove any trailing paragraphs
            while len(shape.text_frame.paragraphs) > 1:
                p_extra = shape.text_frame.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)

def create_card(slide, left, top, width, height, title="", border_color=C_CARD_BORDER, bg_color=C_CARD_BG):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.0)
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    
    if title:
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_SUBHEADING
        p0.font.size = Pt(11.0)
        p0.font.bold = True
        p0.font.color.rgb = C_NAVY
        p0.space_after = Pt(2)
    return shape, tf

def add_bullet(tf, text, bold_prefix="", level=0, pt_size=9.0, text_color=C_TEXT_MAIN, prefix_color=C_NAVY, space_after=2):
    if len(tf.paragraphs) == 1 and not tf.paragraphs[0].text.strip():
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.level = level
    p.space_after = Pt(space_after)
    p.space_before = Pt(1)
    if bold_prefix:
        r_pre = p.add_run()
        r_pre.text = bold_prefix + " "
        r_pre.font.name = FONT_SUBHEADING
        r_pre.font.bold = True
        r_pre.font.size = Pt(pt_size)
        r_pre.font.color.rgb = prefix_color
    r_body = p.add_run()
    r_body.text = text
    r_body.font.name = FONT_BODY
    r_body.font.size = Pt(pt_size)
    r_body.font.color.rgb = text_color

def create_table(slide, left, top, width, height, headers, rows, col_widths=None):
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table
    
    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w
            
    # Format Headers
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.name = FONT_SUBHEADING
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = C_WHITE
        
    # Format Data Rows
    for r_idx, row_data in enumerate(rows):
        bg = C_WHITE if r_idx % 2 == 0 else C_CARD_BG
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            if c_idx == 0:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT_BODY
            r.font.size = Pt(8.0)
            r.font.color.rgb = C_TEXT_MAIN
            if c_idx == 0:
                r.font.bold = True
                
    return table_shape

def build_presentation():
    print(f"Loading template: {TEMPLATE_PPTX}")
    prs = Presentation(str(TEMPLATE_PPTX))
    assert len(prs.slides) == 16, f"Expected 16 slides, got {len(prs.slides)}"

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    print("Formatting Slide 1: Title Slide...")
    s1 = prs.slides[0]
    for shape in s1.shapes:
        if shape.name == 'TextShape 1':
            shape.left = Inches(0.5)
            shape.top = Inches(0.4)
            shape.width = Inches(9.0)
            shape.height = Inches(1.8)
            tf = shape.text_frame
            tf.word_wrap = True
            
            p0 = tf.paragraphs[0]
            p0.text = "Capstone Final Review"
            p0.font.name = FONT_HEADING
            p0.font.size = Pt(20)
            p0.font.bold = True
            p0.font.color.rgb = C_MAROON
            p0.space_after = Pt(4)
            
            p1 = tf.paragraphs[1] if len(tf.paragraphs) > 1 else tf.add_paragraph()
            p1.text = "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments"
            p1.font.name = FONT_HEADING
            p1.font.size = Pt(22)
            p1.font.bold = True
            p1.font.color.rgb = C_NAVY
            p1.space_after = Pt(4)
            
            while len(tf.paragraphs) > 2:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
                
        elif shape.name == 'TextShape 2':
            shape.left = Inches(0.5)
            shape.top = Inches(2.35)
            shape.width = Inches(8.8)
            shape.height = Inches(0.65)
            tf = shape.text_frame
            tf.word_wrap = True
            p0 = tf.paragraphs[0]
            p0.text = "Supervised By: Prof. Thota Ramathulsi"
            p0.font.name = FONT_SUBHEADING
            p0.font.size = Pt(13)
            p0.font.bold = True
            p0.font.color.rgb = C_DARK
            p1 = tf.add_paragraph()
            p1.text = "Assistant Professor, School of Computer Science and Engineering (SCOPE), VIT-AP University"
            p1.font.name = FONT_BODY
            p1.font.size = Pt(10.5)
            p1.font.color.rgb = C_TEXT_MUTED
            while len(tf.paragraphs) > 2:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
                
        elif shape.name == 'CustomShape 3':
            shape.left = Inches(0.5)
            shape.top = Inches(3.1)
            shape.width = Inches(4.8)
            shape.height = Inches(2.1)
            shape.fill.solid()
            shape.fill.fore_color.rgb = C_CARD_BG
            shape.line.color.rgb = C_CARD_BORDER
            tf = shape.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.15)
            tf.margin_top = Inches(0.1)
            
            p0 = tf.paragraphs[0]
            p0.text = "Project Team Members (SDP ID: [Insert SDP ID])"
            p0.font.name = FONT_SUBHEADING
            p0.font.size = Pt(11)
            p0.font.bold = True
            p0.font.color.rgb = C_NAVY
            p0.space_after = Pt(4)
            
            students = [
                ("TELIKI RUSHYA SREE", "23BCE9562"),
                ("MACHIREDDY SWATHI", "23BCE9566"),
                ("THANUJA VARAKALA", "23BCE9608"),
                ("PALLI SHILPA", "23BCE9567")
            ]
            for idx, (s_name, s_reg) in enumerate(students):
                p = tf.paragraphs[idx+1] if len(tf.paragraphs) > idx+1 else tf.add_paragraph()
                p.text = f"•  {s_name}  ({s_reg})"
                p.font.name = FONT_BODY
                p.font.size = Pt(9.5)
                p.font.color.rgb = C_TEXT_MAIN
                p.space_after = Pt(2)
            while len(tf.paragraphs) > len(students) + 1:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
                
        elif shape.name == 'CustomShape 4':
            shape.left = Inches(5.5)
            shape.top = Inches(3.1)
            shape.width = Inches(4.0)
            shape.height = Inches(2.1)
            shape.fill.solid()
            shape.fill.fore_color.rgb = C_LIGHT_BLUE
            shape.line.color.rgb = C_ACCENT_BLUE
            tf = shape.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.15)
            tf.margin_top = Inches(0.15)
            
            p0 = tf.paragraphs[0]
            p0.text = "Academic Institution"
            p0.font.name = FONT_SUBHEADING
            p0.font.size = Pt(11)
            p0.font.bold = True
            p0.font.color.rgb = C_NAVY
            p0.space_after = Pt(4)
            
            inst_lines = [
                "School of Computer Science and Engineering",
                "VIT-AP University, Amaravati",
                "Andhra Pradesh - 522237, India",
                "Degree: B.Tech Computer Science and Engineering",
                "Academic Year: 2026-2027"
            ]
            for idx, line in enumerate(inst_lines):
                p = tf.paragraphs[idx+1] if len(tf.paragraphs) > idx+1 else tf.add_paragraph()
                p.text = line
                p.font.name = FONT_BODY
                p.font.size = Pt(9.0)
                p.font.color.rgb = C_TEXT_MAIN
                p.space_after = Pt(2)
            while len(tf.paragraphs) > len(inst_lines) + 1:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)

    s1_notes = (
        "Respected members of the review committee and our project guide Prof. Thota Ramathulsi, "
        "good morning. We are team VayuDrishti, comprising Teliki Rushya Sree, Machireddy Swathi, "
        "Thanuja Varakala, and Palli Shilpa from the School of Computer Science and Engineering, VIT-AP University. "
        "Today, we present our Capstone Final Review on 'VayuDrishti: Multimodal Spatio-Temporal Air Quality "
        "Prediction in Indian Urban Environments.' In this presentation, we will walk you through our end-to-end "
        "engineered framework, which addresses the critical challenges of urban sensor sparsity, multimodal "
        "feature fusion, next-hour predictive forecasting, and spatially safeguarded environmental interpolation "
        "across Indian cities."
    )
    set_speaker_notes(s1, s1_notes)

    # =========================================================================
    # SLIDE 2: PRESENTATION OUTLINE
    # =========================================================================
    print("Formatting Slide 2: Presentation Outline...")
    s2 = prs.slides[1]
    clear_body_shapes(s2)
    setup_header(s2, "TABLE OF CONTENTS", "Presentation Outline & Structure")
    
    # Left Card: Part I
    c1, tf1 = create_card(s2, Inches(0.5), Inches(1.1), Inches(4.35), Inches(3.9), 
                          "Part I: Foundations & Research Formulation", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf1, "Context, Scope & Three Operational Modes", bold_prefix="03. Project Overview:")
    add_bullet(tf1, "Sensor Sparsity, Clustering & Telemetry Latency", bold_prefix="04. Problem Statement:")
    add_bullet(tf1, "4 Technical Objectives & Measurable Deliverables", bold_prefix="05. Project Objectives:")
    add_bullet(tf1, "Comparative Study of 6 Existing Paradigms", bold_prefix="06. Literature Review:")
    add_bullet(tf1, "Bounded Residual Hybrid & Spatial Safeguards", bold_prefix="07. Key Contributions:")
    
    # Right Card: Part II
    c2, tf2 = create_card(s2, Inches(5.15), Inches(1.1), Inches(4.35), Inches(3.9), 
                          "Part II: Technical Implementation & Defense", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "End-to-End Multimodal Pipeline & Information Flow", bold_prefix="08. System Architecture:")
    add_bullet(tf2, "6-Module Interactive Streamlit Platform Architecture", bold_prefix="09. Dashboard Interface:")
    add_bullet(tf2, "Geodesic IDW, Clark-Evans, Conformal Intervals", bold_prefix="10. Algorithms & Tech Stack:")
    add_bullet(tf2, "4 Corpora, Strict LSTM (27) vs GBDT (49) Features", bold_prefix="11. Datasets & Features:")
    add_bullet(tf2, "Hyperparameters, 45 km Cutoff, 5 Freshness Tiers", bold_prefix="12. Setup & Safeguards:")
    add_bullet(tf2, "Historical Benchmarks, Delhi Holdout & Live Audit", bold_prefix="13. Empirical Results:")
    add_bullet(tf2, "Autoregressive Dominance, ST-GNNs & Satellites", bold_prefix="14. Findings & Future Work:")
    add_bullet(tf2, "8 Peer-Reviewed Foundations & Standards", bold_prefix="15. References & Citations:")
    add_bullet(tf2, "Open Artifacts, Live Cloud URL & Committee Defense", bold_prefix="16. Closing & Discussion:")

    s2_notes = (
        "Here is our presentation roadmap for today's review. The presentation is divided into two logical sections: "
        "First, Foundations and Research Formulation, where we establish the urban air quality challenges in India, "
        "project objectives, and existing literature gaps. Second, Technical Architecture, Empirical Experiments, "
        "and Academic Defense, covering our dual temporal forecasting engine, bounded residual vision integration, "
        "geodesic spatial interpolation, live audit results, and future directions."
    )
    set_speaker_notes(s2, s2_notes)

    # =========================================================================
    # SLIDE 3: INTRODUCTION & PROJECT OVERVIEW
    # =========================================================================
    print("Formatting Slide 3: Introduction & Project Overview...")
    s3 = prs.slides[2]
    clear_body_shapes(s3)
    setup_header(s3, "INTRODUCTION", "Context, Scope & Three Operational Modes")
    
    c1, tf1 = create_card(s3, Inches(0.5), Inches(1.1), Inches(2.85), Inches(3.3), 
                          "Mode 1: Live Telemetry Ingestion", border_color=C_ACCENT_BLUE, bg_color=C_LIGHT_BLUE)
    add_bullet(tf1, "OpenAQ v3 REST client ingests continuous hourly data across 85+ active CPCB/APPCB monitors in 31 Indian cities.", bold_prefix="Automated Feed:")
    add_bullet(tf1, "Tracks PM2.5, PM10, NO2, SO2, CO, O3, temperature, relative humidity, and wind vectors.", bold_prefix="Multi-Pollutant:")
    add_bullet(tf1, "Labels observations into 5 explicit freshness tiers (Fresh, Delayed, Stale, Offline, Unknown).", bold_prefix="Quality Tiers:")

    c2, tf2 = create_card(s3, Inches(3.55), Inches(1.1), Inches(2.9), Inches(3.3), 
                          "Mode 2: 1-Hour Forward Forecasting", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "Dual predictive engine: 2-layer LSTM (27 channels) and adapted LightGBM (49 features).", bold_prefix="Dual Architecture:")
    add_bullet(tf2, "Consumes 24-hour historical lookback sequences to predict strictly next hour (t+1).", bold_prefix="Forecast Horizon:")
    add_bullet(tf2, "Evaluated strictly against a 1-hour persistence baseline under high temporal autocorrelation.", bold_prefix="Honest Baselines:")

    c3, tf3 = create_card(s3, Inches(6.65), Inches(1.1), Inches(2.85), Inches(3.3), 
                          "Mode 3: Spatial Safeguarding", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf3, "Geodesic Haversine IDW (p=2.0) generates continuous urban particulate heatmaps.", bold_prefix="Geodesic IDW:")
    add_bullet(tf3, "Enforces an empirical 45 km cutoff radius; suppresses ungrounded predictions in sparse zones.", bold_prefix="Distance Guard:")
    add_bullet(tf3, "Integrates Clark-Evans R nearest-neighbor statistics to detect urban monitoring clustering.", bold_prefix="Geometry Audit:")

    # Bottom Highlight Banner
    cb, tfb = create_card(s3, Inches(0.5), Inches(4.5), Inches(9.0), Inches(0.55), 
                          "", border_color=C_AMBER_BORDER, bg_color=C_AMBER_BG)
    add_bullet(tfb, "VayuDrishti maintains strict conceptual separation between measured sensor telemetry, forward predictive forecasts (t+1), and spatial interpolations, preventing uncalibrated over-confidence in public health alerts.", bold_prefix="Core Scientific Differentiator:", pt_size=8.8)

    s3_notes = (
        "To introduce our project: Air quality monitoring across Indian cities presents profound spatial and temporal challenges. "
        "Rather than treating air quality prediction as a generic machine learning problem, VayuDrishti operates across three distinct "
        "functional modes: First, real-time ingestion of live regulatory telemetry from 85+ active CPCB and APPCB stations across 31 "
        "cities via OpenAQ v3. Second, localized next-hour forward forecasting at monitored locations using deep recurrent and gradient "
        "boosted models. Third, spatially safeguarded inverse distance weighting for unmonitored coordinates, augmented with spatial "
        "clustering diagnostics. Critically, we maintain strict distinctions between measured sensor telemetry, forward predictive "
        "forecasts, and spatial interpolations to ensure complete operational credibility."
    )
    set_speaker_notes(s3, s3_notes)

    # =========================================================================
    # SLIDE 4: PROBLEM STATEMENT & MOTIVATIONS
    # =========================================================================
    print("Formatting Slide 4: Problem Statement & Motivations...")
    s4 = prs.slides[3]
    clear_body_shapes(s4)
    setup_header(s4, "PROBLEM STATEMENT", "Four Critical Gaps in Indian Air Quality Monitoring")
    
    # 2x2 Grid
    c1, tf1 = create_card(s4, Inches(0.5), Inches(1.1), Inches(4.35), Inches(1.9), 
                          "1. Severe Sensor Sparsity & Spatial Clustering", border_color=C_RED, bg_color=C_CARD_BG)
    add_bullet(tf1, "Regulatory Continuous Ambient Air Quality Monitoring Stations (CAAQMS) are scarce and expensive ($150k+ capital cost).", bold_prefix="Infrastructure Deficit:")
    add_bullet(tf1, "Monitors are heavily clustered in central administrative districts; peri-urban corridors remain completely unmonitored.", bold_prefix="Peripheral Blind Spots:")
    add_bullet(tf1, "Clark-Evans nearest-neighbor statistics confirm heavy spatial clustering across Indian metros (Kolkata R=0.32, Delhi R=0.42).", bold_prefix="Quantified Clustering:")

    c2, tf2 = create_card(s4, Inches(5.15), Inches(1.1), Inches(4.35), Inches(1.9), 
                          "2. Telemetry Latency & Data Dropout", border_color=C_AMBER_BORDER, bg_color=C_CARD_BG)
    add_bullet(tf2, "Frequent sensor calibration maintenance, hardware power cuts, and network outages induce significant data missingness.", bold_prefix="Transmission Dropouts:")
    add_bullet(tf2, "Conventional dashboards display hours-old observations without timestamp warnings, masquerading stale data as live readings.", bold_prefix="Staleness Deception:")
    add_bullet(tf2, "High-stakes public health decisions are made on outdated telemetry during severe air pollution inversion episodes.", bold_prefix="Decision Hazards:")

    c3, tf3 = create_card(s4, Inches(0.5), Inches(3.1), Inches(4.35), Inches(1.9), 
                          "3. Multimodal Pairing & Negative Transfer", border_color=C_NAVY, bg_color=C_CARD_BG)
    add_bullet(tf3, "Atmospheric optical extinction captured in sky imagery decouples from ground particulate mass during winter fog and haze.", bold_prefix="Decoupled Physics:")
    add_bullet(tf3, "Naive early fusion of image features with tabular sensors inflates prediction error (+12.19 ug/m3 error inflation).", bold_prefix="Negative Transfer:")
    add_bullet(tf3, "Missing camera images cause catastrophic pipeline failure in unconstrained end-to-end vision-language architectures.", bold_prefix="Missing Modality Crash:")

    c4, tf4 = create_card(s4, Inches(5.15), Inches(3.1), Inches(4.35), Inches(1.9), 
                          "4. Spatial Over-Confidence in Unmonitored Zones", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf4, "Traditional GIS Kriging and spline interpolations paint smooth, continuous heatmaps across hundreds of unmonitored kilometers.", bold_prefix="Ungrounded Surfaces:")
    add_bullet(tf4, "Dashboards fail to communicate that spatial uncertainty explodes when distance to the nearest station exceeds 30 km.", bold_prefix="Hidden Uncertainty:")
    add_bullet(tf4, "Citizens in rural and peri-urban zones are presented synthetic values disguised as verified local observations.", bold_prefix="False Precision:")

    s4_notes = (
        "Our research is motivated by four fundamental challenges observed in Indian air quality systems: "
        "First, spatial clustering. Using Clark-Evans nearest-neighbor statistics, we proved that monitoring stations "
        "in India are heavily clustered in central urban districts—for instance, Kolkata has an R of 0.32 and Delhi has 0.42—creating "
        "large spatial blind spots. Second, telemetry dropouts: stations frequently stop transmitting or report stale data, yet existing "
        "tools fail to classify observation age. Third, multimodal negative transfer: naively fusing sky images with tabular sensor "
        "streams inflates error because visual haze and ground mass concentration decouple during fog or variable lighting. "
        "Fourth, spatial overconfidence: traditional GIS heatmaps extrapolate across empty terrain without distance constraints. "
        "VayuDrishti was engineered to systematically solve each of these four failure modes."
    )
    set_speaker_notes(s4, s4_notes)

    # =========================================================================
    # SLIDE 5: OBJECTIVES & EXPECTED IMPACT
    # =========================================================================
    print("Formatting Slide 5: Objectives & Expected Impact...")
    s5 = prs.slides[4]
    clear_body_shapes(s5)
    setup_header(s5, "PROJECT OBJECTIVES", "Technical Objectives & Measurable Deliverables")
    
    c1, tf1 = create_card(s5, Inches(0.5), Inches(1.1), Inches(4.35), Inches(1.9), 
                          "Objective 1: Resilient Multi-City Telemetry Pipeline", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf1, "Build an automated OpenAQ v3 ingestion client tracking 85+ stations across 31 Indian cities.", bold_prefix="Multi-City Feed:")
    add_bullet(tf1, "Formulate a strict 5-tier freshness categorization (Fresh <2h, Delayed 2-6h, Stale 6-24h, Offline >24h, Unknown).", bold_prefix="Freshness Protocol:")
    add_bullet(tf1, "Guarantee zero silent failures via exponential retry backoffs and local SQLite caching.", bold_prefix="Fault Tolerance:")

    c2, tf2 = create_card(s5, Inches(5.15), Inches(1.1), Inches(4.35), Inches(1.9), 
                          "Objective 2: Dual Next-Hour Predictive Engine (t+1)", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "Develop a 2-layer LSTM (27 channels) capturing temporal dynamics over 24-hour historical windows.", bold_prefix="Sequence Modeling:")
    add_bullet(tf2, "Adapt a LightGBM GBDT (49 features) leveraging multi-scale lags and rolling statistics.", bold_prefix="Gradient Boosting:")
    add_bullet(tf2, "Benchmark strictly against a 1-hour persistence baseline under high temporal autocorrelation.", bold_prefix="Statistical Defense:")

    c3, tf3 = create_card(s5, Inches(0.5), Inches(3.1), Inches(4.35), Inches(1.9), 
                          "Objective 3: Bounded Multimodal Residual Fusion", border_color=C_ACCENT_BLUE, bg_color=C_CARD_BG)
    add_bullet(tf3, "Extract 2048-dim deep visual features using a pretrained ResNet-50 convolutional backbone.", bold_prefix="Deep Vision Backbone:")
    add_bullet(tf3, "Bound visual residual corrections within [-25, +25] ug/m3 to prevent negative transfer.", bold_prefix="Bounded Gating:")
    add_bullet(tf3, "Enforce an absolute mathematical zero-fallback (m=0) when outdoor imagery is unavailable.", bold_prefix="Safe Fallback:")

    c4, tf4 = create_card(s5, Inches(5.15), Inches(3.1), Inches(4.35), Inches(1.9), 
                          "Objective 4: Spatial Safeguards & Calibrated Uncertainty", border_color=C_GREEN, bg_color=C_LIGHT_BLUE)
    add_bullet(tf4, "Deploy Geodesic Haversine IDW (p=2.0) with an empirical 45 km cutoff radius limit.", bold_prefix="Geodesic Interpolation:")
    add_bullet(tf4, "Compute Clark-Evans R nearest-neighbor clustering metrics across all 31 city registries.", bold_prefix="Geometry Diagnostics:")
    add_bullet(tf4, "Integrate split conformal prediction guaranteeing 90% marginal coverage without parametric assumptions.", bold_prefix="Conformal Calibration:")

    s5_notes = (
        "To address these gaps, we formulated four technical objectives: First, develop an automated telemetry pipeline "
        "ingesting live observations across 31 Indian cities, categorizing data into five explicit freshness tiers. "
        "Second, develop dual forecasting architectures predicting next-hour PM2.5 and PM10 concentrations from 24-hour "
        "lookback sequences, benchmarked against 1-hour persistence. Third, introduce a bounded residual multimodal framework "
        "using ResNet-50 that constrains visual corrections to [-25, +25] micrograms per cubic meter, with an absolute "
        "mathematical fallback when images are missing. Fourth, implement a geodesic inverse-distance-weighted spatial engine "
        "with an empirical 45 km radius limit and distribution-free conformal prediction intervals guaranteeing 90% coverage."
    )
    set_speaker_notes(s5, s5_notes)

    # =========================================================================
    # SLIDE 6: RELATED WORK & LITERATURE GAP ANALYSIS
    # =========================================================================
    print("Formatting Slide 6: Related Work & Literature Gap Analysis...")
    s6 = prs.slides[5]
    clear_body_shapes(s6)
    setup_header(s6, "LITERATURE REVIEW", "Comparative Analysis of Existing Methodologies")
    
    headers = ["Methodology / Model", "Temporal Modeling", "Spatial Awareness", "Multimodal Vision", "Uncertainty Bounds", "Primary Failure Mode"]
    rows = [
        ["ARIMA / SARIMA", "Linear 1D series", "None", "None", "Parametric Gaussian", "Fails on sudden non-linear spikes"],
        ["Random Forest / GBDT", "Tabular features", "Exogenous coordinates", "None", "None", "Treats space as flat static features"],
        ["Vanilla LSTM / GRU", "Non-linear sequences", "Station-specific", "None", "Uncalibrated", "High compute; black-box point predictions"],
        ["Vision CNN / ViT", "None (static)", "None", "Visual Haze Only", "None", "Confounded by fog, glare & lighting shifts"],
        ["Ordinary Kriging / IDW", "None (static)", "Continuous surface", "None", "Semivariogram only", "Over-smooths across sparse networks"],
        ["VayuDrishti (Proposed)", "Dual LSTM + GBDT (t+1)", "Geodesic IDW (Rmax=45km)", "ResNet-50 Bounded Residual", "Split Conformal (90%)", "Eliminates negative transfer & overconfidence"]
    ]
    col_widths = [Inches(1.8), Inches(1.4), Inches(1.6), Inches(1.5), Inches(1.3), Inches(1.4)]
    create_table(s6, Inches(0.5), Inches(1.1), Inches(9.0), Inches(2.7), headers, rows, col_widths)

    cb, tfb = create_card(s6, Inches(0.5), Inches(4.0), Inches(9.0), Inches(1.0), 
                          "Identified Research Gap & Architectural Justification", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tfb, "Existing systems treat air quality prediction as either purely temporal (ignoring spatial dispersion), purely spatial (ignoring lag dynamics), or naively visual (suffering severe negative transfer).", bold_prefix="Disconnected Paradigms:")
    add_bullet(tfb, "VayuDrishti provides the first unified framework coupling next-hour temporal forecasting, bounded visual residuals, geodesic spatial interpolation with a 45 km cutoff, and distribution-free conformal uncertainty.", bold_prefix="Unified Contribution:")

    s6_notes = (
        "Examining the literature reveals clear gaps across existing paradigms. Classical statistical models like ARIMA "
        "cannot model non-linear seasonal interactions. Deep sequence models like LSTM capture temporal dependencies but act "
        "as uncalibrated black boxes. Vision-based CNNs attempt to estimate AQI from photos but collapse during seasonal fog or "
        "sunlight variations. Furthermore, classical GIS interpolation methods like Kriging assume spatial stationarity, which fails "
        "in sparse sensor networks. VayuDrishti bridges these disconnected approaches into a cohesive framework combining dual "
        "temporal forecasting, bounded residual vision fusion, geodesic spatial safeguarding, and conformal prediction."
    )
    set_speaker_notes(s6, s6_notes)

    # =========================================================================
    # SLIDE 7: KEY CONTRIBUTIONS & ARCHITECTURAL INNOVATIONS
    # =========================================================================
    print("Formatting Slide 7: Key Contributions...")
    s7 = prs.slides[6]
    clear_body_shapes(s7)
    setup_header(s7, "CONTRIBUTIONS", "Four Key Architectural & Empirical Innovations")
    
    c1, tf1 = create_card(s7, Inches(0.5), Inches(1.1), Inches(4.35), Inches(1.9), 
                          "1. Bounded Residual Hybrid Visual Fusion", border_color=C_NAVY, bg_color=C_CARD_BG)
    add_bullet(tf1, "Formulated: y = y_tab + m * clamp(delta_vis, -25, +25), eliminating negative transfer (+12.19 ug/m3 error inflation).", bold_prefix="Mathematical Gating:")
    add_bullet(tf1, "Guarantees that when m=0 (no image), the system defaults strictly to pure tabular forecasting accuracy.", bold_prefix="Zero-Fallback Safety:")
    add_bullet(tf1, "Prevents optical weather confounds (fog, humidity, glare) from corrupting ground particulate predictions.", bold_prefix="Domain Invariance:")

    c2, tf2 = create_card(s7, Inches(5.15), Inches(1.1), Inches(4.35), Inches(1.9), 
                          "2. Geodesic Spatial Safeguards & Geometry Diagnostics", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "Replaced flat Euclidean math with great-circle Haversine distance (p=2.0) with an empirical 45 km cutoff radius.", bold_prefix="Geodesic IDW:")
    add_bullet(tf2, "Calculated verified Clark-Evans R clustering: Delhi (0.42), Mumbai (0.56), Kolkata (0.32), Bengaluru (0.45) vs Agra (1.35).", bold_prefix="Clark-Evans Audit:")
    add_bullet(tf2, "Suppresses ungrounded continuous heatmaps in peripheral zones where nearest station distance exceeds 45 km.", bold_prefix="Overconfidence Guard:")

    c3, tf3 = create_card(s7, Inches(0.5), Inches(3.1), Inches(4.35), Inches(1.9), 
                          "3. Distribution-Free Conformal Uncertainty", border_color=C_GREEN, bg_color=C_CARD_BG)
    add_bullet(tf3, "Implemented split conformal prediction using empirical calibration residuals, generating dynamic prediction intervals.", bold_prefix="Conformal Bounds:")
    add_bullet(tf3, "Achieved 90.4% empirical test coverage at 90% nominal confidence without parametric Gaussian assumptions.", bold_prefix="Verified Calibration:")
    add_bullet(tf3, "Dynamically widens uncertainty intervals during volatile inversion spikes and narrows them during stable periods.", bold_prefix="Regime Sensitivity:")

    c4, tf4 = create_card(s7, Inches(5.15), Inches(3.1), Inches(4.35), Inches(1.9), 
                          "4. Multi-City Live Audit & Open Deployment", border_color=C_ACCENT_BLUE, bg_color=C_CARD_BG)
    add_bullet(tf4, "Audited 31 urban locations, registering 85 active CPCB/APPCB stations with continuous live telemetry tracking.", bold_prefix="Multi-City Registry:")
    add_bullet(tf4, "Completed a 14-day field audit (418 pairs across 4 monitors) establishing honest persistence significance (p=0.48/0.62).", bold_prefix="Live Field Audit:")
    add_bullet(tf4, "Delivered an open-source, full-stack Streamlit platform with interactive Leaflet heatmaps and complete diagnostics.", bold_prefix="Open Dashboard:")

    s7_notes = (
        "Our project delivers four verified technical contributions: First, we developed the Bounded Residual Hybrid for "
        "multimodal fusion. By clamping visual updates within [-25, +25] micrograms per cubic meter and enforcing a zero-fallback "
        "flag, we prevent catastrophic negative transfer. Second, we established a rigorous spatial safeguarding protocol using "
        "Geodesic IDW and Clark-Evans spatial statistics, proving that Indian monitoring networks are severely clustered—for example, "
        "R equals 0.42 in Delhi and 0.32 in Kolkata—and enforcing a 45 km reliability boundary. Third, we integrated distribution-free "
        "split conformal prediction, achieving 90.4% empirical coverage at 90% confidence. Fourth, we deployed an end-to-end, "
        "6-module interactive Streamlit application serving real-time telemetry, forecasts, and spatial interpolations across 31 cities."
    )
    set_speaker_notes(s7, s7_notes)

    # =========================================================================
    # SLIDE 8: SYSTEM ARCHITECTURE
    # =========================================================================
    print("Formatting Slide 8: System Architecture...")
    s8 = prs.slides[7]
    clear_body_shapes(s8)
    setup_header(s8, "SYSTEM ARCHITECTURE", "End-to-End Pipeline & Information Flow")
    
    # Left Side: 3 Pipeline Cards
    c1, tf1 = create_card(s8, Inches(0.5), Inches(1.1), Inches(4.2), Inches(1.25), 
                          "1. Data Ingestion & Feature Engineering Tier", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf1, "OpenAQ v3 REST client ingests continuous hourly criteria pollutants and atmospheric vectors.", bold_prefix="Telemetry Stream:")
    add_bullet(tf1, "Forward imputation (<=3h) + median backfill; cyclic trigonometric temporal encodings.", bold_prefix="Preprocessing:")

    c2, tf2 = create_card(s8, Inches(0.5), Inches(2.45), Inches(4.2), Inches(1.35), 
                          "2. Dual Predictive & Bounded Fusion Engine", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "LSTM (27 channels x 24h) and LightGBM (49 lag/rolling features) predict strictly t+1.", bold_prefix="Dual Predictors:")
    add_bullet(tf2, "ResNet-50 visual backbone extracts 2048-dim features; clamped residual delta in [-25, +25].", bold_prefix="Bounded Vision:")

    c3, tf3 = create_card(s8, Inches(0.5), Inches(3.9), Inches(4.2), Inches(1.1), 
                          "3. Spatial Engine & Conformal Presentation", border_color=C_GREEN, bg_color=C_LIGHT_BLUE)
    add_bullet(tf3, "Geodesic IDW (p=2.0) interpolates continuous surface up to 45 km with Clark-Evans diagnostics.", bold_prefix="Spatial Guard:")
    add_bullet(tf3, "Split conformal prediction delivers 90% confidence intervals to Streamlit UI.", bold_prefix="Conformal Service:")

    # Right Side: Embed Architecture Diagram
    arch_fig = FIG_DIR / "fig_arch_overall.png"
    if arch_fig.exists():
        s8.shapes.add_picture(str(arch_fig), Inches(4.9), Inches(1.1), Inches(4.6), Inches(3.9))

    s8_notes = (
        "This slide illustrates VayuDrishti's end-to-end architecture. The pipeline begins on the left with live telemetry "
        "ingestion from OpenAQ v3, accompanied by rolling feature extraction and forward imputation. The processed streams "
        "enter our dual predictive engine: a 2-layer LSTM processing 27 sequential channels, and an adapted LightGBM regressor "
        "utilizing 49 engineered lag and rolling statistics. When camera imagery is available, a pretrained ResNet-50 visual "
        "backbone extracts atmospheric extinction features, which are bounded within [-25, +25] and added to the tabular forecast. "
        "Finally, the spatial engine computes geodesic inverse-distance-weighted surfaces with a 45 km radius cutoff, and the "
        "conformal engine wraps point predictions in calibrated 90% confidence intervals for display on our Streamlit dashboard."
    )
    set_speaker_notes(s8, s8_notes)

    # =========================================================================
    # SLIDE 9: DASHBOARD MODULES & INTERFACE
    # =========================================================================
    print("Formatting Slide 9: Dashboard Modules & Interface...")
    s9 = prs.slides[8]
    clear_body_shapes(s9)
    setup_header(s9, "IMPLEMENTATION", "Interactive Streamlit Platform Architecture")
    
    # Left Side: 6 Modules in 2 Columns or Compact Stack
    c1, tf1 = create_card(s9, Inches(0.5), Inches(1.1), Inches(4.2), Inches(3.9), 
                          "Six Core Operational Platform Modules", border_color=C_NAVY, bg_color=C_CARD_BG)
    add_bullet(tf1, "Real-time telemetry, CPCB Sub-Index calculators, and 5-tier freshness badges.", bold_prefix="M1: Live Monitor:")
    add_bullet(tf1, "Geodesic IDW Leaflet color-ramps, 45 km radius boundaries, and nearest-station distance.", bold_prefix="M2: Spatial Heatmap:")
    add_bullet(tf1, "31-city registry, Clark-Evans R clustering metric, and station health diagnostics.", bold_prefix="M3: Station Network Audit:")
    add_bullet(tf1, "Side-by-side LightGBM and LSTM next-hour forecasts with 90% conformal bands.", bold_prefix="M4: 1-Hour Forecasting:")
    add_bullet(tf1, "Outdoor image upload, ResNet-50 visual extraction, and bounded residual delta.", bold_prefix="M5: Multimodal Studio:")
    add_bullet(tf1, "Historical ablation benchmarks, feature importances, and group-k-fold cross-validation.", bold_prefix="M6: Model Diagnostics:")

    # Right Side: Embed Dashboard Interface Image
    dash_fig = FIG_DIR / "dashboard_monitoring_interface.png"
    if dash_fig.exists():
        s9.shapes.add_picture(str(dash_fig), Inches(4.9), Inches(1.1), Inches(4.6), Inches(3.9))

    s9_notes = (
        "To operationalize our research, we built a comprehensive, 6-module Streamlit dashboard, as shown on the right. "
        "Module 1 displays real-time telemetry from active CPCB stations. Module 2 renders interactive Leaflet heatmaps using "
        "our geodesic IDW engine with strict 45 km reliability perimeters. Module 3 provides network health audits across 31 cities, "
        "tracking Clark-Evans spatial clustering and observation freshness. Module 4 presents next-hour forecasts with calibrated "
        "conformal uncertainty bounds. Module 5 allows users to test multimodal image uploads, showing the visual feature extraction "
        "and bounded residual adjustment. Finally, Module 6 provides transparency into model training history, feature importances, "
        "and historical benchmark evaluations."
    )
    set_speaker_notes(s9, s9_notes)

    # =========================================================================
    # SLIDE 10: ALGORITHMS & IMPLEMENTATION STACK
    # =========================================================================
    print("Formatting Slide 10: Algorithms & Implementation Stack...")
    s10 = prs.slides[9]
    clear_body_shapes(s10)
    setup_header(s10, "METHODOLOGY", "Mathematical Formulations & Technology Stack")
    
    # Left Column: Mathematical Formulations
    c1, tf1 = create_card(s10, Inches(0.5), Inches(1.1), Inches(4.35), Inches(3.9), 
                          "Core Mathematical Formulations", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf1, "Z_hat(x) = sum(w_i * Z_i) / sum(w_i), where w_i = d(x, x_i)^(-p) with p=2.0 and d <= 45 km cutoff.", bold_prefix="Geodesic IDW:")
    add_bullet(tf1, "R = r_A / r_E = (mean nearest-neighbor distance) / (0.5 / sqrt(density)). R<1 indicates spatial clustering; R>1 indicates dispersion.", bold_prefix="Clark-Evans Metric:")
    add_bullet(tf1, "y_final = y_tabular + m * clamp(delta_visual, -25, +25), where m in {0, 1} guarantees zero-fallback when camera input is absent.", bold_prefix="Bounded Residual Fusion:")
    add_bullet(tf1, "C(x) = [y_hat - q_(1-alpha), y_hat + q_(1-alpha)], where q is the empirical nonconformity quantile over calibration residuals (alpha=0.10).", bold_prefix="Split Conformal Bounds:")

    # Right Column: Technology Stack & Execution Environment
    c2, tf2 = create_card(s10, Inches(5.15), Inches(1.1), Inches(4.35), Inches(3.9), 
                          "Technology Stack & Hardware Profile", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "Python 3.10, PyTorch (CPU-optimized build), LightGBM, Scikit-Learn, Pandas, NumPy.", bold_prefix="Machine Learning Stack:")
    add_bullet(tf2, "Streamlit, Folium / Leaflet.js, Matplotlib, Seaborn, Geopy (Geodesic distance calculation).", bold_prefix="Web & Geospatial:")
    add_bullet(tf2, "OpenAQ v3 REST API with tenacity exponential backoff, rate limiting, and local disk cache.", bold_prefix="Telemetry Client:")
    add_bullet(tf2, "Intel Core i7-1165G7 @ 2.80GHz (4 cores / 8 threads), Intel Iris Xe Graphics, 16 GB DDR4 RAM. Fully optimized for CPU-only execution without CUDA.", bold_prefix="Hardware Profile:")
    add_bullet(tf2, "Sub-20ms per-step inference latency on CPU, enabling immediate dashboard responsiveness.", bold_prefix="Inference Speed:")

    s10_notes = (
        "On this slide, we present our core mathematical formulations and system stack. First, our spatial interpolation uses "
        "great-circle geodesic distances with power p=2.0 and a 45 km cutoff. Second, the Clark-Evans nearest-neighbor ratio R "
        "quantifies spatial clustering by comparing observed mean nearest-neighbor distance r_A to expected distance r_E under "
        "complete spatial randomness. Third, our bounded residual equation clamps visual adjustments within [-25, +25] micrograms "
        "per cubic meter, with m acting as an image availability mask. Fourth, split conformal prediction computes the nonconformity "
        "quantile over calibration residuals, guaranteeing 90% coverage. All models were developed, trained, and evaluated on a "
        "standard quad-core Intel Core i7 processor with 16 GB RAM, proving our architecture runs efficiently on accessible CPU "
        "hardware without requiring high-end GPUs."
    )
    set_speaker_notes(s10, s10_notes)

    # =========================================================================
    # SLIDE 11: DATASET DESCRIPTIONS & FEATURE MANIFESTS
    # =========================================================================
    print("Formatting Slide 11: Dataset Descriptions & Feature Manifests...")
    s11 = prs.slides[10]
    clear_body_shapes(s11)
    setup_header(s11, "DATASETS", "Empirical Data Corpora & Feature Separation")
    
    headers = ["Corpus", "Timeframe", "Size / Records", "Station Coverage", "Role in VayuDrishti"]
    rows = [
        ["Historical Corpus", "2015 – 2020", "1,373,612 hourly rows", "90 stations (26 cities)", "Offline benchmarks, ablations & seasonal cross-validation"],
        ["Recent Telemetry", "2022 – 2024", "3,348,768 records", "85 active stations", "Model adaptation, tuning & multi-city spatial verification"],
        ["Live Validation", "2024 (14 days)", "418 matched pairs", "4 continuous stations", "Operational audit & persistence hypothesis testing"],
        ["PM25Vision", "2023 – 2024", "199 aligned pairs", "Urban ground cameras", "ResNet-50 visual feature extractor & bounded residual testing"]
    ]
    col_widths = [Inches(1.8), Inches(1.3), Inches(1.8), Inches(1.8), Inches(2.3)]
    create_table(s11, Inches(0.5), Inches(1.1), Inches(9.0), Inches(1.9), headers, rows, col_widths)

    c_feat, tf_feat = create_card(s11, Inches(0.5), Inches(3.15), Inches(9.0), Inches(1.85), 
                                 "Strict Feature Manifest Separation Across Models", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf_feat, "Takes 27 numerical channels across 24 historical timesteps: 6 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3), 8 meteorological variables (temperature, humidity, wind components, pressure, dew point, radiation), 5 temporal cyclic encodings (hour sin/cos, day sin/cos, day of week), and 8 rolling/ratio stats. Predicts strictly t+1.", bold_prefix="LSTM Sequence Architecture (27 Features):")
    add_bullet(tf_feat, "Takes 49 engineered tabular features: current pollutant telemetry, explicit multi-scale temporal lags (1h, 2h, 3h, 6h, 12h, 24h), rolling window statistics (3h, 6h, 12h, 24h means and variances), and diurnal/seasonal interaction terms. Predicts strictly t+1.", bold_prefix="Adapted LightGBM GBDT (49 Features):")

    s11_notes = (
        "Turning to our datasets and feature manifests: VayuDrishti leverages four distinct empirical corpora: First, a 1.37-million-row "
        "historical dataset spanning 90 stations for baseline training. Second, 3.3 million recent telemetry records from 85 stations "
        "for adaptation. Third, a 14-day live audit comprising 418 aligned observation pairs across 4 monitors. Fourth, the PM25Vision "
        "dataset of 199 paired outdoor images and ground readings. Crucially, we maintain strict architectural separation in our feature "
        "manifests: The LSTM model takes 27 numerical channels across a 24-hour sequence to predict next-hour levels. The adapted GBDT "
        "model operates on 49 engineered tabular features, including multi-scale lags up to 24 hours and rolling variances. Both models "
        "strictly forecast the next hour, t+1, rather than a 24-hour horizon."
    )
    set_speaker_notes(s11, s11_notes)

    # =========================================================================
    # SLIDE 12: PARAMETERS & EXPERIMENTAL SETUP
    # =========================================================================
    print("Formatting Slide 12: Parameters & Experimental Setup...")
    s12 = prs.slides[11]
    clear_body_shapes(s12)
    setup_header(s12, "EXPERIMENTAL SETUP", "Hyperparameters, Validation Protocol & Safeguard Rules")
    
    c1, tf1 = create_card(s12, Inches(0.5), Inches(1.1), Inches(2.85), Inches(3.9), 
                          "Model Hyperparameters", border_color=C_NAVY, bg_color=C_CARD_BG)
    add_bullet(tf1, "2 hidden layers (128, 64 units), Dropout=0.20, Adam optimizer, lr=0.001, batch_size=64. Lookback=24h, target=t+1.", bold_prefix="LSTM Config:")
    add_bullet(tf1, "100 trees, max_depth=6, lr=0.05, num_leaves=31, subsample=0.80, objective=regression_l1.", bold_prefix="LightGBM Config:")
    add_bullet(tf1, "Pretrained ImageNet weights, 2048->128->1 linear projection, tanh activation scaled to [-25, +25] ug/m3.", bold_prefix="ResNet-50 Config:")

    c2, tf2 = create_card(s12, Inches(3.55), Inches(1.1), Inches(2.9), Inches(3.9), 
                          "Spatial Setup & Safeguards", border_color=C_TEAL, bg_color=C_LIGHT_BLUE)
    add_bullet(tf2, "Great-circle Haversine distance with inverse power exponent p=2.0.", bold_prefix="Distance Metric:")
    add_bullet(tf2, "R_max = 45 km cutoff radius; queries beyond 45 km trigger an explicit 'Sparse Coverage' warning.", bold_prefix="Cutoff Radius:")
    add_bullet(tf2, "Stations with data age >24h (Offline) are strictly excluded from spatial interpolation calculations.", bold_prefix="Offline Filtering:")
    add_bullet(tf2, "Reports nearest monitor distance and Clark-Evans R for every interpolated coordinate.", bold_prefix="Spatial Transparency:")

    c3, tf3 = create_card(s12, Inches(6.65), Inches(1.1), Inches(2.85), Inches(3.9), 
                          "Validation Methodology", border_color=C_GREEN, bg_color=C_CARD_BG)
    add_bullet(tf3, "Chronological 80/20 train/test splits to prevent future information leakage.", bold_prefix="Temporal Splits:")
    add_bullet(tf3, "Spatial Leave-One-Station-Out (LOSO) cross-validation across all stations in Delhi NCR.", bold_prefix="Spatial Holdout:")
    add_bullet(tf3, "Paired two-tailed Student's t-tests (alpha=0.05) evaluated against 1-hour persistence.", bold_prefix="Hypothesis Tests:")
    add_bullet(tf3, "Split conformal prediction evaluated over 418 held-out calibration residuals.", bold_prefix="Conformal Protocol:")

    s12_notes = (
        "Our experimental design follows strict protocols to ensure scientific reproducibility: Our LSTM employs a 2-layer "
        "architecture with 128 and 64 hidden units, regularized by 20% dropout and trained using Adam. LightGBM is tuned with "
        "100 trees, learning rate 0.05, and L1 loss for robustness against outliers. The visual network utilizes a pretrained "
        "ResNet-50 backbone, with its residual projection clamped to [-25, +25] via a scaled tanh function. For spatial interpolation, "
        "we enforce an empirical 45 km cutoff radius using geodesic distance with power p=2.0. Validation uses temporal train-test "
        "splits to prevent data leakage, and spatial Leave-One-Station-Out evaluation across Delhi NCR. Furthermore, all comparative "
        "gains against persistence are evaluated using paired two-tailed t-tests at the 5% significance level."
    )
    set_speaker_notes(s12, s12_notes)

    # =========================================================================
    # SLIDE 13: EXPERIMENTAL RESULTS & COMPARATIVE ANALYSIS
    # =========================================================================
    print("Formatting Slide 13: Experimental Results...")
    s13 = prs.slides[12]
    clear_body_shapes(s13)
    setup_header(s13, "RESULTS", "Empirical Performance & Benchmark Evaluations")
    
    # Table A: Historical Forecasting Benchmarks
    headers_a = ["Model", "PM2.5 MAE", "PM2.5 RMSE", "PM10 MAE", "PM10 RMSE"]
    rows_a = [
        ["1-Hour Persistence", "14.82", "22.40", "34.15", "51.20"],
        ["Multi-Layer Perceptron (MLP)", "11.20", "18.10", "27.50", "43.80"],
        ["LSTM (27 channels)", "9.85", "16.20", "24.10", "39.50"],
        ["LightGBM GBDT (49 features)", "9.12", "15.04", "22.45", "37.12"],
        ["Robust Weighted Ensemble", "8.94", "14.78", "21.90", "36.45"]
    ]
    col_widths_a = [Inches(1.8), Inches(0.95), Inches(0.95), Inches(0.95), Inches(0.95)]
    create_table(s13, Inches(0.5), Inches(1.1), Inches(5.6), Inches(1.8), headers_a, rows_a, col_widths_a)

    # Embed Actual vs Predicted Figure on Top Right
    pred_fig = FIG_DIR / "actual_vs_predicted.png"
    if pred_fig.exists():
        s13.shapes.add_picture(str(pred_fig), Inches(6.25), Inches(1.1), Inches(3.25), Inches(1.8))

    # Bottom Left: Spatial Holdout Card
    c_spat, tf_spat = create_card(s13, Inches(0.5), Inches(3.0), Inches(4.5), Inches(2.0), 
                                 "Spatial Holdout Validation (Delhi NCR LOSO)", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf_spat, "Nearest-neighbor lookup achieves PM2.5 MAE of 30.06 ug/m3 and PM10 MAE of 70.68 ug/m3.", bold_prefix="1-NN Baseline:")
    add_bullet(tf_spat, "Geodesic IDW (p=2.0) reduces PM2.5 MAE to 24.07 ug/m3 (-20.0%) and PM10 MAE to 52.76 ug/m3 (-25.4%).", bold_prefix="Geodesic IDW Gain:")
    add_bullet(tf_spat, "Spatial gains confirm that multi-station distance-weighted geometry outperforms single-station proximity.", bold_prefix="Validation Insight:")

    # Bottom Right: Live Field Audit Card
    c_live, tf_live = create_card(s13, Inches(5.15), Inches(3.0), Inches(4.35), Inches(2.0), 
                                 "14-Day Field Audit & Statistical Significance", border_color=C_AMBER_BORDER, bg_color=C_AMBER_BG)
    add_bullet(tf_live, "LightGBM MAE 9.64 vs Persistence MAE 9.86 (t = -0.71, p = 0.48).", bold_prefix="PM2.5 Live (418 pairs):")
    add_bullet(tf_live, "LightGBM MAE 22.13 vs Persistence MAE 22.35 (t = -0.50, p = 0.62).", bold_prefix="PM10 Live (418 pairs):")
    add_bullet(tf_live, "Next-hour gains over persistence are statistically non-significant due to extreme autocorrelation (rho > 0.95); reported honestly.", bold_prefix="Honest Finding:")
    add_bullet(tf_live, "Split conformal prediction achieved 90.4% coverage at 90% nominal confidence.", bold_prefix="Conformal Coverage:")

    s13_notes = (
        "Here are our empirical results across all evaluation tiers: On the historical corpus, our robust weighted ensemble "
        "achieves the lowest error, with a PM2.5 MAE of 8.94 and PM10 MAE of 21.90, outperforming MLP and individual models. "
        "In spatial holdout testing across Delhi NCR, Geodesic IDW reduces interpolation error by 20.0% for PM2.5 and 25.4% for "
        "PM10 compared to nearest-neighbor lookup. However, our 14-day live field audit provides a vital scientific insight: while "
        "LightGBM achieves lower absolute errors (9.64 vs 9.86 for PM2.5), paired t-tests yield p-values of 0.48 and 0.62, showing "
        "that next-hour gains over persistence are statistically non-significant due to extreme 1-hour autocorrelation. We report "
        "this honestly as a foundational finding. Finally, our split conformal intervals achieved 90.4% coverage, verifying "
        "distribution-free calibration."
    )
    set_speaker_notes(s13, s13_notes)

    # =========================================================================
    # SLIDE 14: FINDINGS, LIMITATIONS & FUTURE WORK
    # =========================================================================
    print("Formatting Slide 14: Findings, Limitations & Future Work...")
    s14 = prs.slides[13]
    clear_body_shapes(s14)
    setup_header(s14, "DISCUSSION", "Scientific Insights, System Limitations & Future Scope")
    
    # Left Column: Insights & Limitations
    c1, tf1 = create_card(s14, Inches(0.5), Inches(1.1), Inches(4.35), Inches(3.9), 
                          "Key Scientific Insights & Limitations", border_color=C_NAVY, bg_color=C_CARD_BG)
    add_bullet(tf1, "At 1-hour horizons, temporal autocorrelation (rho > 0.95) heavily dominates. ML models provide bounded variance during sudden spikes, but incremental single-step gains over lag-1 persistence are modest.", bold_prefix="Autoregressive Dominance:")
    add_bullet(tf1, "Outdoor imagery decouples from ground mass concentration during winter fog and haze. Without bounded residual gating ([-25, +25]), image fusion causes severe negative transfer.", bold_prefix="Multimodal Decoupling:")
    add_bullet(tf1, "Clark-Evans nearest-neighbor statistics confirm heavy spatial clustering (Delhi R=0.42, Kolkata R=0.32). Interpolation accuracy degrades near urban boundaries where nearest station >30 km.", bold_prefix="Clustered Topology:")
    add_bullet(tf1, "Rigorous spatial leave-one-out validation is currently established in Delhi NCR; multi-city expansion requires denser peripheral monitors.", bold_prefix="Validation Scope:")

    # Right Column: Future Directions
    c2, tf2 = create_card(s14, Inches(5.15), Inches(1.1), Inches(4.35), Inches(3.9), 
                          "Future Research Roadmap", border_color=C_TEAL, bg_color=C_LIGHT_BLUE)
    add_bullet(tf2, "Deploy Spatio-Temporal Graph Neural Networks where edge weights are dynamically modulated by real-time wind speed, direction, and terrain elevation.", bold_prefix="ST-GNNs & Wind Advection:")
    add_bullet(tf2, "Incorporate INSAT-3D and MODIS satellite Aerosol Optical Depth (AOD) products at 1 km resolution to bridge unmonitored peri-urban and rural gaps.", bold_prefix="Satellite AOD Remote Sensing:")
    add_bullet(tf2, "Extend predictive horizons from single-step (t+1) to multi-step recursive horizons (t+3, t+6, t+12, t+24 hours) where machine learning value is magnified.", bold_prefix="Multi-Step Forecasting:")
    add_bullet(tf2, "Develop transfer-learning calibration models to correct baseline drift and humidity bias in dense low-cost IoT sensor deployments.", bold_prefix="Low-Cost IoT Calibration:")

    s14_notes = (
        "This slide summarizes our core scientific findings, limitations, and future directions. Our primary finding is that "
        "high autocorrelation in hourly air quality makes single-step gains over persistence statistically modest, emphasizing "
        "the need to evaluate models on multi-step horizons. Second, unconstrained image fusion is hazardous in hazy winter climates, "
        "validating our bounded residual design. Third, high spatial clustering in Indian cities limits interpolation accuracy near "
        "urban edges. To overcome these limitations, our future roadmap includes: deploying Spatio-Temporal Graph Neural Networks "
        "with dynamic wind advection edges, integrating INSAT-3D satellite Aerosol Optical Depth data for unmonitored expanses, "
        "extending forecast horizons to 6 and 24 hours, and integrating transfer-calibrated low-cost IoT sensor networks."
    )
    set_speaker_notes(s14, s14_notes)

    # =========================================================================
    # SLIDE 15: BIBLIOGRAPHY
    # =========================================================================
    print("Formatting Slide 15: Bibliography...")
    s15 = prs.slides[14]
    
    # Adjust Title 1 and Subtitle 2 or clear
    for shape in s15.shapes:
        if shape.name == 'Title 1':
            shape.left = Inches(0.5)
            shape.top = Inches(0.2)
            shape.width = Inches(7.5)
            shape.height = Inches(0.7)
            tf = shape.text_frame
            p0 = tf.paragraphs[0]
            p0.text = "Bibliography & Literature Foundations"
            p0.font.name = FONT_HEADING
            p0.font.size = Pt(20)
            p0.font.bold = True
            p0.font.color.rgb = C_MAROON
        elif shape.name == 'Subtitle 2':
            # Repurpose subtitle into a clean references card
            shape.left = Inches(0.5)
            shape.top = Inches(0.95)
            shape.width = Inches(9.0)
            shape.height = Inches(4.4)
            tf = shape.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.12)
            tf.margin_top = Inches(0.08)
            tf.margin_right = Inches(0.12)
            
            citations = [
                ("[1]", "CPCB India, 'National Air Quality Index: Technical Report,' Central Pollution Control Board, Ministry of Environment, Forest & Climate Change, New Delhi, 2014."),
                ("[2]", "S. Hochreiter and J. Schmidhuber, 'Long Short-Term Memory,' Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997."),
                ("[3]", "G. Ke, Q. Meng, T. Finley, T. Wang, et al., 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree,' in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 3146–3154."),
                ("[4]", "K. He, X. Zhang, S. Ren, and J. Sun, 'Deep Residual Learning for Image Recognition,' in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778."),
                ("[5]", "P. J. Clark and F. C. Evans, 'Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations,' Ecology, vol. 35, no. 4, pp. 445–453, 1954."),
                ("[6]", "D. Shepard, 'A two-dimensional interpolation function for irregularly-spaced data,' in Proc. 23rd ACM National Conference, 1968, pp. 517–524."),
                ("[7]", "A. N. Angelopoulos and S. Bates, 'A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,' arXiv:2107.07511 [stat.ME], 2021."),
                ("[8]", "OpenAQ Community, 'OpenAQ Air Quality API Documentation (v3),' OpenAQ Data Platform, 2024. [Online]. Available: https://api.openaq.org.")
            ]
            
            p0 = tf.paragraphs[0]
            p0.text = ""
            for idx, (c_num, c_text) in enumerate(citations):
                p = tf.paragraphs[idx] if len(tf.paragraphs) > idx else tf.add_paragraph()
                p.space_after = Pt(4)
                p.space_before = Pt(1)
                r_num = p.add_run()
                r_num.text = c_num + " "
                r_num.font.name = FONT_SUBHEADING
                r_num.font.bold = True
                r_num.font.size = Pt(8.5)
                r_num.font.color.rgb = C_NAVY
                
                r_txt = p.add_run()
                r_txt.text = c_text
                r_txt.font.name = FONT_BODY
                r_txt.font.size = Pt(8.2)
                r_txt.font.color.rgb = C_TEXT_MAIN
            while len(tf.paragraphs) > len(citations):
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)

    s15_notes = (
        "Shown here are the foundational scholarly references supporting VayuDrishti, including the CPCB National "
        "Air Quality Index technical report, Hochreiter and Schmidhuber's foundational LSTM work, Ke et al. for LightGBM, "
        "He et al. for Deep Residual Learning, Clark and Evans' spatial nearest-neighbor clustering analysis, Shepard's "
        "inverse distance weighting formulation, Angelopoulos and Bates on Conformal Prediction, and the OpenAQ v3 API specification."
    )
    set_speaker_notes(s15, s15_notes)

    # =========================================================================
    # SLIDE 16: CLOSING & THANK YOU
    # =========================================================================
    print("Formatting Slide 16: Closing & Thank You...")
    s16 = prs.slides[15]
    clear_body_shapes(s16, keep_names=('TextShape 2', 'Picture 2'))
    
    # Header
    setup_header(s16, "CONCLUSION", "Concluding Remarks, Project Artifacts & Viva Defense")
    
    # 3 Summary Cards
    c1, tf1 = create_card(s16, Inches(0.5), Inches(1.1), Inches(2.85), Inches(3.2), 
                          "Project Achievements", border_color=C_NAVY, bg_color=C_LIGHT_BLUE)
    add_bullet(tf1, "Live ingestion of 85+ active stations across 31 Indian cities via OpenAQ v3.", bold_prefix="Multi-City Feed:")
    add_bullet(tf1, "Dual LSTM (27 channels) and LightGBM (49 features) predicting strictly next hour (t+1).", bold_prefix="Dual Predictors:")
    add_bullet(tf1, "ResNet-50 visual features bounded to [-25, +25] with automatic zero-fallback (m=0).", bold_prefix="Bounded Vision:")
    add_bullet(tf1, "Geodesic IDW (p=2.0) with an empirical 45 km cutoff radius and Clark-Evans spatial audit.", bold_prefix="Spatial Guard:")
    add_bullet(tf1, "Split conformal prediction guaranteeing 90% calibrated coverage.", bold_prefix="Conformal Intervals:")

    c2, tf2 = create_card(s16, Inches(3.55), Inches(1.1), Inches(2.9), Inches(3.2), 
                          "Deliverables & Artifacts", border_color=C_TEAL, bg_color=C_CARD_BG)
    add_bullet(tf2, "Complete modular Python code, unit tests, and reproducible audit scripts.", bold_prefix="Source Codebase:")
    add_bullet(tf2, "Full-stack 6-module interactive application with Leaflet geospatial heatmaps.", bold_prefix="Streamlit Dashboard:")
    add_bullet(tf2, "45-page IEEE-standard comprehensive report with verified tables.", bold_prefix="Capstone Report:")
    add_bullet(tf2, "Trained PyTorch LSTM, LightGBM models, and precomputed conformal quantiles.", bold_prefix="Model Checkpoints:")
    add_bullet(tf2, "Slide audit, viva defense question bank, and figure register.", bold_prefix="Audit Companion:")

    c3, tf3 = create_card(s16, Inches(6.65), Inches(1.1), Inches(2.85), Inches(3.2), 
                          "Project Repositories & Access", border_color=C_GREEN, bg_color=C_LIGHT_BLUE)
    add_bullet(tf3, "github.com/trush/vayu-drishti", bold_prefix="Source Repository:")
    add_bullet(tf3, "multimodal-air-quality-c38w.streamlit.app", bold_prefix="Live Web App:")
    add_bullet(tf3, "Prof. Thota Ramathulsi, SCOPE, VIT-AP University", bold_prefix="Faculty Guide:")
    add_bullet(tf3, "Teliki Rushya Sree, Machireddy Swathi, Thanuja Varakala, Palli Shilpa", bold_prefix="Project Team:")
    add_bullet(tf3, "Intel Core i7-1165G7 CPU / 16 GB RAM (No CUDA / GPU)", bold_prefix="Target Platform:")

    # Bottom Callout Banner
    cb, tfb = create_card(s16, Inches(0.5), Inches(4.45), Inches(9.0), Inches(0.6), 
                          "", border_color=C_NAVY, bg_color=C_NAVY)
    p = tfb.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Thank You! We invite questions, feedback, and academic discussion from the esteemed review committee."
    r.font.name = FONT_SUBHEADING
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = C_WHITE

    s16_notes = (
        "In conclusion, VayuDrishti provides a transparent, academically rigorous, and operationally viable framework "
        "for urban air quality intelligence in India. By acknowledging physical sensor clustering, bounding multimodal "
        "corrections, enforcing spatial distance limits, and reporting honest statistical significance against persistence, "
        "our work sets a reliable standard for environmental computing. Our complete codebase, trained model checkpoints, "
        "and interactive Streamlit dashboard are fully accessible. We thank our supervisor Prof. Thota Ramathulsi and the "
        "review committee for their time and guidance. We are now open for questions and discussion."
    )
    set_speaker_notes(s16, s16_notes)

    # =========================================================================
    # SAVE PRESENTATION
    # =========================================================================
    OUT_PPTX_PRES.parent.mkdir(parents=True, exist_ok=True)
    OUT_PPTX_DOCS.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {OUT_PPTX_PRES}...")
    prs.save(str(OUT_PPTX_PRES))
    
    print(f"Mirroring to {OUT_PPTX_DOCS}...")
    shutil.copy2(str(OUT_PPTX_PRES), str(OUT_PPTX_DOCS))
    
    print(f"Successfully generated presentation ({len(prs.slides)} slides)!")

if __name__ == "__main__":
    build_presentation()
