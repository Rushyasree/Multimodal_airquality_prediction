"""
scripts/build_neat_presentation.py

Generates a beautifully aligned, executive-grade, professional Capstone Final Review presentation:
- Targets:
    presentation/VayuDrishti_Final_Review_Simple_Language.pptx
    docs/VayuDrishti_Final_Review_Simple_Language.pptx
- Adheres strictly to the university template sequence and visual branding.
- Clean typography, TOP-aligned text frames, strictly LEFT-ALIGNED card titles, and balanced margins.
- Sets explicit run-level font sizes on all headers so they render bold, crisp, and prominent.
- Preserves Slide 1 university design (zero amateur boxes).
- Preserves Slide 16 fountain pen graphic unobstructed.
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
OUT_PPTX_PRES = ROOT / "presentation/VayuDrishti_Final_Review_Simple_Language.pptx"
OUT_PPTX_DOCS = ROOT / "docs/VayuDrishti_Final_Review_Simple_Language.pptx"
FIG_DIR = ROOT / "paper/figures"

# Professional executive color palette
C_MAROON = RGBColor(87, 35, 20)      # University Header Maroon #572314
C_RED = RGBColor(211, 47, 47)        # Academic Red #D32F2F
C_NAVY = RGBColor(16, 44, 87)        # Executive Navy #102C57
C_DARK = RGBColor(24, 43, 73)        # Dark Slate #182B49
C_TEAL = RGBColor(14, 116, 144)      # Teal #0E7490
C_ACCENT_BLUE = RGBColor(37, 99, 235)# Blue #2563EB
C_GREEN = RGBColor(22, 101, 52)      # Forest Emerald #166534
C_AMBER = RGBColor(217, 119, 6)      # Amber #D97706

C_WHITE = RGBColor(255, 255, 255)
C_CARD_BG = RGBColor(255, 255, 255)  # Pure White
C_PANEL_BG = RGBColor(248, 250, 252) # Slate 50
C_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
C_TEXT_MAIN = RGBColor(30, 41, 59)   # Slate 800 #1E293B
C_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500 #64748B

FONT_SERIF = "Times New Roman"
FONT_SANS = "Calibri"
FONT_HEADING = "Gill Sans MT"

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
    """Sets up the standard two-tier university header cleanly with run-level font styling."""
    for shape in slide.shapes:
        if shape.name == 'TextShape 1':
            shape.left = Inches(0.4)
            shape.top = Inches(0.10)
            shape.width = Inches(7.6)
            shape.height = Inches(0.46)
            tf = shape.text_frame
            tf.word_wrap = False
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.margin_left = Inches(0)
            tf.margin_top = Inches(0)
            while len(tf.paragraphs) > 1:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
            p = tf.paragraphs[0]
            p.text = ""
            r = p.add_run()
            r.text = section_text
            r.font.name = FONT_SERIF
            r.font.size = Pt(24)
            r.font.bold = True
            r.font.color.rgb = C_MAROON
                
        elif shape.name == 'TextShape 2':
            shape.left = Inches(0.4)
            shape.top = Inches(0.56)
            shape.width = Inches(7.6)
            shape.height = Inches(0.42)
            tf = shape.text_frame
            tf.word_wrap = False
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.margin_left = Inches(0)
            tf.margin_top = Inches(0)
            while len(tf.paragraphs) > 1:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
            p = tf.paragraphs[0]
            p.text = ""
            r = p.add_run()
            r.text = topic_text
            r.font.name = FONT_HEADING
            r.font.size = Pt(16.5)
            r.font.bold = True
            r.font.color.rgb = C_DARK

def create_executive_card(slide, left, top, width, height, title="", subtitle="", accent_color=C_NAVY, bg_color=C_CARD_BG):
    """
    Creates an executive card with:
    - Pure white background with subtle border
    - Top accent stripe (colored 3pt line)
    - TOP-ALIGNED text frame
    - LEFT-ALIGNED title & subtitle
    """
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = C_CARD_BORDER
    card.line.width = Pt(1.0)
    
    # Top accent bar
    bar_height = Inches(0.04)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, bar_height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent_color
    bar.line.fill.background()
    
    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.14)
    tf.margin_right = Inches(0.14)
    tf.margin_top = Inches(0.12)
    tf.margin_bottom = Inches(0.08)
    
    if title:
        p0 = tf.paragraphs[0]
        p0.alignment = PP_ALIGN.LEFT
        p0.text = title
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(11.5)
        p0.font.bold = True
        p0.font.color.rgb = C_NAVY
        p0.space_after = Pt(1)
        p0.space_before = Pt(0)
        
        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.alignment = PP_ALIGN.LEFT
            p_sub.text = subtitle
            p_sub.font.name = FONT_SANS
            p_sub.font.size = Pt(8.8)
            p_sub.font.italic = True
            p_sub.font.color.rgb = C_TEXT_MUTED
            p_sub.space_after = Pt(4)
        else:
            p0.space_after = Pt(4)
            
    return card, tf

def add_clean_bullet(tf, text, bold_prefix="", level=0, pt_size=9.4, space_after=4):
    if len(tf.paragraphs) == 1 and not tf.paragraphs[0].text.strip():
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.level = level
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(space_after)
    p.space_before = Pt(1)
    
    if bold_prefix:
        r_pre = p.add_run()
        r_pre.text = "• " + bold_prefix + " "
        r_pre.font.name = FONT_HEADING
        r_pre.font.bold = True
        r_pre.font.size = Pt(pt_size)
        r_pre.font.color.rgb = C_NAVY
    else:
        r_bullet = p.add_run()
        r_bullet.text = "• "
        r_bullet.font.name = FONT_HEADING
        r_bullet.font.bold = True
        r_bullet.font.size = Pt(pt_size)
        r_bullet.font.color.rgb = C_TEAL
        
    r_body = p.add_run()
    r_body.text = text
    r_body.font.name = FONT_SANS
    r_body.font.size = Pt(pt_size)
    r_body.font.color.rgb = C_TEXT_MAIN

def create_clean_table(slide, left, top, width, height, headers, rows, col_widths=None):
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table
    
    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w
            
    # Headers
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.name = FONT_HEADING
        r.font.bold = True
        r.font.size = Pt(8.8)
        r.font.color.rgb = C_WHITE
        
    # Data Rows
    for r_idx, row_data in enumerate(rows):
        bg = C_WHITE if r_idx % 2 == 0 else RGBColor(248, 250, 252)
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
            r.font.name = FONT_SANS
            r.font.size = Pt(8.2)
            r.font.color.rgb = C_TEXT_MAIN
            if c_idx == 0:
                r.font.bold = True
                
    return table_shape

def build_presentation():
    print(f"Loading template: {TEMPLATE_PPTX}")
    prs = Presentation(str(TEMPLATE_PPTX))
    assert len(prs.slides) == 16, f"Expected 16 slides, got {len(prs.slides)}"

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (University Clean Standard, NO Boxes)
    # =========================================================================
    print("Formatting Slide 1: Title Slide (University Clean Standard)...")
    s1 = prs.slides[0]
    
    for shape in s1.shapes:
        if shape.name == 'TextShape 1':
            shape.left = Inches(0.4)
            shape.top = Inches(0.28)
            shape.width = Inches(7.7)
            shape.height = Inches(1.8)
            tf = shape.text_frame
            tf.word_wrap = True
            
            p0 = tf.paragraphs[0]
            p0.text = "Capstone Final Review"
            p0.font.name = FONT_SERIF
            p0.font.size = Pt(22)
            p0.font.bold = True
            p0.font.color.rgb = C_MAROON
            p0.space_after = Pt(4)
            
            p1 = tf.paragraphs[1] if len(tf.paragraphs) > 1 else tf.add_paragraph()
            p1.text = "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments"
            p1.font.name = FONT_SERIF
            p1.font.size = Pt(22)
            p1.font.bold = True
            p1.font.color.rgb = C_NAVY
            p1.space_after = Pt(4)

            p2 = tf.add_paragraph()
            p2.text = "Air Quality Forecasting and Location-Based Pollution Mapping Using Multiple Data Sources"
            p2.font.name = FONT_SANS
            p2.font.size = Pt(11.5)
            p2.font.italic = True
            p2.font.color.rgb = C_TEAL
            
            while len(tf.paragraphs) > 3:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
                
        elif shape.name == 'TextShape 2':
            shape.left = Inches(0.4)
            shape.top = Inches(2.25)
            shape.width = Inches(8.5)
            shape.height = Inches(0.75)
            tf = shape.text_frame
            tf.word_wrap = True
            
            p0 = tf.paragraphs[0]
            p0.text = ""
            r_sup = p0.add_run()
            r_sup.text = "Supervised By: "
            r_sup.font.name = FONT_HEADING
            r_sup.font.size = Pt(13.5)
            r_sup.font.bold = True
            r_sup.font.color.rgb = C_RED
            
            r_name = p0.add_run()
            r_name.text = "Prof. Thota Ramathulsi"
            r_name.font.name = FONT_HEADING
            r_name.font.size = Pt(13.5)
            r_name.font.bold = True
            r_name.font.color.rgb = C_NAVY
            
            p1 = tf.add_paragraph()
            p1.text = "Assistant Professor, School of Computer Science and Engineering (SCOPE), VIT-AP University"
            p1.font.name = FONT_SANS
            p1.font.size = Pt(10)
            p1.font.color.rgb = C_TEXT_MUTED
            
            while len(tf.paragraphs) > 2:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
                
        elif shape.name == 'CustomShape 3':
            shape.left = Inches(0.4)
            shape.top = Inches(3.08)
            shape.width = Inches(5.4)
            shape.height = Inches(2.15)
            shape.fill.background()
            shape.line.fill.background()
            
            tf = shape.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0)
            tf.margin_top = Inches(0)
            
            p0 = tf.paragraphs[0]
            p0.text = ""
            r_cap = p0.add_run()
            r_cap.text = "CAPSTONE ID: "
            r_cap.font.name = FONT_HEADING
            r_cap.font.size = Pt(12)
            r_cap.font.bold = True
            r_cap.font.color.rgb = C_RED
            
            r_id = p0.add_run()
            r_id.text = "F5815A89"
            r_id.font.name = FONT_SANS
            r_id.font.size = Pt(12)
            r_id.font.bold = True
            r_id.font.color.rgb = C_NAVY
            p0.space_after = Pt(8)
            
            p1 = tf.paragraphs[1] if len(tf.paragraphs) > 1 else tf.add_paragraph()
            p1.text = "Project Team Members:"
            p1.font.name = FONT_HEADING
            p1.font.size = Pt(11.5)
            p1.font.bold = True
            p1.font.color.rgb = C_NAVY
            p1.space_after = Pt(4)
            
            students = [
                ("TELIKI RUSHYA SREE", "23BCE9562"),
                ("MACHIREDDY SWATHI", "23BCE9566"),
                ("THANUJA VARAKALA", "23BCE9608"),
                ("PALLI SHILPA", "23BCE9567")
            ]
            for idx, (s_name, s_reg) in enumerate(students):
                p = tf.paragraphs[idx+2] if len(tf.paragraphs) > idx+2 else tf.add_paragraph()
                p.text = ""
                r_bullet = p.add_run()
                r_bullet.text = "•  "
                r_bullet.font.bold = True
                r_bullet.font.color.rgb = C_RED
                
                r_name = p.add_run()
                r_name.text = f"{s_name}  "
                r_name.font.name = FONT_SANS
                r_name.font.size = Pt(10)
                r_name.font.bold = True
                r_name.font.color.rgb = C_TEXT_MAIN
                
                r_reg = p.add_run()
                r_reg.text = f"({s_reg})"
                r_reg.font.name = FONT_SANS
                r_reg.font.size = Pt(9.5)
                r_reg.font.color.rgb = C_TEAL
                p.space_after = Pt(2.5)
            while len(tf.paragraphs) > len(students) + 2:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)
                
        elif shape.name == 'CustomShape 4':
            shape.left = Inches(6.2)
            shape.top = Inches(3.20)
            shape.width = Inches(3.4)
            shape.height = Inches(1.8)
            shape.fill.background()
            shape.line.fill.background()
            
            tf = shape.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0)
            tf.margin_top = Inches(0)
            
            p0 = tf.paragraphs[0]
            p0.text = "SCOPE"
            p0.font.name = FONT_HEADING
            p0.font.size = Pt(26)
            p0.font.bold = True
            p0.font.color.rgb = C_RED
            p0.space_after = Pt(1)
            
            p1 = tf.paragraphs[1] if len(tf.paragraphs) > 1 else tf.add_paragraph()
            p1.text = "School of Computer Science and Engineering"
            p1.font.name = FONT_SANS
            p1.font.size = Pt(9.5)
            p1.font.color.rgb = C_TEXT_MUTED
            p1.space_after = Pt(4)
            
            p2 = tf.paragraphs[2] if len(tf.paragraphs) > 2 else tf.add_paragraph()
            p2.text = "VIT-AP University,"
            p2.font.name = FONT_HEADING
            p2.font.size = Pt(17)
            p2.font.bold = True
            p2.font.color.rgb = C_NAVY
            
            p3 = tf.paragraphs[3] if len(tf.paragraphs) > 3 else tf.add_paragraph()
            p3.text = "Amaravati - 522237, AP, India"
            p3.font.name = FONT_SANS
            p3.font.size = Pt(10.5)
            p3.font.color.rgb = C_TEXT_MUTED
            
            while len(tf.paragraphs) > 4:
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)

    # Add a thin stylish horizontal accent line below the title block
    line = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(2.14), Inches(8.8), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(226, 232, 240) # Slate 200
    line.line.fill.background()

    s1_notes = (
        "Good morning respected panel members and our guide, Prof. Thota Ramathulsi. "
        "We are team VayuDrishti, consisting of Rushya Sree, Swathi, Thanuja, and Shilpa from Computer Science "
        "and Engineering at VIT-AP University. Today, we are presenting our final capstone project on air quality "
        "prediction for Indian cities. In this project, we built a system that collects live sensor data, predicts "
        "pollution levels for the next hour using machine learning, combines sky images safely to refine predictions, "
        "and estimates pollution at unmonitored locations with clear safety limits."
    )
    set_speaker_notes(s1, s1_notes)

    # =========================================================================
    # SLIDE 2: PRESENTATION OUTLINE
    # =========================================================================
    print("Formatting Slide 2: Presentation Outline (Clean Top-Aligned Cards)...")
    s2 = prs.slides[1]
    clear_body_shapes(s2)
    setup_header(s2, "Presentation Outline", "Table of Contents & Presentation Roadmap")
    
    c1, tf1 = create_executive_card(s2, Inches(0.4), Inches(1.15), Inches(4.4), Inches(3.95), 
                                    "Part I: Background & Research Formulation", 
                                    "Problem background, goals, and literature analysis", accent_color=C_NAVY)
    add_clean_bullet(tf1, "What VayuDrishti is and its 3 working modes.", bold_prefix="03. Project Overview:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "4 practical challenges with Indian air quality monitoring.", bold_prefix="04. Problem Statement:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "4 technical goals and quantifiable deliverables.", bold_prefix="05. Project Objectives:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "How our approach compares to 5 existing methods.", bold_prefix="06. Literature Review:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "Our safe image combination and distance safeguards.", bold_prefix="07. Key Contributions:", pt_size=9.2, space_after=3)
    
    c2, tf2 = create_executive_card(s2, Inches(5.1), Inches(1.15), Inches(4.4), Inches(3.95), 
                                    "Part II: Technical Work, Results & Defense", 
                                    "Architecture, algorithms, live evaluation, and viva", accent_color=C_TEAL)
    add_clean_bullet(tf2, "End-to-end data flow from live sensors to user display.", bold_prefix="08. System Architecture:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "6 interactive modules in our Streamlit web application.", bold_prefix="09. Web Dashboard:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "IDW, Clark-Evans, Conformal intervals & tech stack.", bold_prefix="10. Algorithms Used:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "4 datasets and distinct features for LSTM vs LightGBM.", bold_prefix="11. Datasets & Features:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Hyperparameters, 45 km cutoff & 5 freshness levels.", bold_prefix="12. Setup & Safeguards:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Historical benchmarks, Delhi test & 14-day live audit.", bold_prefix="13. Experimental Results:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Key takeaways, system limits & future research steps.", bold_prefix="14. Findings & Next Steps:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "8 core academic and regulatory references.", bold_prefix="15. References:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Live web app, source code repository & viva defense.", bold_prefix="16. Summary & Q&A:", pt_size=9.2, space_after=3)

    s2_notes = (
        "Here is an overview of what we will cover today. We divided our presentation into two parts. "
        "In the first part, we explain why air quality monitoring in India is difficult, what goals we set, "
        "and what other researchers have done. In the second part, we walk through our system design, the machine "
        "learning models we built, our experimental results on both historical and live data, and the lessons "
        "we learned from this project."
    )
    set_speaker_notes(s2, s2_notes)

    # =========================================================================
    # SLIDE 3: INTRODUCTION & PROJECT OVERVIEW (3 Columns, Left-Aligned, Top-Anchored)
    # =========================================================================
    print("Formatting Slide 3: Introduction & Project Overview...")
    s3 = prs.slides[2]
    clear_body_shapes(s3)
    setup_header(s3, "Introduction", "Project Overview: Three Distinct Operational Modes")
    
    c1, tf1 = create_executive_card(s3, Inches(0.4), Inches(1.15), Inches(2.9), Inches(3.45), 
                                    "Mode 1: Live Sensor Telemetry", "Real-Time Data Collection", accent_color=C_TEAL)
    add_clean_bullet(tf1, "Pulls hourly data from 85+ active CPCB/APPCB stations across 31 Indian cities using OpenAQ v3.", bold_prefix="Live Data Feed:", pt_size=9.4, space_after=5)
    add_clean_bullet(tf1, "Monitors criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3) and local weather conditions.", bold_prefix="Multi-Pollutant:", pt_size=9.4, space_after=5)
    add_clean_bullet(tf1, "Labels readings with 5 freshness tiers (Fresh, Delayed, Stale, Offline, Unknown) to prevent using old data.", bold_prefix="Data Freshness:", pt_size=9.4, space_after=5)

    c2, tf2 = create_executive_card(s3, Inches(3.5), Inches(1.15), Inches(2.9), Inches(3.45), 
                                    "Mode 2: Next-Hour Forecasting", "1-Hour Forward Prediction (t+1)", accent_color=C_NAVY)
    add_clean_bullet(tf2, "Dual model engine: 2-layer LSTM (sequence patterns) and adapted LightGBM (tabular rules).", bold_prefix="Two Models:", pt_size=9.4, space_after=5)
    add_clean_bullet(tf2, "Uses the past 24 hours of sensor history to predict air quality strictly for the next hour (t+1).", bold_prefix="Target Horizon:", pt_size=9.4, space_after=5)
    add_clean_bullet(tf2, "Compares predictions against persistence (assuming next hour equals current hour) for honest testing.", bold_prefix="Honest Baseline:", pt_size=9.4, space_after=5)

    c3, tf3 = create_executive_card(s3, Inches(6.6), Inches(1.15), Inches(2.9), Inches(3.45), 
                                    "Mode 3: Location Mapping", "Spatial Estimation & Safeguards", accent_color=C_ACCENT_BLUE)
    add_clean_bullet(tf3, "Uses Inverse Distance Weighting (IDW) to estimate pollution at coordinates between nearby stations.", bold_prefix="Spatial IDW:", pt_size=9.4, space_after=5)
    add_clean_bullet(tf3, "Stops estimating if the nearest station is farther than 45 km, avoiding wild guesses in empty zones.", bold_prefix="45 km Distance Limit:", pt_size=9.4, space_after=5)
    add_clean_bullet(tf3, "Calculates Clark-Evans R to verify whether stations in a city are bunched together or well distributed.", bold_prefix="Station Clustering:", pt_size=9.4, space_after=5)

    bar_b = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(4.75), Inches(9.1), Inches(0.42))
    bar_b.fill.solid()
    bar_b.fill.fore_color.rgb = C_PANEL_BG
    bar_b.line.color.rgb = C_CARD_BORDER
    tf_b = bar_b.text_frame
    tf_b.word_wrap = True
    p_b = tf_b.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    r_b1 = p_b.add_run()
    r_b1.text = "Core Principle: "
    r_b1.font.name = FONT_HEADING
    r_b1.font.bold = True
    r_b1.font.size = Pt(9.5)
    r_b1.font.color.rgb = C_NAVY
    r_b2 = p_b.add_run()
    r_b2.text = "Live sensor telemetry, predictive forecasts, and spatial interpolations are kept strictly separate so users always know the data source."
    r_b2.font.name = FONT_SANS
    r_b2.font.size = Pt(9.2)
    r_b2.font.color.rgb = C_TEXT_MAIN

    s3_notes = (
        "Air pollution in Indian cities changes rapidly with weather and seasons, but continuous monitoring stations "
        "are expensive and limited in number. VayuDrishti addresses this by doing three specific things: First, it collects "
        "live hourly data from over 85 government stations and clearly labels whether the reading is fresh or delayed. "
        "Second, it uses the last 24 hours of sensor history to predict pollution for the next hour at that station. "
        "Third, if someone asks for pollution at an unmonitored location, it estimates the value using nearby stations, "
        "but stops if the nearest station is farther than 45 km. This keeps our system honest and prevents guessing."
    )
    set_speaker_notes(s3, s3_notes)

    # =========================================================================
    # SLIDE 4: PROBLEM STATEMENT (2x2 Grid)
    # =========================================================================
    print("Formatting Slide 4: Problem Statement...")
    s4 = prs.slides[3]
    clear_body_shapes(s4)
    setup_header(s4, "Introduction", "Problem Statement: Four Real-World Challenges in Indian Cities")
    
    c1, tf1 = create_executive_card(s4, Inches(0.4), Inches(1.15), Inches(4.4), Inches(1.9), 
                                    "1. Limited Stations & Spatial Clustering", "Urban Sensor Sparsity", accent_color=C_RED)
    add_clean_bullet(tf1, "Continuous monitoring stations are expensive (₹1–1.5 crore each), so cities have very few.", bold_prefix="Equipment Cost:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "Stations are heavily clustered in central urban areas, leaving outer suburbs unmonitored.", bold_prefix="Suburban Gaps:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "Clark-Evans spatial tests confirm high clustering in major cities (Kolkata R=0.32, Delhi R=0.42).", bold_prefix="Bunched Monitors:", pt_size=9.2, space_after=3)

    c2, tf2 = create_executive_card(s4, Inches(5.1), Inches(1.15), Inches(4.4), Inches(1.9), 
                                    "2. Delayed Data & Unannounced Outages", "Telemetry Latency & Missingness", accent_color=C_AMBER)
    add_clean_bullet(tf2, "Stations often drop offline due to power cuts, sensor calibration, or internet dropouts.", bold_prefix="Frequent Outages:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Standard dashboards show hours-old readings without warning users that the data is old.", bold_prefix="Missing Tags:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "This misleads citizens into believing air is safe when pollution has already spiked.", bold_prefix="Decision Risks:", pt_size=9.2, space_after=3)

    c3, tf3 = create_executive_card(s4, Inches(0.4), Inches(3.2), Inches(4.4), Inches(1.9), 
                                    "3. Photos Can Confuse Prediction Models", "Optical Decoupling & Negative Transfer", accent_color=C_NAVY)
    add_clean_bullet(tf3, "Photos show visual haze, but sensors measure the actual weight of particles in the air.", bold_prefix="Decoupled Physics:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf3, "During foggy mornings or rainy days, a camera sees white haze even when pollution is moderate.", bold_prefix="Weather Confound:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf3, "Simply feeding images directly into models made predictions worse (+12.19 ug/m3 higher error).", bold_prefix="Negative Transfer:", pt_size=9.2, space_after=3)

    c4, tf4 = create_executive_card(s4, Inches(5.1), Inches(3.2), Inches(4.4), Inches(1.9), 
                                    "4. Misleading Heatmaps in Empty Areas", "Spatial Overconfidence in GIS", accent_color=C_TEAL)
    add_clean_bullet(tf4, "Normal map tools draw smooth colored heatmaps across empty rural areas without sensors.", bold_prefix="Ungrounded Maps:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf4, "This creates false confidence for citizens where no real measurement exists.", bold_prefix="False Precision:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf4, "Systems must clearly show where data is missing instead of making up numbers.", bold_prefix="Safety Boundary:", pt_size=9.2, space_after=3)

    s4_notes = (
        "When working with Indian air quality data, we ran into four main problems. First, monitoring stations "
        "are scarce and clustered in downtown areas, leaving outer suburbs unmonitored. Second, stations often stop "
        "transmitting or report delayed numbers, but normal dashboards don't warn users that the data is old. Third, "
        "using outdoor photos directly made our predictions worse—adding images blindly increased our error by over "
        "12 units because winter fog and camera lighting confuse the model. Fourth, normal map tools draw continuous "
        "colored heatmaps across entire states, giving people false confidence in areas where there are no sensors. "
        "We designed VayuDrishti to directly solve these four issues."
    )
    set_speaker_notes(s4, s4_notes)

    # =========================================================================
    # SLIDE 5: OBJECTIVES & EXPECTED IMPACT
    # =========================================================================
    print("Formatting Slide 5: Objectives...")
    s5 = prs.slides[4]
    clear_body_shapes(s5)
    setup_header(s5, "Introduction", "Project Objectives: Our Four Core Engineering Goals")
    
    c1, tf1 = create_executive_card(s5, Inches(0.4), Inches(1.15), Inches(4.4), Inches(1.9), 
                                    "Goal 1: Reliable Live Data Pipeline", "Automated Multi-City Collection", accent_color=C_NAVY)
    add_clean_bullet(tf1, "Connect to the OpenAQ v3 API to fetch live hourly data for 31 Indian cities.", bold_prefix="31-City Pipeline:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "Classify every reading into 5 freshness tiers (Fresh, Delayed, Stale, Offline, Unknown).", bold_prefix="5 Freshness Tiers:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "Save data locally on disk so the dashboard stays online even during network dropouts.", bold_prefix="Fault Tolerance:", pt_size=9.2, space_after=3)

    c2, tf2 = create_executive_card(s5, Inches(5.1), Inches(1.15), Inches(4.4), Inches(1.9), 
                                    "Goal 2: Accurate Next-Hour Forecasting", "Predictive Modeling (t+1)", accent_color=C_TEAL)
    add_clean_bullet(tf2, "Build two models: a 2-layer LSTM and a tuned LightGBM decision tree model.", bold_prefix="Dual Predictors:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Use the past 24 hours of sensor history to predict pollution for the next hour (t+1).", bold_prefix="24h Lookback:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Benchmark strictly against a 1-hour persistence baseline under real field conditions.", bold_prefix="Honest Baseline:", pt_size=9.2, space_after=3)

    c3, tf3 = create_executive_card(s5, Inches(0.4), Inches(3.2), Inches(4.4), Inches(1.9), 
                                    "Goal 3: Safe Integration of Sky Images", "Bounded Multimodal Residual Fusion", accent_color=C_ACCENT_BLUE)
    add_clean_bullet(tf3, "Use a pretrained ResNet-50 vision model to extract features from outdoor sky photos.", bold_prefix="ResNet-50 Model:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf3, "Limit image corrections to a safe range between -25 and +25 ug/m3 to prevent errors.", bold_prefix="Bounded +/-25 Limit:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf3, "Automatically fall back to pure sensor predictions when no image is uploaded.", bold_prefix="Safe Fallback:", pt_size=9.2, space_after=3)

    c4, tf4 = create_executive_card(s5, Inches(5.1), Inches(3.2), Inches(4.4), Inches(1.9), 
                                    "Goal 4: Safe Location Mapping & Uncertainty", "Spatial Safeguards & Conformal Ranges", accent_color=C_GREEN)
    add_clean_bullet(tf4, "Use Inverse Distance Weighting (IDW) to estimate pollution between nearby stations.", bold_prefix="IDW Mapping:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf4, "Enforce a 45 km cutoff radius to avoid making unsupported estimates in empty areas.", bold_prefix="45 km Distance Limit:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf4, "Provide 90% prediction intervals (Conformal Prediction) so users see the expected range.", bold_prefix="90% Prediction Range:", pt_size=9.2, space_after=3)

    s5_notes = (
        "Based on these challenges, we set four clear project goals: First, build an automated pipeline that pulls "
        "live data across 31 cities and marks data freshness. Second, build two models—an LSTM and a LightGBM—to "
        "forecast next-hour pollution and test them against a realistic 1-hour persistence baseline. Third, safely add "
        "camera photos by capping visual adjustments to plus-or-minus 25 units, while falling back to pure sensor predictions "
        "if no photo is available. Fourth, estimate pollution between stations using Inverse Distance Weighting with a 45 km "
        "cutoff limit, and show 90% confidence ranges instead of just a single number."
    )
    set_speaker_notes(s5, s5_notes)

    # =========================================================================
    # SLIDE 6: RELATED WORK & LITERATURE GAP ANALYSIS
    # =========================================================================
    print("Formatting Slide 6: Related Work...")
    s6 = prs.slides[5]
    clear_body_shapes(s6)
    setup_header(s6, "Background & Related Work", "Related Work: How Existing Approaches Compare to VayuDrishti")
    
    headers = ["Approach / Model", "Time Patterns", "Location Awareness", "Camera Images", "Prediction Range", "Common Limitation"]
    rows = [
        ["ARIMA / Linear Models", "Linear 1-hour only", "None", "None", "Gaussian curve", "Fails during sudden pollution spikes"],
        ["Random Forest / GBDT", "Historical averages", "Static coordinates", "None", "None", "Treats locations as flat numbers"],
        ["Deep Learning (LSTM)", "24-hour sequences", "Single station only", "None", "Single number", "Slow to train; uncalibrated output"],
        ["Camera-Only CNNs", "None (single photo)", "None", "Full image only", "None", "Confused by fog, rain & lighting changes"],
        ["Standard GIS / Kriging", "None (static snapshot)", "Continuous surface", "None", "Variogram only", "Guesses smoothly across empty zones"],
        ["VayuDrishti (Our Work)", "LSTM + LightGBM (t+1)", "IDW with 45 km cutoff", "ResNet-50 (Safe +/-25)", "90% Conformal Range", "Combines time, space & images safely"]
    ]
    col_widths = [Inches(1.8), Inches(1.4), Inches(1.6), Inches(1.5), Inches(1.3), Inches(1.5)]
    create_clean_table(s6, Inches(0.4), Inches(1.12), Inches(9.1), Inches(2.55), headers, rows, col_widths)

    c_gap, tf_gap = create_executive_card(s6, Inches(0.4), Inches(3.85), Inches(9.1), Inches(1.25), 
                                          "The Research Gap We Addressed", 
                                          "Connecting isolated methods with operational safeguards", accent_color=C_NAVY)
    add_clean_bullet(tf_gap, "Existing research typically focuses on only one modality: either time series alone, camera photos alone, or GIS maps alone.", bold_prefix="Disconnected Methods:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf_gap, "VayuDrishti connects all three together into one system, while adding safety checks to protect against bad data or missing sensors.", bold_prefix="Our Integrated Solution:", pt_size=9.2, space_after=3)

    s6_notes = (
        "Looking at previous literature, most researchers focused on only one aspect. Statistical models like ARIMA "
        "only look at time and fail during sudden spikes. Deep learning models like LSTM capture time patterns well, but act as "
        "black boxes and output only a single number without confidence bounds. Computer vision models attempt to guess AQI from "
        "photos alone, but they easily get tricked by morning fog and lighting changes. Finally, GIS tools like Kriging draw "
        "smooth surfaces across hundreds of kilometers without checking station density. VayuDrishti bridges these methods "
        "together, using sensors for time, IDW for space, ResNet for images, and conformal prediction for reliable confidence ranges."
    )
    set_speaker_notes(s6, s6_notes)

    # =========================================================================
    # SLIDE 7: KEY CONTRIBUTIONS & NOVEL IDEAS
    # =========================================================================
    print("Formatting Slide 7: Key Contributions...")
    s7 = prs.slides[6]
    clear_body_shapes(s7)
    setup_header(s7, "Background & Related Work", "Improvements Over Existing Solutions: Four Key Contributions")
    
    c1, tf1 = create_executive_card(s7, Inches(0.4), Inches(1.15), Inches(4.4), Inches(1.9), 
                                    "1. Safe Image Integration (Bounded Residuals)", "Eliminating Negative Transfer", accent_color=C_NAVY)
    add_clean_bullet(tf1, "Instead of letting images change forecasts freely, we capped image adjustments between -25 and +25 ug/m3.", bold_prefix="Bounded Correction:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "If no image is uploaded, the correction is set to zero, so the model relies purely on sensor data.", bold_prefix="Safe Fallback:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf1, "This stops fog, rain, or glare from ruining our predictions (+12.19 ug/m3 error prevented).", bold_prefix="Noise Prevention:", pt_size=9.2, space_after=3)

    c2, tf2 = create_executive_card(s7, Inches(5.1), Inches(1.15), Inches(4.4), Inches(1.9), 
                                    "2. Distance Limits & Station Pattern Checks", "Spatial Safeguards & Geometry Audit", accent_color=C_TEAL)
    add_clean_bullet(tf2, "We used geographic distance (Haversine formula) and set a 45 km cutoff to stop guessing in empty areas.", bold_prefix="45 km Distance Limit:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "We measured station clustering (Clark-Evans R) across 31 cities, showing where stations are bunched together.", bold_prefix="Clustering Audit:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf2, "Identified clustered cities like Kolkata (R=0.32) and Delhi (R=0.42), versus spread-out cities like Agra (R=1.35).", bold_prefix="City Geometry Scores:", pt_size=9.2, space_after=3)

    c3, tf3 = create_executive_card(s7, Inches(0.4), Inches(3.2), Inches(4.4), Inches(1.9), 
                                    "3. Reliable 90% Prediction Ranges", "Distribution-Free Conformal Uncertainty", accent_color=C_GREEN)
    add_clean_bullet(tf3, "Instead of giving a single number, we provide an upper and lower bound using Conformal Prediction.", bold_prefix="Prediction Intervals:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf3, "In our live tests, exactly 90.4% of real observations fell inside our predicted 90% range.", bold_prefix="90.4% Live Accuracy:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf3, "Prediction bands automatically widen during sudden spikes and narrow during stable hours.", bold_prefix="Dynamic Calibration:", pt_size=9.2, space_after=3)

    c4, tf4 = create_executive_card(s7, Inches(5.1), Inches(3.2), Inches(4.4), Inches(1.9), 
                                    "4. Live City Dashboard & Honest Field Testing", "31-City Operational Deployment", accent_color=C_ACCENT_BLUE)
    add_clean_bullet(tf4, "Built an interactive Streamlit app connecting to 85 live stations across 31 Indian cities.", bold_prefix="Live Dashboard:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf4, "Ran a 14-day field test (418 paired readings) to honestly compare our models against persistence.", bold_prefix="14-Day Field Test:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf4, "Demonstrated transparent reporting: acknowledged where ML beats persistence and where it is comparable.", bold_prefix="Academic Honesty:", pt_size=9.2, space_after=3)

    s7_notes = (
        "We highlight four main contributions in our work: First, we solved the photo problem by capping image-based "
        "adjustments to plus-or-minus 25 units, with a safe fallback to sensor data when photos are missing. Second, we added "
        "spatial safeguards using geographic distance and a 45 km cutoff, alongside Clark-Evans clustering numbers that prove "
        "stations in Indian metros are heavily bunched in city centers. Third, we implemented conformal prediction, which gives "
        "users a verified 90% confidence range rather than a misleading single number. Fourth, we deployed an interactive 6-module "
        "dashboard and tested it live for 14 days, verifying our findings with real field data."
    )
    set_speaker_notes(s7, s7_notes)

    # =========================================================================
    # SLIDE 8: SYSTEM ARCHITECTURE & DATA FLOW
    # =========================================================================
    print("Formatting Slide 8: System Architecture...")
    s8 = prs.slides[7]
    clear_body_shapes(s8)
    setup_header(s8, "Proposed Solution", "System Architecture: How Data Flows Through Our System")
    
    c1, tf1 = create_executive_card(s8, Inches(0.4), Inches(1.15), Inches(4.3), Inches(1.22), 
                                    "1. Data Collection & Cleaning", "Ingestion & Missingness Handling", accent_color=C_NAVY)
    add_clean_bullet(tf1, "Collects hourly pollutant and weather data from OpenAQ v3.", bold_prefix="Live Feed:", pt_size=9.2, space_after=2)
    add_clean_bullet(tf1, "Fills short gaps up to 3h and creates cyclical time features (hour, day, month).", bold_prefix="Preprocessing:", pt_size=9.2, space_after=2)

    c2, tf2 = create_executive_card(s8, Inches(0.4), Inches(2.47), Inches(4.3), Inches(1.32), 
                                    "2. Forecasting & Safe Image Combination", "Dual Predictors & Bounded Vision", accent_color=C_TEAL)
    add_clean_bullet(tf2, "LightGBM (49 features) and LSTM (27 channels over 24h) predict next hour (t+1).", bold_prefix="Predictors:", pt_size=9.2, space_after=2)
    add_clean_bullet(tf2, "ResNet-50 extracts photo features, bounded to a safe +/-25 ug/m3 adjustment.", bold_prefix="Vision Head:", pt_size=9.2, space_after=2)

    c3, tf3 = create_executive_card(s8, Inches(0.4), Inches(3.89), Inches(4.3), Inches(1.18), 
                                    "3. Location Mapping & Final Display", "Spatial Safeguards & Web UI", accent_color=C_GREEN)
    add_clean_bullet(tf3, "IDW estimates pollution for any point within 45 km of active stations.", bold_prefix="IDW Mapping:", pt_size=9.2, space_after=2)
    add_clean_bullet(tf3, "Conformal prediction adds 90% prediction bands before showing results on the app.", bold_prefix="Conformal UI:", pt_size=9.2, space_after=2)

    arch_fig = FIG_DIR / "fig_arch_overall.png"
    if arch_fig.exists():
        frame = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.9), Inches(1.15), Inches(4.65), Inches(3.92))
        frame.fill.solid()
        frame.fill.fore_color.rgb = C_WHITE
        frame.line.color.rgb = C_CARD_BORDER
        frame.line.width = Pt(1.0)
        s8.shapes.add_picture(str(arch_fig), Inches(4.95), Inches(1.20), Inches(4.55), Inches(3.82))

    s8_notes = (
        "This diagram shows the complete workflow of VayuDrishti. On the left, we pull live sensor feeds from OpenAQ v3, "
        "clean missing data for up to three hours, and calculate time cycles. Next, the data feeds into our two forecasting models: "
        "LightGBM using 49 tabular features and LSTM using 27 sequential features over 24 hours. When an outdoor image is provided, "
        "ResNet-50 extracts visual features and applies a bounded adjustment. Finally, our spatial engine interpolates values for "
        "locations within 45 km, and our conformal module adds 90% prediction bands before presenting everything on our Streamlit web interface."
    )
    set_speaker_notes(s8, s8_notes)

    # =========================================================================
    # SLIDE 9: DASHBOARD MODULES & INTERFACE
    # =========================================================================
    print("Formatting Slide 9: Dashboard Modules...")
    s9 = prs.slides[8]
    clear_body_shapes(s9)
    setup_header(s9, "Proposed Solution", "Key Components: Six Interactive Dashboard Modules")
    
    c1, tf1 = create_executive_card(s9, Inches(0.4), Inches(1.15), Inches(4.3), Inches(3.92), 
                                    "Six Core Application Modules", "Streamlit Web Platform Features", accent_color=C_NAVY)
    add_clean_bullet(tf1, "Shows current readings, AQI sub-indices, and data freshness badges for 85+ stations.", bold_prefix="M1: Live Monitor:", pt_size=9.2, space_after=4)
    add_clean_bullet(tf1, "Interactive colored map showing pollution between stations within 45 km.", bold_prefix="M2: City Heatmap:", pt_size=9.2, space_after=4)
    add_clean_bullet(tf1, "Audits 31 cities, tracking active stations, clustering scores (R), and offline monitors.", bold_prefix="M3: Station Audit:", pt_size=9.2, space_after=4)
    add_clean_bullet(tf1, "Next-hour predictions from LSTM and LightGBM with 90% confidence ranges.", bold_prefix="M4: 1-Hour Forecast:", pt_size=9.2, space_after=4)
    add_clean_bullet(tf1, "Upload outdoor sky photos to inspect visual features and bounded adjustments.", bold_prefix="M5: Camera Studio:", pt_size=9.2, space_after=4)
    add_clean_bullet(tf1, "Displays historical training charts, feature importance rankings, and test results.", bold_prefix="M6: Model Diagnostics:", pt_size=9.2, space_after=4)

    dash_fig = FIG_DIR / "dashboard_monitoring_interface.png"
    if dash_fig.exists():
        frame = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.9), Inches(1.15), Inches(4.65), Inches(3.92))
        frame.fill.solid()
        frame.fill.fore_color.rgb = C_WHITE
        frame.line.color.rgb = C_CARD_BORDER
        frame.line.width = Pt(1.0)
        s9.shapes.add_picture(str(dash_fig), Inches(4.95), Inches(1.20), Inches(4.55), Inches(3.82))

    s9_notes = (
        "To make our research easy to use, we built a 6-module interactive dashboard using Streamlit, shown on the right. "
        "Module 1 displays live station readings and highlights if data is fresh or delayed. Module 2 renders interactive maps "
        "showing interpolated pollution within 45 km. Module 3 audits monitoring stations across 31 cities, tracking which monitors "
        "are active or offline. Module 4 displays our next-hour forecasts alongside 90% confidence bands. Module 5 allows testing "
        "image uploads to inspect visual adjustments. And Module 6 provides full transparency into our model training metrics "
        "and feature importances."
    )
    set_speaker_notes(s9, s9_notes)

    # =========================================================================
    # SLIDE 10: ALGORITHMS & IMPLEMENTATION DETAILS
    # =========================================================================
    print("Formatting Slide 10: Algorithms & Implementation...")
    s10 = prs.slides[9]
    clear_body_shapes(s10)
    setup_header(s10, "Proposed Solution", "Algorithms & Technologies: Key Formulas & Software Tools Used")
    
    c1, tf1 = create_executive_card(s10, Inches(0.4), Inches(1.15), Inches(4.4), Inches(3.92), 
                                    "Core Mathematical Formulations", "Explained simply with operational rules", accent_color=C_NAVY)
    add_clean_bullet(tf1, "Averages nearby station readings using geographic distance. Closer stations have much higher influence than far ones (power = 2.0, max radius = 45 km).", bold_prefix="1. Inverse Distance Weighting:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Compares average station distance to a random spread. R < 1 means stations are bunched together; R > 1 means they are evenly spread out.", bold_prefix="2. Clark-Evans Clustering (R):", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Final Prediction = Sensor Forecast + Image Adjustment. Image adjustment is clamped between -25 and +25. If no photo is uploaded, adjustment is 0.", bold_prefix="3. Bounded Image Adjustment:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Calculates an error margin from test residuals so that 90% of real future values fall within [Prediction - margin, Prediction + margin].", bold_prefix="4. Conformal Prediction Interval:", pt_size=9.3, space_after=4)

    c2, tf2 = create_executive_card(s10, Inches(5.1), Inches(1.15), Inches(4.4), Inches(3.92), 
                                    "Software & Hardware Setup", "Accessible CPU implementation", accent_color=C_TEAL)
    add_clean_bullet(tf2, "Python 3.10, PyTorch (CPU-optimized build), LightGBM, Scikit-Learn, Pandas, NumPy.", bold_prefix="Machine Learning Stack:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Streamlit, Folium / Leaflet maps, Geopy (distance calculation).", bold_prefix="Web & Geospatial:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "OpenAQ v3 REST API with automatic retries and local disk caching.", bold_prefix="Data Ingestion Client:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Runs on an Intel Core i7-1165G7 processor (4 cores, 8 threads) with 16 GB RAM. Fully CPU-based, with no GPU required.", bold_prefix="Hardware Environment:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Prediction takes under 20 milliseconds per sample on standard CPU, ensuring fast dashboard response.", bold_prefix="Inference Speed:", pt_size=9.3, space_after=4)

    s10_notes = (
        "On this slide, we summarize the four core formulas and the technologies we used. First, IDW calculates a weighted "
        "average where closer stations matter much more than far ones. Second, the Clark-Evans ratio R measures whether stations "
        "are bunched together or spread out evenly. Third, our bounded fusion formula ensures that image adjustments cannot "
        "exceed plus-or-minus 25 units. Fourth, conformal prediction calculates an error margin from test residuals to guarantee "
        "90% coverage. Everything was built in Python using PyTorch, LightGBM, and Streamlit, running completely on a standard "
        "quad-core Intel Core i7 laptop without needing a dedicated GPU."
    )
    set_speaker_notes(s10, s10_notes)

    # =========================================================================
    # SLIDE 11: DATASETS USED & INPUT FEATURES
    # =========================================================================
    print("Formatting Slide 11: Datasets Used & Input Features...")
    s11 = prs.slides[10]
    clear_body_shapes(s11)
    setup_header(s11, "Implementation and Results", "Dataset Descriptions: Data Sources & Distinct Input Features")
    
    headers = ["Dataset Name", "Period", "Records / Rows", "Number of Stations", "Role in Project"]
    rows = [
        ["Historical Dataset", "2015 – 2020", "1,373,612 rows", "90 stations (26 cities)", "Main training, baseline comparisons & cross-validation"],
        ["Recent Telemetry", "2022 – 2024", "3,348,768 rows", "85 active stations", "Model fine-tuning & multi-city station audit"],
        ["Live Testing Set", "2024 (14 days)", "418 paired rows", "4 continuous stations", "Real-world validation against 1-hour persistence"],
        ["PM25Vision (Photos)", "2023 – 2024", "199 photo-sensor pairs", "Outdoor ground cameras", "Training ResNet-50 visual feature extractor"]
    ]
    col_widths = [Inches(1.8), Inches(1.3), Inches(1.8), Inches(1.8), Inches(2.4)]
    create_clean_table(s11, Inches(0.4), Inches(1.12), Inches(9.1), Inches(1.85), headers, rows, col_widths)

    c_feat, tf_feat = create_executive_card(s11, Inches(0.4), Inches(3.1), Inches(9.1), Inches(1.95), 
                                           "Distinct Features for Each Model (Predicting Next Hour t+1)", 
                                           "Strict architectural separation between sequence channels and tabular lags", accent_color=C_NAVY)
    add_clean_bullet(tf_feat, "Uses 27 continuous channels across 24 historical hours: 6 pollutants (PM2.5, PM10, NO2, SO2, CO, O3), 8 weather factors (temperature, humidity, wind, pressure, solar radiation, etc.), 5 cyclical time encodings (hour of day, day of year), and 8 rolling averages and pollutant ratios. Predicts strictly next hour (t+1).", bold_prefix="LSTM Sequence Model (27 Inputs):", pt_size=9.3, space_after=4)
    add_clean_bullet(tf_feat, "Uses 49 tabular features: current pollutant and weather values, explicit historical lags (1h, 2h, 3h, 6h, 12h, 24h), rolling averages and variations over 3h, 6h, 12h, and 24h, and weather-pollution interaction terms. Predicts strictly next hour (t+1).", bold_prefix="LightGBM Tabular Model (49 Features):", pt_size=9.3, space_after=4)

    s11_notes = (
        "Here are the four datasets we used. We trained our primary models on a historical dataset of 1.37 million rows "
        "across 90 stations. We also processed 3.3 million recent rows to verify 85 active stations, collected 418 live readings "
        "over two weeks to test real-world accuracy, and used 199 paired outdoor photos to test camera features. Importantly, "
        "we clearly separated our input features: the LSTM takes 27 sequential measurements across 24 hours, while LightGBM "
        "uses 49 engineered tabular features, including multi-hour lags and rolling averages. Both models are set up strictly "
        "to predict the next hour, t+1."
    )
    set_speaker_notes(s11, s11_notes)

    # =========================================================================
    # SLIDE 12: MODEL SETTINGS & TESTING SETUP
    # =========================================================================
    print("Formatting Slide 12: Model Settings & Testing Setup...")
    s12 = prs.slides[11]
    clear_body_shapes(s12)
    setup_header(s12, "Implementation and Results", "Parameters Descriptions: Hyperparameters, Safeguards & Testing Rules")
    
    c1, tf1 = create_executive_card(s12, Inches(0.4), Inches(1.15), Inches(2.9), Inches(3.92), 
                                    "Model Configurations", "Architecture & Tuning", accent_color=C_NAVY)
    add_clean_bullet(tf1, "2 layers (128 & 64 units), 20% dropout, Adam optimizer (lr=0.001), 24h input, predicting t+1.", bold_prefix="LSTM Config:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "100 trees, max depth 6, lr 0.05, 31 leaves per tree, trained with L1 loss for outlier robustness.", bold_prefix="LightGBM Config:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Pretrained ImageNet weights, 2048-dim feature pooling, projected to a safe [-25, +25] ug/m3 range.", bold_prefix="ResNet-50 Config:", pt_size=9.3, space_after=4)

    c2, tf2 = create_executive_card(s12, Inches(3.5), Inches(1.15), Inches(2.9), Inches(3.92), 
                                    "Spatial Settings & Rules", "Geodesic Constraints", accent_color=C_TEAL)
    add_clean_bullet(tf2, "Haversine great-circle formula calculating true geographic distance.", bold_prefix="Distance Metric:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Queries farther than 45 km from an active station display an 'Insufficient Station Density' notice.", bold_prefix="45 km Cutoff:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Any monitor with data older than 24 hours is automatically excluded from map interpolation.", bold_prefix="Offline Filtering:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Displays distance to the nearest station and city clustering score (R) for every query.", bold_prefix="Transparency:", pt_size=9.3, space_after=4)

    c3, tf3 = create_executive_card(s12, Inches(6.6), Inches(1.15), Inches(2.9), Inches(3.92), 
                                    "Validation Rules", "Evaluation Protocol", accent_color=C_GREEN)
    add_clean_bullet(tf3, "Split data by time: 80% older data for training, 20% newer data for testing (avoids looking into the future).", bold_prefix="Time-Based Splits:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf3, "Tested in Delhi NCR by hiding an entire station from the map to see if neighbors can predict it.", bold_prefix="Spatial Holdout:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf3, "Paired two-tailed Student's t-tests (at p < 0.05) to test if ML truly beats persistence.", bold_prefix="Significance Tests:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf3, "Evaluated conformal coverage over 418 held-out test residuals.", bold_prefix="Conformal Test:", pt_size=9.3, space_after=4)

    s12_notes = (
        "To make sure our results are fair and reproducible, we followed strict testing rules. Our LSTM uses two layers "
        "with 128 and 64 units, while LightGBM uses 100 trees with depth 6. Our ResNet-50 visual model is clamped to adjust "
        "values by no more than plus-or-minus 25 units. For mapping, we use Haversine distance with an inverse-square power "
        "of 2, and we enforce a 45 km maximum radius, automatically excluding stations that have been offline for over 24 hours. "
        "For validation, we strictly split data by time—80% train, 20% test—and ran leave-one-station-out tests in Delhi "
        "to verify spatial predictions."
    )
    set_speaker_notes(s12, s12_notes)

    # =========================================================================
    # SLIDE 13: EXPERIMENTAL RESULTS & LIVE TESTING
    # =========================================================================
    print("Formatting Slide 13: Results...")
    s13 = prs.slides[12]
    clear_body_shapes(s13)
    setup_header(s13, "Implementation and Results", "Experimental Outcomes: Benchmark & Live Evaluation")
    
    headers_a = ["Model", "PM2.5 MAE", "PM2.5 RMSE", "PM10 MAE", "PM10 RMSE"]
    rows_a = [
        ["1-Hour Persistence (Baseline)", "14.82", "22.40", "34.15", "51.20"],
        ["Multi-Layer Perceptron (MLP)", "11.20", "18.10", "27.50", "43.80"],
        ["LSTM (27 features)", "9.85", "16.20", "24.10", "39.50"],
        ["LightGBM (49 features)", "9.12", "15.04", "22.45", "37.12"],
        ["Weighted Ensemble (LSTM+GBDT)", "8.94", "14.78", "21.90", "36.45"]
    ]
    col_widths_a = [Inches(1.8), Inches(0.95), Inches(0.95), Inches(0.95), Inches(0.95)]
    create_clean_table(s13, Inches(0.4), Inches(1.15), Inches(5.6), Inches(1.85), headers_a, rows_a, col_widths_a)

    pred_fig = FIG_DIR / "actual_vs_predicted.png"
    if pred_fig.exists():
        frame = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.2), Inches(1.15), Inches(3.3), Inches(1.85))
        frame.fill.solid()
        frame.fill.fore_color.rgb = C_WHITE
        frame.line.color.rgb = C_CARD_BORDER
        frame.line.width = Pt(1.0)
        s13.shapes.add_picture(str(pred_fig), Inches(6.25), Inches(1.18), Inches(3.2), Inches(1.79))

    c_spat, tf_spat = create_executive_card(s13, Inches(0.4), Inches(3.12), Inches(4.4), Inches(1.95), 
                                           "Spatial Holdout in Delhi NCR (LOSO Test)", 
                                           "Testing on unmonitored station locations", accent_color=C_NAVY)
    add_clean_bullet(tf_spat, "Picking the single closest station gives an error of 30.06 for PM2.5 and 70.68 for PM10.", bold_prefix="Nearest Station Baseline:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf_spat, "IDW weighted average reduces error to 24.07 (-20.0%) for PM2.5 and 52.76 (-25.4%) for PM10.", bold_prefix="IDW Weighted Gain:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf_spat, "Confirms that combining multiple nearby stations gives significantly better estimates than just picking the single closest station.", bold_prefix="Key Takeaway:", pt_size=9.2, space_after=3)

    c_live, tf_live = create_executive_card(s13, Inches(5.1), Inches(3.12), Inches(4.4), Inches(1.95), 
                                           "14-Day Live Test & Honest Findings (418 pairs)", 
                                           "Real-world operational audit across 4 monitors", accent_color=C_AMBER)
    add_clean_bullet(tf_live, "LightGBM MAE was 9.64 vs. Persistence MAE 9.86 (t = -0.71, p = 0.48).", bold_prefix="PM2.5 Live (418 pairs):", pt_size=9.2, space_after=3)
    add_clean_bullet(tf_live, "LightGBM MAE was 22.13 vs. Persistence MAE 22.35 (t = -0.50, p = 0.62).", bold_prefix="PM10 Live (418 pairs):", pt_size=9.2, space_after=3)
    add_clean_bullet(tf_live, "Consecutive hourly readings are very similar, so persistence is already strong. ML helps most during sudden weather changes.", bold_prefix="Honest Finding:", pt_size=9.2, space_after=3)
    add_clean_bullet(tf_live, "Exactly 90.4% of real observations fell inside our 90% predicted range.", bold_prefix="Conformal Coverage:", pt_size=9.2, space_after=3)

    s13_notes = (
        "Here are our key experimental results. On historical data, combining LSTM and LightGBM gave the lowest errors—an "
        "MAE of 8.94 for PM2.5 and 21.90 for PM10, clearly beating simpler baselines. In our spatial test in Delhi, IDW reduced "
        "error by 20% for PM2.5 and 25% for PM10 compared to simply using the nearest station. However, our 14-day live test "
        "revealed an important reality: for a 1-hour forecast, persistence is already strong because consecutive hourly readings "
        "are very similar. LightGBM had slightly lower error—9.64 versus 9.86—but the p-value of 0.48 shows the difference is "
        "not statistically significant over a 1-hour window. We report this honestly. Where machine learning truly helps is in "
        "preventing large errors when weather suddenly changes, and our conformal intervals achieved 90.4% coverage."
    )
    set_speaker_notes(s13, s13_notes)

    # =========================================================================
    # SLIDE 14: KEY FINDINGS, LIMITATIONS & NEXT STEPS
    # =========================================================================
    print("Formatting Slide 14: Findings & Next Steps...")
    s14 = prs.slides[13]
    clear_body_shapes(s14)
    setup_header(s14, "Summary", "Key Findings, Current Limitations & Future Roadmap")
    
    c1, tf1 = create_executive_card(s14, Inches(0.4), Inches(1.15), Inches(4.4), Inches(3.92), 
                                    "What We Learned & Current Limitations", "Lessons from empirical field evaluation", accent_color=C_NAVY)
    add_clean_bullet(tf1, "Air quality changes slowly from hour to hour (correlation > 0.95). Machine learning prevents large spike errors, but 1-hour gains over persistence are modest.", bold_prefix="1-Hour Persistence Dominance:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Outdoor photos decouple from ground sensors during winter fog or changing sunlight. Without our +/-25 limit, adding images degrades accuracy.", bold_prefix="Photos Need Limits:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Stations in cities like Kolkata (R=0.32) and Delhi (R=0.42) are clustered downtown. Locations near city boundaries are too far from monitors for reliable interpolation.", bold_prefix="Bunched Station Networks:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf1, "Rigorous spatial leave-one-out testing was only possible in Delhi NCR because other cities have too few monitoring stations.", bold_prefix="Testing Limited by Data:", pt_size=9.3, space_after=4)

    c2, tf2 = create_executive_card(s14, Inches(5.1), Inches(1.15), Inches(4.4), Inches(3.92), 
                                    "Future Research Roadmap", "Concrete next engineering steps", accent_color=C_TEAL)
    add_clean_bullet(tf2, "Connect stations in a graph network and use live wind speed and direction to model how pollution drifts across town.", bold_prefix="Graph Networks & Wind Drifts:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Incorporate satellite Aerosol Optical Depth (AOD) from INSAT-3D to fill in unmonitored rural and suburban areas.", bold_prefix="Satellite Data Integration:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Extend forecasts to 3, 6, 12, and 24 hours ahead, where machine learning clearly outperforms simple persistence.", bold_prefix="Longer Forecast Horizons:", pt_size=9.3, space_after=4)
    add_clean_bullet(tf2, "Use machine learning to calibrate cheap IoT sensors using nearby government stations as reference standards.", bold_prefix="Low-Cost IoT Calibration:", pt_size=9.3, space_after=4)

    s14_notes = (
        "This slide summarizes our main takeaways and next steps. First, because hourly air quality changes gradually, "
        "a 1-hour forecast is heavily dominated by persistence, meaning machine learning should be evaluated over longer "
        "multi-hour horizons. Second, winter fog and changing light easily confuse camera photos, which proves why our bounded "
        "safety limit is essential. Third, heavy station clustering means spatial mapping works well downtown but weakens near "
        "city outskirts. For future work, we plan to use Graph Neural Networks with live wind directions to model how pollution "
        "drifts, integrate INSAT-3D satellite data to cover unmonitored rural zones, extend forecasts to 6 and 24 hours, and "
        "calibrate cheap IoT air sensors."
    )
    set_speaker_notes(s14, s14_notes)

    # =========================================================================
    # SLIDE 15: KEY REFERENCES
    # =========================================================================
    print("Formatting Slide 15: Key References...")
    s15 = prs.slides[14]
    
    for shape in s15.shapes:
        if shape.name == 'Title 1':
            shape.left = Inches(0.4)
            shape.top = Inches(0.18)
            shape.width = Inches(7.5)
            shape.height = Inches(0.55)
            tf = shape.text_frame
            p0 = tf.paragraphs[0]
            p0.text = ""
            r = p0.add_run()
            r.text = "Bibliography"
            r.font.name = FONT_SERIF
            r.font.size = Pt(24)
            r.font.bold = True
            r.font.color.rgb = C_MAROON
            
        elif shape.name == 'Subtitle 2':
            shape.left = Inches(0.4)
            shape.top = Inches(0.85)
            shape.width = Inches(9.1)
            shape.height = Inches(4.3)
            shape.fill.solid()
            shape.fill.fore_color.rgb = C_CARD_BG
            shape.line.color.rgb = C_CARD_BORDER
            shape.line.width = Pt(1.0)
            
            tf = shape.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.margin_left = Inches(0.18)
            tf.margin_top = Inches(0.14)
            tf.margin_right = Inches(0.18)
            
            citations = [
                ("[1]", "CPCB India, 'National Air Quality Index: Technical Report,' Central Pollution Control Board, New Delhi, 2014. [Defines official Indian AQI breakpoints and health categories]"),
                ("[2]", "S. Hochreiter and J. Schmidhuber, 'Long Short-Term Memory,' Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997. [Foundational paper introducing LSTM recurrent networks]"),
                ("[3]", "G. Ke et al., 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree,' in NeurIPS, 2017, pp. 3146–3154. [Introduces fast tree-based gradient boosting used for tabular forecasting]"),
                ("[4]", "K. He, X. Zhang, S. Ren, and J. Sun, 'Deep Residual Learning for Image Recognition,' in Proc. IEEE CVPR, 2016, pp. 770–778. [Introduces ResNet-50 used for visual feature extraction]"),
                ("[5]", "P. J. Clark and F. C. Evans, 'Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations,' Ecology, vol. 35, no. 4, 1954. [Method used to measure station clustering]"),
                ("[6]", "D. Shepard, 'A two-dimensional interpolation function for irregularly-spaced data,' in Proc. 23rd ACM National Conference, 1968. [Defines Inverse Distance Weighting interpolation]"),
                ("[7]", "A. N. Angelopoulos and S. Bates, 'A Gentle Introduction to Conformal Prediction,' arXiv:2107.07511, 2021. [Framework used for calculating 90% prediction intervals]"),
                ("[8]", "OpenAQ Community, 'OpenAQ Air Quality API Documentation (v3),' 2024. [Online API used for our automated live data collection pipeline]")
            ]
            
            p0 = tf.paragraphs[0]
            p0.text = ""
            for idx, (c_num, c_text) in enumerate(citations):
                p = tf.paragraphs[idx] if len(tf.paragraphs) > idx else tf.add_paragraph()
                p.space_after = Pt(4)
                p.space_before = Pt(1)
                r_num = p.add_run()
                r_num.text = c_num + " "
                r_num.font.name = FONT_HEADING
                r_num.font.bold = True
                r_num.font.size = Pt(8.8)
                r_num.font.color.rgb = C_NAVY
                
                r_txt = p.add_run()
                r_txt.text = c_text
                r_txt.font.name = FONT_SANS
                r_txt.font.size = Pt(8.4)
                r_txt.font.color.rgb = C_TEXT_MAIN
            while len(tf.paragraphs) > len(citations):
                p_extra = tf.paragraphs[-1]
                p_extra._p.getparent().remove(p_extra._p)

    s15_notes = (
        "Here are the eight foundational references that guided our system. These include the official CPCB technical report "
        "for Indian AQI standards, the original research papers for our LSTM, LightGBM, and ResNet models, classical papers by "
        "Clark-Evans and Shepard for spatial statistics and IDW, recent work on Conformal Prediction for uncertainty, and the "
        "OpenAQ v3 documentation for our live data pipeline."
    )
    set_speaker_notes(s15, s15_notes)

    # =========================================================================
    # SLIDE 16: CLOSING & THANK YOU (Clean, Dignified, Unobstructed Fountain Pen)
    # =========================================================================
    print("Formatting Slide 16: Closing & Thank You (Unobstructed)...")
    s16 = prs.slides[15]
    clear_body_shapes(s16, keep_names=('TextShape 1', 'TextShape 2', 'Picture 2'))
    
    # Top Header
    title_box = s16.shapes.add_textbox(Inches(0.4), Inches(0.20), Inches(9.1), Inches(0.5))
    tf_t = title_box.text_frame
    tf_t.vertical_anchor = MSO_ANCHOR.TOP
    p_t = tf_t.paragraphs[0]
    p_t.text = ""
    r_t = p_t.add_run()
    r_t.text = "Conclusion & Viva Defense"
    r_t.font.name = FONT_SERIF
    r_t.font.size = Pt(24)
    r_t.font.bold = True
    r_t.font.color.rgb = C_MAROON
    
    # Sleek 2-line summary strip (above the graphic, from top=0.75 to 1.65)
    strip_box = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(0.75), Inches(9.1), Inches(0.85))
    strip_box.fill.solid()
    strip_box.fill.fore_color.rgb = C_PANEL_BG
    strip_box.line.color.rgb = C_CARD_BORDER
    tf_s = strip_box.text_frame
    tf_s.word_wrap = True
    tf_s.vertical_anchor = MSO_ANCHOR.TOP
    tf_s.margin_left = Inches(0.15)
    tf_s.margin_top = Inches(0.10)
    
    p_s1 = tf_s.paragraphs[0]
    p_s1.alignment = PP_ALIGN.LEFT
    r1 = p_s1.add_run()
    r1.text = "Project Summary: "
    r1.font.name = FONT_HEADING
    r1.font.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = C_NAVY
    r2 = p_s1.add_run()
    r2.text = "VayuDrishti integrates 85+ live CPCB stations across 31 cities, dual next-hour forecasting (LSTM & LightGBM), bounded visual integration (+/-25 ug/m3), 45 km spatial safeguards, and 90% conformal intervals."
    r2.font.name = FONT_SANS
    r2.font.size = Pt(9.2)
    r2.font.color.rgb = C_TEXT_MAIN
    
    p_s2 = tf_s.add_paragraph()
    p_s2.space_before = Pt(3)
    r3 = p_s2.add_run()
    r3.text = "Artifacts & Repositories: "
    r3.font.name = FONT_HEADING
    r3.font.bold = True
    r3.font.size = Pt(10)
    r3.font.color.rgb = C_TEAL
    r4 = p_s2.add_run()
    r4.text = "Capstone ID: F5815A89   |   GitHub: github.com/trush/vayu-drishti   |   Live Web App: multimodal-air-quality-c38w.streamlit.app   |   Platform: Intel Core i7 CPU"
    r4.font.name = FONT_SANS
    r4.font.size = Pt(9.2)
    r4.font.color.rgb = C_TEXT_MAIN

    # Bottom Callout (below the fountain pen graphic, at top=4.55)
    callout = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(4.55), Inches(9.1), Inches(0.55))
    callout.fill.solid()
    callout.fill.fore_color.rgb = C_NAVY
    callout.line.fill.background()
    tf_c = callout.text_frame
    p_c = tf_c.paragraphs[0]
    p_c.alignment = PP_ALIGN.CENTER
    r_c = p_c.add_run()
    r_c.text = "Thank You! We welcome questions, suggestions, and feedback from the review committee."
    r_c.font.name = FONT_HEADING
    r_c.font.size = Pt(11.5)
    r_c.font.bold = True
    r_c.font.color.rgb = C_WHITE

    s16_notes = (
        "To conclude, VayuDrishti provides a transparent, practical, and student-built platform for urban air quality "
        "intelligence. By acknowledging real-world constraints—such as clustered monitoring stations, delayed sensor reports, "
        "and the risks of blind image fusion—we designed a system with realistic safeguards and honest evaluation. Our code, "
        "trained models, and live web dashboard are fully accessible. We sincerely thank our supervisor Prof. Thota Ramathulsi "
        "and the review committee for your time and guidance. We are now ready for your questions and feedback."
    )
    set_speaker_notes(s16, s16_notes)

    # Save & Mirror
    OUT_PPTX_PRES.parent.mkdir(parents=True, exist_ok=True)
    OUT_PPTX_DOCS.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving presentation to {OUT_PPTX_PRES}...")
    prs.save(str(OUT_PPTX_PRES))
    
    print(f"Mirroring presentation to {OUT_PPTX_DOCS}...")
    shutil.copy2(str(OUT_PPTX_PRES), str(OUT_PPTX_DOCS))
    
    print(f"Successfully generated neat presentation ({len(prs.slides)} slides)!")

if __name__ == "__main__":
    build_presentation()
