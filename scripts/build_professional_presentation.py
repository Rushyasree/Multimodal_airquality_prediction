"""
scripts/build_professional_presentation.py

Builds the complete visual redesign of the VayuDrishti Capstone Review presentation:
- Target: presentation/VayuDrishti_Final_Review_Professional_Design.pptx
          docs/VayuDrishti_Final_Review_Professional_Design.pptx
- Visual Identity: AI + Air Quality + Environmental Intelligence
- Deep navy/slate canvas (#0B132B, #0F172A), electric cyan & teal highlights,
  translucent elevated glass cards, diverse custom layouts per slide, genuine
  project figures, and natural speaker notes on all 16 slides.
- CAPSTONE ID: F5815A89 and all official academic details preserved.
"""

import os
import shutil
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT_PPTX_PRES = ROOT / "presentation/VayuDrishti_Final_Review_Professional_Design.pptx"
OUT_PPTX_DOCS = ROOT / "docs/VayuDrishti_Final_Review_Professional_Design.pptx"
ASSETS_DIR = ROOT / "assets"
PAPER_FIGS = ROOT / "paper/figures"

# =============================================================================
# COLOR PALETTE: AI + ENVIRONMENTAL INTELLIGENCE (Deep Navy + Cyan / Teal)
# =============================================================================
C_DEEP_NAVY   = RGBColor(11, 19, 43)    # #0B132B Base canvas
C_DARK_SLATE  = RGBColor(15, 23, 42)    # #0F172A Section canvas
C_CARD_BG     = RGBColor(22, 32, 50)    # #162032 Elevated dark glass card
C_CARD_BG_ALT = RGBColor(30, 41, 59)    # #1E293B Secondary card fill
C_CARD_BORDER = RGBColor(51, 65, 85)    # #334155 Subtle structural border
C_BORDER_CYAN = RGBColor(14, 116, 144)  # #0E7490 Glowing cyan border

# Accents
C_CYAN        = RGBColor(56, 189, 248)  # #38BDF8 Electric Cyan
C_TEAL        = RGBColor(45, 212, 191)  # #2DD4BF Bright Teal
C_EMERALD     = RGBColor(16, 185, 129)  # #10B981 Emerald Green (Safe / Success)
C_AMBER       = RGBColor(245, 158, 11)  # #F59E0B Amber (Warning / Stale)
C_RED         = RGBColor(239, 68, 68)   # #EF4444 Crimson (Danger / Offline / ID)
C_PURPLE      = RGBColor(168, 85, 247)  # #A855F7 Purple (Deep Learning)

# Typography
C_WHITE       = RGBColor(255, 255, 255) # Pure White (Headings)
C_ICE_BLUE    = RGBColor(226, 232, 240) # Slate-200 (Readable Body Text)
C_MUTED_SLATE = RGBColor(148, 163, 184) # Slate-400 (Secondary / Labels)

FONT_HEADING = "Segoe UI"
FONT_BODY    = "Calibri"

def set_slide_background(slide, color=C_DEEP_NAVY):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def set_speaker_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text.strip()

def add_header(slide, kicker_text, title_text, slide_num=None):
    """Adds a modern tech header with kicker, main title, accent rule, and VIT logo."""
    # Top Kicker
    kicker_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.18), Inches(6.5), Inches(0.28))
    tf_k = kicker_box.text_frame
    tf_k.margin_left = tf_k.margin_top = tf_k.margin_right = tf_k.margin_bottom = 0
    tf_k.word_wrap = False
    p_k = tf_k.paragraphs[0]
    r_k = p_k.add_run()
    r_k.text = kicker_text.upper()
    r_k.font.name = FONT_HEADING
    r_k.font.size = Pt(10)
    r_k.font.bold = True
    r_k.font.color.rgb = C_CYAN
    
    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.4), Inches(0.44), Inches(6.8), Inches(0.46))
    tf_t = title_box.text_frame
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    r_t = p_t.add_run()
    r_t.text = title_text
    r_t.font.name = FONT_HEADING
    r_t.font.size = Pt(17.5)
    r_t.font.bold = True
    r_t.font.color.rgb = C_WHITE

    # Accent Underline Rule
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(0.96), Inches(9.2), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(30, 58, 138) # Dark blue rule
    line.line.fill.background()

    # White VIT-AP Logo on Top Right
    logo_path = ASSETS_DIR / "vit_logo_white.png"
    if logo_path.exists():
        slide.shapes.add_picture(str(logo_path), Inches(7.6), Inches(0.22), width=Inches(1.8))

    # Slide Number Badge
    if slide_num is not None:
        num_box = slide.shapes.add_textbox(Inches(9.1), Inches(5.20), Inches(0.6), Inches(0.3))
        tf_n = num_box.text_frame
        p_n = tf_n.paragraphs[0]
        p_n.alignment = PP_ALIGN.RIGHT
        r_n = p_n.add_run()
        r_n.text = f"{slide_num:02d}"
        r_n.font.name = FONT_BODY
        r_n.font.size = Pt(9)
        r_n.font.color.rgb = C_MUTED_SLATE

def add_glass_card(slide, left, top, width, height, title="", subtitle="", top_accent=None, border_color=C_CARD_BORDER, bg_color=C_CARD_BG):
    """Creates a modern elevated dark glass card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)

    # Top accent line if requested
    if top_accent is not None:
        stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.04))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = top_accent
        stripe.line.fill.background()

    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.14)
    tf.margin_bottom = Inches(0.14)

    curr_p = tf.paragraphs[0]
    if title:
        curr_p.text = ""
        r_t = curr_p.add_run()
        r_t.text = title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(13)
        r_t.font.bold = True
        r_t.font.color.rgb = C_WHITE
        curr_p.space_after = Pt(2)

        if subtitle:
            p_sub = tf.add_paragraph()
            r_s = p_sub.add_run()
            r_s.text = subtitle
            r_s.font.name = FONT_BODY
            r_s.font.size = Pt(9.5)
            r_s.font.italic = True
            r_s.font.color.rgb = C_CYAN
            p_sub.space_after = Pt(6)

    return card, tf

def add_stat_box(slide, left, top, width, height, num_str, label_str, sub_str="", accent_color=C_CYAN):
    """Creates a high-impact numeric statistic card."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = C_CARD_BG
    box.line.color.rgb = C_CARD_BORDER
    box.line.width = Pt(1)

    # Top accent stripe
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.04))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent_color
    stripe.line.fill.background()

    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.10)
    tf.margin_bottom = Inches(0.08)

    p0 = tf.paragraphs[0]
    r0 = p0.add_run()
    r0.text = num_str
    r0.font.name = FONT_HEADING
    r0.font.size = Pt(23)
    r0.font.bold = True
    r0.font.color.rgb = accent_color
    p0.space_after = Pt(1)

    p1 = tf.add_paragraph()
    r1 = p1.add_run()
    r1.text = label_str
    r1.font.name = FONT_HEADING
    r1.font.size = Pt(10)
    r1.font.bold = True
    r1.font.color.rgb = C_WHITE
    p1.space_after = Pt(1)

    if sub_str:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub_str
        r2.font.name = FONT_BODY
        r2.font.size = Pt(8.5)
        r2.font.color.rgb = C_MUTED_SLATE

    return box

def add_bullet(tf, lead_text, body_text, font_size=9.5, lead_color=C_CYAN, space_after=3.5):
    p = tf.add_paragraph()
    p.space_after = Pt(space_after)
    
    r_bullet = p.add_run()
    r_bullet.text = "•  "
    r_bullet.font.bold = True
    r_bullet.font.color.rgb = lead_color
    
    if lead_text:
        r_lead = p.add_run()
        r_lead.text = f"{lead_text}: "
        r_lead.font.name = FONT_HEADING
        r_lead.font.size = Pt(font_size)
        r_lead.font.bold = True
        r_lead.font.color.rgb = C_WHITE
        
    r_body = p.add_run()
    r_body.text = body_text
    r_body.font.name = FONT_BODY
    r_body.font.size = Pt(font_size)
    r_body.font.color.rgb = C_ICE_BLUE

# =============================================================================
# BUILD PRESENTATION
# =============================================================================
def build_professional_deck():
    prs = Presentation()
    prs.slide_width = Inches(10.0)
    prs.slide_height = Inches(5.625)
    blank_layout = prs.slide_layouts[6]

    print("Building 16-Slide Professional Deck...")

    # =========================================================================
    # SLIDE 1: HERO TITLE SLIDE
    # =========================================================================
    print("Formatting Slide 01: Hero Title...")
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, C_DEEP_NAVY)

    # Atmospheric hero background image
    hero_bg = ASSETS_DIR / "slide1_hero_bg.png"
    if hero_bg.exists():
        s1.shapes.add_picture(str(hero_bg), 0, 0, width=Inches(10.0), height=Inches(5.625))

    # Top White University Logo
    vit_logo = ASSETS_DIR / "vit_logo_white.png"
    if vit_logo.exists():
        s1.shapes.add_picture(str(vit_logo), Inches(7.5), Inches(0.35), width=Inches(2.0))

    # Category Pill Tag
    cat_tag = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(0.38), Inches(2.8), Inches(0.30))
    cat_tag.fill.solid()
    cat_tag.fill.fore_color.rgb = RGBColor(14, 116, 144)
    cat_tag.line.fill.background()
    tf_c = cat_tag.text_frame
    tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
    p_ct = tf_c.paragraphs[0]
    p_ct.alignment = PP_ALIGN.CENTER
    r_ct = p_ct.add_run()
    r_ct.text = "CAPSTONE FINAL REVIEW  •  CSE 2026"
    r_ct.font.name = FONT_HEADING
    r_ct.font.size = Pt(9)
    r_ct.font.bold = True
    r_ct.font.color.rgb = C_WHITE

    # Main Project Title
    title_box = s1.shapes.add_textbox(Inches(0.5), Inches(0.78), Inches(8.8), Inches(1.35))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t1 = tf_t.paragraphs[0]
    r_t1 = p_t1.add_run()
    r_t1.text = "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction"
    r_t1.font.name = FONT_HEADING
    r_t1.font.size = Pt(24)
    r_t1.font.bold = True
    r_t1.font.color.rgb = C_WHITE
    p_t1.space_after = Pt(1)

    p_t2 = tf_t.add_paragraph()
    r_t2 = p_t2.add_run()
    r_t2.text = "in Indian Urban Environments"
    r_t2.font.name = FONT_HEADING
    r_t2.font.size = Pt(22)
    r_t2.font.bold = True
    r_t2.font.color.rgb = C_CYAN
    p_t2.space_after = Pt(2)

    p_t3 = tf_t.add_paragraph()
    r_t3 = p_t3.add_run()
    r_t3.text = "Air Quality Forecasting and Location-Based Pollution Mapping Using Multiple Data Sources"
    r_t3.font.name = FONT_BODY
    r_t3.font.size = Pt(11)
    r_t3.font.italic = True
    r_t3.font.color.rgb = C_ICE_BLUE

    # Thin Divider
    div_l = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(2.30), Inches(9.0), Inches(0.015))
    div_l.fill.solid()
    div_l.fill.fore_color.rgb = RGBColor(56, 189, 248)
    div_l.line.fill.background()

    # Left Metadata Card: Team Roster & Capstone ID
    left_c, tf_l = add_glass_card(s1, Inches(0.5), Inches(2.46), Inches(4.35), Inches(2.74), 
                                  title="", top_accent=C_CYAN, bg_color=RGBColor(15, 23, 42))
    
    # Capstone ID Tag
    p_id = tf_l.paragraphs[0]
    r_id_lbl = p_id.add_run()
    r_id_lbl.text = "CAPSTONE ID: "
    r_id_lbl.font.name = FONT_HEADING
    r_id_lbl.font.size = Pt(11.5)
    r_id_lbl.font.bold = True
    r_id_lbl.font.color.rgb = C_RED
    r_id_val = p_id.add_run()
    r_id_val.text = "F5815A89"
    r_id_val.font.name = FONT_BODY
    r_id_val.font.size = Pt(11.5)
    r_id_val.font.bold = True
    r_id_val.font.color.rgb = C_CYAN
    p_id.space_after = Pt(6)

    p_tm = tf_l.add_paragraph()
    r_tm = p_tm.add_run()
    r_tm.text = "Project Team Members:"
    r_tm.font.name = FONT_HEADING
    r_tm.font.size = Pt(11)
    r_tm.font.bold = True
    r_tm.font.color.rgb = C_WHITE
    p_tm.space_after = Pt(4)

    team = [
        ("TELIKI RUSHYA SREE", "23BCE9562"),
        ("MACHIREDDY SWATHI", "23BCE9566"),
        ("THANUJA VARAKALA", "23BCE9608"),
        ("PALLI SHILPA", "23BCE9567")
    ]
    for s_name, s_reg in team:
        p_st = tf_l.add_paragraph()
        r_b = p_st.add_run()
        r_b.text = "•  "
        r_b.font.bold = True
        r_b.font.color.rgb = C_CYAN
        r_sn = p_st.add_run()
        r_sn.text = f"{s_name}  "
        r_sn.font.name = FONT_BODY
        r_sn.font.size = Pt(9.5)
        r_sn.font.bold = True
        r_sn.font.color.rgb = C_WHITE
        r_sr = p_st.add_run()
        r_sr.text = f"({s_reg})"
        r_sr.font.name = FONT_BODY
        r_sr.font.size = Pt(9)
        r_sr.font.color.rgb = C_TEAL
        p_st.space_after = Pt(2.5)

    # Right Metadata Card: Supervisor & Institution
    right_c, tf_r = add_glass_card(s1, Inches(5.15), Inches(2.46), Inches(4.35), Inches(2.74), 
                                   title="", top_accent=C_TEAL, bg_color=RGBColor(15, 23, 42))
    
    p_sup = tf_r.paragraphs[0]
    r_sp_lbl = p_sup.add_run()
    r_sp_lbl.text = "Project Supervisor:"
    r_sp_lbl.font.name = FONT_HEADING
    r_sp_lbl.font.size = Pt(11)
    r_sp_lbl.font.bold = True
    r_sp_lbl.font.color.rgb = C_TEAL
    p_sup.space_after = Pt(2)

    p_sp_name = tf_r.add_paragraph()
    r_sp_n = p_sp_name.add_run()
    r_sp_n.text = "Prof. Thota Ramathulsi"
    r_sp_n.font.name = FONT_HEADING
    r_sp_n.font.size = Pt(14)
    r_sp_n.font.bold = True
    r_sp_n.font.color.rgb = C_WHITE
    p_sp_name.space_after = Pt(1)

    p_sp_des = tf_r.add_paragraph()
    r_sp_d = p_sp_des.add_run()
    r_sp_d.text = "Assistant Professor, SCOPE, VIT-AP University"
    r_sp_d.font.name = FONT_BODY
    r_sp_d.font.size = Pt(9.5)
    r_sp_d.font.color.rgb = C_MUTED_SLATE
    p_sp_des.space_after = Pt(8)

    # Divider in supervisor card
    p_dept = tf_r.add_paragraph()
    r_dp = p_dept.add_run()
    r_dp.text = "SCOPE — School of Computer Science & Engineering"
    r_dp.font.name = FONT_HEADING
    r_dp.font.size = Pt(11)
    r_dp.font.bold = True
    r_dp.font.color.rgb = C_CYAN
    p_dept.space_after = Pt(2)

    p_uni = tf_r.add_paragraph()
    r_un = p_uni.add_run()
    r_un.text = "VIT-AP University, Amaravati - 522237, AP, India"
    r_un.font.name = FONT_BODY
    r_un.font.size = Pt(9.5)
    r_un.font.color.rgb = C_ICE_BLUE
    p_uni.space_after = Pt(6)

    # Mini map preview badge inside supervisor card
    map_path = ASSETS_DIR / "monitoring_map.png"
    if map_path.exists():
        s1.shapes.add_picture(str(map_path), Inches(7.5), Inches(4.35), width=Inches(1.8))

    s1_notes = (
        "Good morning respected committee members and our project guide, Prof. Thota Ramathulsi. "
        "We are team VayuDrishti, consisting of Rushya Sree, Swathi, Thanuja, and Shilpa from Computer Science "
        "and Engineering at VIT-AP University, presenting our Capstone Project (ID: F5815A89). "
        "In this project, we have designed an integrated multimodal air-quality intelligence system that ingests live "
        "sensor data across 31 Indian cities, predicts next-hour particulate pollution using machine learning, "
        "safely refines predictions with outdoor sky imagery, and maps pollution across unmonitored urban locations with strict safety limits."
    )
    set_speaker_notes(s1, s1_notes)

    # =========================================================================
    # SLIDE 2: PRESENTATION OUTLINE (Visual 2-Phase Roadmap)
    # =========================================================================
    print("Formatting Slide 02: Presentation Roadmap...")
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, C_DEEP_NAVY)
    add_header(s2, "01 / Roadmap", "Presentation Outline & Research Trajectory", 2)

    # Phase 1: Research Foundation
    c1, tf1 = add_glass_card(s2, Inches(0.4), Inches(1.08), Inches(4.45), Inches(4.25),
                             title="Part I: Research Foundation & Formulation",
                             subtitle="Theoretical background, problem definition, and objectives",
                             top_accent=C_CYAN)
    add_bullet(tf1, "03. Project Overview", "Core problem, 3 distinct operational modes, and multimodal intelligence.", 8.8, space_after=3.0)
    add_bullet(tf1, "04. Problem Statement", "Four critical urban monitoring bottlenecks: sparsity, downtime, image bias, and unconstrained maps.", 8.8, space_after=3.0)
    add_bullet(tf1, "05. Project Objectives", "Four quantifiable engineering goals covering data, ML models, and safeguards.", 8.8, space_after=3.0)
    add_bullet(tf1, "06. Literature Review", "Comparative analysis of 5 existing paradigms versus VayuDrishti's bounded safeguards.", 8.8, space_after=3.0)
    add_bullet(tf1, "07. Key Contributions", "Bounded vision residual gate, 45 km spatial cutoff, conformal intervals, and web UI.", 8.8, space_after=3.0)

    # Phase 2: System Development & Results
    c2, tf2 = add_glass_card(s2, Inches(5.15), Inches(1.08), Inches(4.45), Inches(4.25),
                             title="Part II: Engineering, Empirical Results & Viva",
                             subtitle="Architecture, real-time algorithms, benchmark results, and defense",
                             top_accent=C_TEAL)
    add_bullet(tf2, "08. System Pipeline & Flow", "Trace data from OpenAQ live sensors to LSTM/LightGBM, ResNet-50, and UI.", 8.8, space_after=2.0)
    add_bullet(tf2, "09. Web Dashboard (6 Tabs)", "Live telemetry monitor, city heatmap, station audit, forecast, camera, diagnostics.", 8.8, space_after=2.0)
    add_bullet(tf2, "10. Algorithms & Tech Stack", "Formulations for IDW, Clark-Evans clustering, Conformal prediction, CPU setup.", 8.8, space_after=2.0)
    add_bullet(tf2, "11. Datasets & Feature Sets", "2.4M CPCB records, 418 live pairs, 27 sequence vs 49 tabular features.", 8.8, space_after=2.0)
    add_bullet(tf2, "12. Setup & Operational Bounds", "Training hyperparams, chronological splits, 45 km cutoff, 5 freshness tiers.", 8.8, space_after=2.0)
    add_bullet(tf2, "13. Empirical Benchmarks & Audit", "Benchmark tables, parity scatter plots, Delhi NCR holdout (-20%), 14-day live audit.", 8.8, space_after=2.0)
    add_bullet(tf2, "14-16. Findings, Roadmap & Viva", "Key takeaways, ST-GNN roadmap, academic references, and open Q&A defense.", 8.8, space_after=2.0)

    s2_notes = (
        "Here is the roadmap of our presentation. We have structured our defense into two clear parts: "
        "Part I establishes the research foundation, urban challenges, and literature gaps. "
        "Part II presents our engineering implementation, architecture, experimental benchmarks, honest 14-day live audit results, and viva defense."
    )
    set_speaker_notes(s2, s2_notes)

    # =========================================================================
    # SLIDE 3: PROJECT OVERVIEW (Three Connected Working Modes)
    # =========================================================================
    print("Formatting Slide 03: Project Overview...")
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, C_DEEP_NAVY)
    add_header(s3, "02 / Overview", "Three Distinct Operational Working Modes", 3)

    col_w = Inches(2.95)
    gap = Inches(0.18)
    left_base = Inches(0.4)

    # Mode 1 Card
    c1, tf1 = add_glass_card(s3, left_base, Inches(1.15), col_w, Inches(3.40),
                             title="Mode 1: Live Telemetry",
                             subtitle="Real-Time Data Ingestion & Quality Audit",
                             top_accent=C_CYAN)
    add_bullet(tf1, "Live Ingestion", "Automated hourly data ingestion from 85+ active CPCB monitors across 31 Indian cities using OpenAQ v3 API.", 9.2)
    add_bullet(tf1, "Multi-Pollutant", "Monitors 6 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3) plus weather variables.", 9.2)
    add_bullet(tf1, "Freshness Engine", "Enforces 5 explicit freshness categories (Fresh, Delayed, Stale, Offline, Unknown) with true observation ages.", 9.2)

    # Mode 2 Card
    c2, tf2 = add_glass_card(s3, left_base + col_w + gap, Inches(1.15), col_w, Inches(3.40),
                             title="Mode 2: 1-Hour Forecast",
                             subtitle="Dual Next-Hour Prediction (t+1)",
                             top_accent=C_TEAL)
    add_bullet(tf2, "Dual Engines", "Combines 2-layer LSTM sequence pattern learner and tuned LightGBM tabular decision forest.", 9.2)
    add_bullet(tf2, "Prediction Horizon", "Uses 24h rolling lookback window to forecast strictly the next hour (t+1) particulate concentrations.", 9.2)
    add_bullet(tf2, "Calibrated Bounds", "Generates distribution-free 90% conformal prediction intervals for transparent uncertainty quantification.", 9.2)

    # Mode 3 Card
    c3, tf3 = add_glass_card(s3, left_base + (col_w + gap)*2, Inches(1.15), col_w, Inches(3.40),
                             title="Mode 3: Location Mapping",
                             subtitle="Spatial Interpolation with Hard Safeguards",
                             top_accent=C_EMERALD)
    add_bullet(tf3, "Geodesic IDW", "Inverse Distance Weighting estimates air quality at arbitrary unmonitored coordinates between nearby stations.", 9.2)
    add_bullet(tf3, "45 km Distance Limit", "Strictly halts spatial interpolation if nearest active station exceeds 45 km, preventing wild guesses.", 9.2)
    add_bullet(tf3, "Clustering Audit", "Computes Clark-Evans Nearest Neighbor index (R) to detect sensor clustering versus geographic spread.", 9.2)

    # Bottom Core Principle Strip
    p_strip = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4), Inches(4.70), Inches(9.2), Inches(0.52))
    p_strip.fill.solid()
    p_strip.fill.fore_color.rgb = C_CARD_BG_ALT
    p_strip.line.color.rgb = C_BORDER_CYAN
    tf_p = p_strip.text_frame
    tf_p.margin_left = Inches(0.2)
    tf_p.margin_top = Inches(0.08)
    p_core = tf_p.paragraphs[0]
    r_core_lbl = p_core.add_run()
    r_core_lbl.text = "Core Architectural Principle: "
    r_core_lbl.font.name = FONT_HEADING
    r_core_lbl.font.size = Pt(10)
    r_core_lbl.font.bold = True
    r_core_lbl.font.color.rgb = C_CYAN
    r_core_txt = p_core.add_run()
    r_core_txt.text = "Physical sensor observations, machine learning forecasts, and spatial interpolations are kept strictly separate. The system never labels interpolated guesses as real sensor readings."
    r_core_txt.font.name = FONT_BODY
    r_core_txt.font.size = Pt(9.5)
    r_core_txt.font.color.rgb = C_ICE_BLUE

    s3_notes = (
        "Air quality in Indian cities changes rapidly, but continuous physical stations are expensive and scarce. "
        "To solve this practically, VayuDrishti operates across three distinct modes: "
        "First, collecting live sensor data and tracking observation age. "
        "Second, forecasting next-hour pollution levels using machine learning with uncertainty bands. "
        "Third, estimating pollution between nearby stations with strict distance boundaries. "
        "Critically, all three modes remain separated so users always know whether they are looking at real measurements or statistical estimates."
    )
    set_speaker_notes(s3, s3_notes)

    # =========================================================================
    # SLIDE 4: PROBLEM STATEMENT (4 Distinct Problem Cards in 2x2 Grid)
    # =========================================================================
    print("Formatting Slide 04: Problem Statement...")
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, C_DEEP_NAVY)
    add_header(s4, "03 / Problem Statement", "Four Practical Bottlenecks in Urban Air Quality Intelligence", 4)

    card_w = Inches(4.45)
    card_h = Inches(1.95)

    # Problem 1: High Cost & Sparsity
    c1, tf1 = add_glass_card(s4, Inches(0.4), Inches(1.15), card_w, card_h,
                             title="1. Sensor Sparsity & High Capital Costs",
                             subtitle="Continuous reference stations are severely limited",
                             top_accent=C_RED)
    add_bullet(tf1, "Severe Blindspots", "A government CAAQMS monitoring station costs > ₹1.5 Crore. Consequently, only ~85 continuous stations serve 31 major Indian cities.", 8.8, space_after=2.5)
    add_bullet(tf1, "Suburban Voids", "Large residential suburbs, peri-urban corridors, and university zones have zero physical sensor coverage.", 8.8, space_after=2.5)

    # Problem 2: Data Downtime & Missingness
    c2, tf2 = add_glass_card(s4, Inches(5.15), Inches(1.15), card_w, card_h,
                             title="2. Frequent Sensor Downtime & Data Delays",
                             subtitle="Telemetry feeds suffer persistent gaps",
                             top_accent=C_AMBER)
    add_bullet(tf2, "Stale Observations", "Frequent power interruptions, network drops, and maintenance cause >35% of stations to report stale or missing data.", 8.8, space_after=2.5)
    add_bullet(tf2, "Silent Failures", "Most consumer apps silently display hours-old readings as 'current live air quality' without warning citizens.", 8.8, space_after=2.5)

    # Problem 3: Camera Vision Vulnerabilities
    c3, tf3 = add_glass_card(s4, Inches(0.4), Inches(3.20), card_w, card_h,
                             title="3. Camera-Based Prediction Vulnerabilities",
                             subtitle="Unbounded vision models fail in outdoor weather",
                             top_accent=C_PURPLE)
    add_bullet(tf3, "Weather Confusion", "End-to-end vision models confuse morning fog, rain droplets, camera lens glare, and dusk darkness for severe particulate pollution.", 8.8, space_after=2.5)
    add_bullet(tf3, "False Alarms", "Unconstrained neural networks produce massive error spikes (>200 µg/m³) during harmless atmospheric changes.", 8.8, space_after=2.5)

    # Problem 4: Misleading Spatial Extrapolation
    c4, tf4 = add_glass_card(s4, Inches(5.15), Inches(3.20), card_w, card_h,
                             title="4. Misleading Continuous Spatial Extrapolation",
                             subtitle="Naive GIS models interpolate across vast empty zones",
                             top_accent=C_CYAN)
    add_bullet(tf4, "Empty Extrapolation", "Standard GIS systems draw smooth, continuous color gradients across 100+ km gaps where microclimates differ completely.", 8.8, space_after=2.5)
    add_bullet(tf4, "False Reassurance", "Users are shown seemingly precise pollution readings at locations far beyond the valid physical correlation range.", 8.8, space_after=2.5)

    s4_notes = (
        "When designing VayuDrishti, we analyzed four core problems with existing Indian air quality monitoring: "
        "First, continuous stations cost over 1.5 Crore rupees each, leaving huge urban blindspots. "
        "Second, over 35% of registered stations frequently suffer data delays or downtime. "
        "Third, recent deep-learning papers rely purely on outdoor phone photos, but camera models are easily tricked by morning fog and glare. "
        "Fourth, existing mapping tools draw smooth heatmaps across empty zones where no sensors exist. "
        "Our project was engineered specifically to solve these four practical limitations."
    )
    set_speaker_notes(s4, s4_notes)

    # =========================================================================
    # SLIDE 5: PROJECT OBJECTIVES (4-Step Visual Process Timeline)
    # =========================================================================
    print("Formatting Slide 05: Project Objectives...")
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, C_DEEP_NAVY)
    add_header(s5, "04 / Objectives", "Four Core Engineering Goals & Quantifiable Deliverables", 5)

    step_w = Inches(2.18)
    step_gap = Inches(0.16)
    s_top = Inches(1.20)
    s_h = Inches(3.95)

    steps_data = [
        ("01", "Automate Telemetry & Freshness", C_CYAN,
         "OpenAQ v3 API + 5-Tier Engine",
         "Pulls hourly criteria pollutants across 31 cities automatically.",
         "Categorizes data into Fresh (<2h), Delayed (2-6h), Stale (6-24h), Offline (>24h), Unknown.",
         "Eliminates silent data staleness; tracks true observation age."),
        ("02", "Dual 1-Hour Forecasters", C_TEAL,
         "LSTM Sequence + LightGBM Tabular",
         "Forecasts PM2.5 and PM10 concentrations strictly for the next hour (t+1).",
         "Captures recurrent sequential patterns (27 features) and tabular rules (49 features).",
         "Generates distribution-free 90% conformal prediction confidence intervals."),
        ("03", "Bounded Vision & Safeguards", C_PURPLE,
         "ResNet-50 Projector + 45 km Cutoff",
         "Limits image adjustments to a safe range of ±25 µg/m³ around the sensor baseline.",
         "Guarantees automatic zero adjustment (fallback) when sky photos are missing.",
         "Halts spatial IDW interpolation beyond 45 km to avoid unsupported continuous maps."),
        ("04", "Operational Web Dashboard", C_EMERALD,
         "Streamlit Cloud + Folium + SciPy",
         "Provides a public, interactive 6-module intelligence interface deployed 24/7.",
         "Integrates Clark-Evans clustering diagnostics (R) to evaluate station geometry.",
         "Displays transparent model diagnostic benchmarks and residual distributions.")
    ]

    for idx, (num, title, color, tech, p1, p2, p3) in enumerate(steps_data):
        c_left = Inches(0.4) + (step_w + step_gap) * idx
        card, tf = add_glass_card(s5, c_left, s_top, step_w, s_h, title="", top_accent=color)
        
        # Step number pill
        p_num = tf.paragraphs[0]
        r_num = p_num.add_run()
        r_num.text = f"GOAL {num}"
        r_num.font.name = FONT_HEADING
        r_num.font.size = Pt(11)
        r_num.font.bold = True
        r_num.font.color.rgb = color
        p_num.space_after = Pt(2)

        # Title
        p_t = tf.add_paragraph()
        r_t = p_t.add_run()
        r_t.text = title
        r_t.font.name = FONT_HEADING
        r_t.font.size = Pt(11)
        r_t.font.bold = True
        r_t.font.color.rgb = C_WHITE
        p_t.space_after = Pt(4)

        # Tech Badge
        p_tech = tf.add_paragraph()
        r_tc = p_tech.add_run()
        r_tc.text = f"Tech: {tech}"
        r_tc.font.name = FONT_BODY
        r_tc.font.size = Pt(8.5)
        r_tc.font.bold = True
        r_tc.font.color.rgb = C_CYAN
        p_tech.space_after = Pt(6)

        # Bullets
        add_bullet(tf, "", p1, 8.8, color, 3)
        add_bullet(tf, "", p2, 8.8, color, 3)
        add_bullet(tf, "", p3, 8.8, color, 3)

    s5_notes = (
        "Based on these four challenges, we defined four measurable engineering objectives: "
        "First, build an automated live ingestion pipeline that tracks true observation age across 31 cities. "
        "Second, train dual forecasting models—LSTM and LightGBM—with calibrated uncertainty intervals. "
        "Third, design a bounded image fusion mechanism capped at plus-or-minus 25 units, combined with a 45 km spatial cutoff. "
        "Fourth, deploy a responsive 6-module Streamlit web application that runs smoothly on standard CPU hardware."
    )
    set_speaker_notes(s5, s5_notes)

    # =========================================================================
    # SLIDE 6: LITERATURE REVIEW (Comparative Matrix)
    # =========================================================================
    print("Formatting Slide 06: Literature Review Matrix...")
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, C_DEEP_NAVY)
    add_header(s6, "05 / Literature Review", "Comparative Paradigm Analysis: Prior Works vs. VayuDrishti", 6)

    # Table layout
    rows = 7
    cols = 6
    left = Inches(0.4)
    top = Inches(1.15)
    width = Inches(9.2)
    height = Inches(3.95)

    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(1.8) # Paradigm
    table.columns[1].width = Inches(1.4) # Studies
    table.columns[2].width = Inches(1.3) # Modality
    table.columns[3].width = Inches(1.3) # Spatial Bound
    table.columns[4].width = Inches(1.7) # Key Limitation
    table.columns[5].width = Inches(1.7) # VayuDrishti Advantage

    headers = ["Approach / Paradigm", "Key Studies", "Modalities", "Spatial Limits", "Major Vulnerability", "VayuDrishti Distinction"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(14, 116, 144)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        r.font.name = FONT_HEADING
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = C_WHITE

    matrix_data = [
        ("1. Classical Statistical", "Box-Jenkins (1970)\nZhang et al. (2018)", "Sensors Only\n(Single series)", "None (Point-only)", "Cannot capture non-linear weather interactions", "Uses deep LSTM + tuned LightGBM with 49 features"),
        ("2. Standard Deep Learning", "Hochreiter (1997)\nLi et al. (2020)", "Sensors Only\n(Time series)", "None (Point-only)", "Lacks spatial interpolation & image perception", "Integrates multi-station IDW + bounded vision"),
        ("3. Vision-Only Estimation", "DeadCardassian (2024)\nLiu et al. (2021)", "Images Only\n(Outdoor photos)", "None (Camera only)", "Extreme false spikes from morning fog, rain & glare", "Binds image delta to ±25 µg/m³; pure sensor baseline"),
        ("4. Unbounded Fusion", "Baltrušaitis (2018)\nChen et al. (2022)", "Concatenated\nSensor + Image", "Unconstrained", "High risk of image noise corrupting physical sensor data", "Gated residual architecture: zero fallback when m=0"),
        ("5. Continuous Spatial GIS", "Shepard (1968)\nKrige (1951)", "Spatial Sensors\n(Coordinates)", "Unbounded (Global GIS)", "Interpolates over empty zones without monitors", "Hard 45 km cutoff + Clark-Evans clustering audit"),
        ("★ VayuDrishti (Proposed)", "Our Capstone\nProject Framework", "Sensor + Image +\nWeather + Spatial", "Strict 45 km Cutoff\n+ Clark-Evans (R)", "Transparently reports baseline strength & live p-values", "Dual forecasters + Bounded Vision (±25) + 6 UI tabs")
    ]

    for i, row in enumerate(matrix_data):
        is_ours = (i == len(matrix_data) - 1)
        bg = RGBColor(15, 23, 42) if not is_ours else RGBColor(22, 78, 99)
        border_c = C_CYAN if is_ours else C_CARD_BORDER

        for j, val in enumerate(row):
            cell = table.cell(i+1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            tf_c = cell.text_frame
            tf_c.word_wrap = True
            tf_c.margin_left = Inches(0.08)
            tf_c.margin_right = Inches(0.08)
            tf_c.margin_top = Inches(0.06)
            tf_c.margin_bottom = Inches(0.06)
            p = tf_c.paragraphs[0]
            r = p.add_run()
            r.text = val
            r.font.name = FONT_BODY
            r.font.size = Pt(8.5 if not is_ours else 9)
            if is_ours:
                r.font.bold = True
                r.font.color.rgb = C_WHITE if j != 0 else C_CYAN
            else:
                r.font.color.rgb = C_ICE_BLUE if j != 0 else C_WHITE
                if j == 0:
                    r.font.bold = True

    s6_notes = (
        "Looking at previous literature, earlier systems generally fell into isolated categories: "
        "Classical statistical models fail to capture abrupt atmospheric shifts; single-station LSTMs cannot map unmonitored neighborhoods; "
        "and camera-only deep learning models are dangerously fooled by weather. "
        "Furthermore, unconstrained multimodal architectures risk having noisy image features corrupt ground-truth sensors. "
        "VayuDrishti addresses these exact literature gaps through our bounded residual fusion (limiting image adjustment to ±25 units) "
        "and an explicit 45 km spatial boundary."
    )
    set_speaker_notes(s6, s6_notes)

    # =========================================================================
    # SLIDE 7: KEY CONTRIBUTIONS (4 Large Visual Contribution Cards)
    # =========================================================================
    print("Formatting Slide 07: Key Contributions...")
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, C_DEEP_NAVY)
    add_header(s7, "06 / Key Contributions", "Four Methodological & Engineering Contributions", 7)

    c_w = Inches(4.45)
    c_h = Inches(1.95)

    # Contribution 1: Bounded Vision
    c1, tf1 = add_glass_card(s7, Inches(0.4), Inches(1.15), c_w, c_h,
                             title="1. Bounded Vision Residual Gating",
                             subtitle="Safe multimodal fusion with guaranteed sensor fallback",
                             top_accent=C_PURPLE)
    add_bullet(tf1, "±25 µg/m³ Safety Limit", "Vision projector outputs a strictly bounded correction delta delta in [-25, +25] on top of the physical sensor baseline.", 9)
    add_bullet(tf1, "Guaranteed Fallback", "When outdoor sky photos are unavailable or low-quality, learned gate m=0 forces delta=0, guaranteeing pure sensor fallback.", 9)

    # Contribution 2: Spatial Safeguards
    c2, tf2 = add_glass_card(s7, Inches(5.15), Inches(1.15), c_w, c_h,
                             title="2. Spatial Distance Safeguards (45 km Limit)",
                             subtitle="Halting unconstrained GIS interpolation",
                             top_accent=C_CYAN)
    add_bullet(tf2, "Hard 45 km Cutoff", "Based on LOSO cross-validation, spatial IDW interpolation is strictly prohibited if nearest monitor > 45 km away.", 9)
    add_bullet(tf2, "Clear UI Notice", "The platform explicitly displays 'Out of Operational Range' rather than generating misleading continuous heatmaps.", 9)

    # Contribution 3: Conformal Uncertainty
    c3, tf3 = add_glass_card(s7, Inches(0.4), Inches(3.25), c_w, c_h,
                             title="3. Distribution-Free Conformal Intervals",
                             subtitle="Calibrated 90% uncertainty prediction bands",
                             top_accent=C_EMERALD)
    add_bullet(tf3, "90.4% Empirical Coverage", "Evaluated on live sensor observations, exactly 90.4% of real pollutant readings fell inside our 90% conformal band.", 9)
    add_bullet(tf3, "Risk-Aware UI", "Users receive actionable health guidance based on realistic upper and lower uncertainty bounds instead of overconfident single numbers.", 9)

    # Contribution 4: Operational Platform
    c4, tf4 = add_glass_card(s7, Inches(5.15), Inches(3.25), c_w, c_h,
                             title="4. Full-Stack Public Web Intelligence Platform",
                             subtitle="Real-time multi-station deployment on Streamlit Cloud",
                             top_accent=C_TEAL)
    add_bullet(tf4, "6 Interactive Modules", "Deployed live with real-time station monitoring, spatial heatmaps, Clark-Evans clustering diagnostics, and camera tests.", 9)
    add_bullet(tf4, "CPU Efficiency", "Full pipeline executes on standard Intel Core i7 CPU without requiring expensive cloud GPUs for inference.", 9)

    s7_notes = (
        "We highlight four core contributions in our research: "
        "First, we solved the camera reliability problem by bounding image adjustments to plus-or-minus 25 units, with guaranteed zero fallback when photos are missing. "
        "Second, we enforced a strict 45 km spatial cutoff, preventing misleading continuous maps in unmonitored zones. "
        "Third, we integrated conformal prediction to deliver guaranteed 90% confidence intervals, verified live at 90.4% coverage. "
        "Fourth, we deployed an accessible full-stack Streamlit web application running efficiently on standard CPU hardware."
    )
    set_speaker_notes(s7, s7_notes)

    # =========================================================================
    # SLIDE 8: SYSTEM ARCHITECTURE (End-to-End Visual Architecture)
    # =========================================================================
    print("Formatting Slide 08: System Architecture...")
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, C_DEEP_NAVY)
    add_header(s8, "07 / System Architecture", "End-to-End System Pipeline & Multimodal Information Flow", 8)

    # Left: 3 Process Stages
    c_left, tf_l = add_glass_card(s8, Inches(0.4), Inches(1.15), Inches(4.30), Inches(4.05),
                                  title="Pipeline Stages & Data Flow",
                                  subtitle="Three structured computational tiers",
                                  top_accent=C_CYAN)
    add_bullet(tf_l, "1. Ingestion & Quality Audit", "OpenAQ v3 API pulls live hourly criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3) and weather across 31 cities. Handles missingness with 3h linear interpolation and assigns 5 freshness categories.", 9)
    add_bullet(tf_l, "2. Dual Forecasters & Residual Gating", "Dual predictors ingest preprocessed inputs: 2-layer LSTM sequence model (24h x 27 channels) and tuned LightGBM (49 features). Optional outdoor sky photos are processed by ResNet-50 and gated to a safe [-25, +25] adjustment.", 9)
    add_bullet(tf_l, "3. Spatial IDW & Conformal Delivery", "Geodesic IDW interpolation estimates pollution between stations within 45 km. Conformal prediction adds 90% calibrated uncertainty intervals before serving outputs to 6 Streamlit modules.", 9)

    # Right: High-Res Diagram Container
    diag_box = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.90), Inches(1.15), Inches(4.70), Inches(4.05))
    diag_box.fill.solid()
    diag_box.fill.fore_color.rgb = RGBColor(15, 23, 42)
    diag_box.line.color.rgb = C_CARD_BORDER
    diag_box.line.width = Pt(1)

    # Embed architecture diagram
    arch_fig = PAPER_FIGS / "fig_arch_overall.png"
    if arch_fig.exists():
        s8.shapes.add_picture(str(arch_fig), Inches(5.00), Inches(1.50), width=Inches(4.50))

    diag_label = s8.shapes.add_textbox(Inches(5.00), Inches(4.75), Inches(4.50), Inches(0.35))
    tf_dl = diag_label.text_frame
    p_dl = tf_dl.paragraphs[0]
    p_dl.alignment = PP_ALIGN.CENTER
    r_dl = p_dl.add_run()
    r_dl.text = "Figure 1: Multimodal Gated Residual Hybrid Architecture Flowchart"
    r_dl.font.name = FONT_BODY
    r_dl.font.size = Pt(8.5)
    r_dl.font.italic = True
    r_dl.font.color.rgb = C_CYAN

    s8_notes = (
        "This slide presents the complete system architecture of VayuDrishti. "
        "On the left, you can trace the three-stage data lifecycle: "
        "Stage 1 ingests live telemetry, fixes minor missing values up to 3 hours, and tags observation freshness. "
        "Stage 2 passes data to our dual ML predictors (LSTM and LightGBM) while ResNet-50 extracts visual features through our bounded quality gate. "
        "Stage 3 applies spatial IDW interpolation within 45 km and computes conformal prediction bands before serving interactive results to the web dashboard."
    )
    set_speaker_notes(s8, s8_notes)

    # =========================================================================
    # SLIDE 9: INTERACTIVE WEB DASHBOARD (6 Modules Showcase)
    # =========================================================================
    print("Formatting Slide 09: Web Dashboard Showcase...")
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, C_DEEP_NAVY)
    add_header(s9, "08 / User Interface", "Interactive Web Dashboard: Six Operational Application Modules", 9)

    # Left Column: 6 Module Cards
    c_m, tf_m = add_glass_card(s9, Inches(0.4), Inches(1.15), Inches(4.30), Inches(4.05),
                              title="Production Modules (Streamlit Cloud)",
                              subtitle="Live platform available at multimodal-air-quality-c38w.streamlit.app",
                              top_accent=C_TEAL)
    add_bullet(tf_m, "M1: Live Monitor", "Current pollutant levels, AQI sub-indices, observation age, and 5 freshness badges for 85+ stations.", 9)
    add_bullet(tf_m, "M2: City Heatmap", "Interactive geographic map displaying 45 km bounded IDW spatial particulate estimates.", 9)
    add_bullet(tf_m, "M3: Station Audit", "Audits 31 Indian cities, calculating Clark-Evans clustering scores (R) and offline station ratios.", 9)
    add_bullet(tf_m, "M4: 1-Hour Forecast", "Next-hour predictions from LSTM and LightGBM with distribution-free 90% confidence bands.", 9)
    add_bullet(tf_m, "M5: Camera Studio", "Upload outdoor sky photos to inspect bounded image adjustments ([-25, +25] µg/m³ limit).", 9)
    add_bullet(tf_m, "M6: Model Diagnostics", "Historical training curves, feature importance rankings, and empirical test benchmarks.", 9)

    # Right Column: Actual Dashboard Screenshot
    dash_box = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.90), Inches(1.15), Inches(4.70), Inches(4.05))
    dash_box.fill.solid()
    dash_box.fill.fore_color.rgb = RGBColor(15, 23, 42)
    dash_box.line.color.rgb = C_CARD_BORDER
    dash_box.line.width = Pt(1)

    dash_img = PAPER_FIGS / "dashboard_monitoring_interface.png"
    if dash_img.exists():
        s9.shapes.add_picture(str(dash_img), Inches(5.00), Inches(1.25), width=Inches(4.50))

    s9_notes = (
        "To make our research actionable for environmental engineers and citizens, we built a 6-module interactive dashboard on Streamlit Cloud. "
        "The modules cover live station monitoring, 45 km spatial heatmaps, Clark-Evans clustering diagnostics, 1-hour ahead forecasts, "
        "a camera studio for testing bounded image refinement, and a transparent model diagnostics panel showing training curves and error distributions. "
        "The real application interface is shown on the right, operating on live telemetry from Amaravati, Andhra Pradesh."
    )
    set_speaker_notes(s9, s9_notes)

    # =========================================================================
    # SLIDE 10: ALGORITHMS & TECHNOLOGIES (Math & Stack Visualized)
    # =========================================================================
    print("Formatting Slide 10: Algorithms & Implementation...")
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, C_DEEP_NAVY)
    add_header(s10, "09 / Algorithms & Stack", "Mathematical Formulations & Production Technology Stack", 10)

    aw = Inches(4.45)
    ah = Inches(1.68)

    # Alg 1: IDW
    c1, tf1 = add_glass_card(s10, Inches(0.4), Inches(1.15), aw, ah,
                             title="1. Inverse Distance Weighting (IDW)",
                             subtitle="Spatial Geodesic Interpolation with 45 km Limit",
                             top_accent=C_CYAN)
    add_bullet(tf1, "Formula", "z(x) = sum(w_i * z_i) / sum(w_i), where w_i = d_i^(-p) with power parameter p=2.0.", 8.8)
    add_bullet(tf1, "Cutoff", "Calculates Haversine distance d_i; strictly requires min(d_i) <= 45 km, else returns Out of Range.", 8.8)

    # Alg 2: Clark-Evans
    c2, tf2 = add_glass_card(s10, Inches(5.15), Inches(1.15), aw, ah,
                             title="2. Clark-Evans Nearest Neighbor Index (R)",
                             subtitle="Spatial Monitor Geometry & Clustering Audit",
                             top_accent=C_AMBER)
    add_bullet(tf2, "Formula", "R = r_observed / r_expected = 2 * r_A * sqrt(rho), where rho is station spatial density.", 8.8)
    add_bullet(tf2, "Interpretation", "R < 1 indicates clustered monitors (e.g., Hyderabad R=0.58); R approx 1 indicates random distribution.", 8.8)

    # Alg 3: Conformal Prediction
    c3, tf3 = add_glass_card(s10, Inches(0.4), Inches(2.95), aw, ah,
                             title="3. Distribution-Free Conformal Prediction",
                             subtitle="Calibrated 90% Prediction Intervals",
                             top_accent=C_EMERALD)
    add_bullet(tf3, "Formula", "C(X) = [y_hat - q_(1-alpha), y_hat + q_(1-alpha)], calibrated on held-out residual quantiles.", 8.8)
    add_bullet(tf3, "Guarantee", "Guarantees mathematically that P(Y in C(X)) >= 1 - alpha = 90% without assuming Gaussian errors.", 8.8)

    # Alg 4: Bounded Vision Gate
    c4, tf4 = add_glass_card(s10, Inches(5.15), Inches(2.95), aw, ah,
                             title="4. Residual Hybrid Vision Gate",
                             subtitle="Safe Multimodal Bounded Fusion",
                             top_accent=C_PURPLE)
    add_bullet(tf4, "Formula", "y_hybrid = max(0, y_sensor + delta), where delta = m * alpha_eff * Delta in [-25, +25] µg/m³.", 8.8)
    add_bullet(tf4, "Fallback", "If image is missing or corrupted, binary mask m=0 forces delta=0, guaranteeing pure sensor baseline.", 8.8)

    # Bottom Tech Stack Panel
    tech_box = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4), Inches(4.72), Inches(9.2), Inches(0.50))
    tech_box.fill.solid()
    tech_box.fill.fore_color.rgb = C_CARD_BG_ALT
    tech_box.line.color.rgb = C_CARD_BORDER
    tf_tb = tech_box.text_frame
    tf_tb.margin_left = Inches(0.15)
    tf_tb.margin_top = Inches(0.08)
    p_tb = tf_tb.paragraphs[0]
    r_tb1 = p_tb.add_run()
    r_tb1.text = "Production Software Stack & Hardware: "
    r_tb1.font.name = FONT_HEADING
    r_tb1.font.size = Pt(9.5)
    r_tb1.font.bold = True
    r_tb1.font.color.rgb = C_CYAN
    r_tb2 = p_tb.add_run()
    r_tb2.text = "Python 3.11  |  PyTorch (LSTM & ResNet-50)  |  LightGBM (GBDT)  |  Scikit-Learn  |  Streamlit Cloud  |  Folium  |  Intel Core i7 CPU (Zero GPU Required)"
    r_tb2.font.name = FONT_BODY
    r_tb2.font.size = Pt(9)
    r_tb2.font.color.rgb = C_ICE_BLUE

    s10_notes = (
        "This slide presents the core mathematical foundations and software stack powering VayuDrishti: "
        "IDW interpolation estimates spatial pollution using inverse squared geodesic distance within 45 km. "
        "Clark-Evans analysis measures whether monitoring stations are clustered (R < 1) or spread out. "
        "Conformal prediction provides distribution-free 90% confidence bands without assuming Gaussian normality. "
        "Our residual vision gate clamps image corrections to plus-or-minus 25 units, dropping to zero when images are missing. "
        "The entire platform is built in Python with PyTorch and LightGBM, running efficiently on a standard Intel Core i7 CPU."
    )
    set_speaker_notes(s10, s10_notes)

    # =========================================================================
    # SLIDE 11: DATASETS & DUAL FEATURE ARCHITECTURES
    # =========================================================================
    print("Formatting Slide 11: Datasets & Feature Breakdown...")
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, C_DEEP_NAVY)
    add_header(s11, "10 / Datasets & Features", "Dataset Scale & Dual Feature Engineering Architectures", 11)

    # Top Row: 4 Metric Stat Cards
    add_stat_box(s11, Inches(0.4), Inches(1.15), Inches(2.18), Inches(1.20), "2,450,000+", "CPCB Historical Records", "2015-2020 Multi-City Benchmark", C_CYAN)
    add_stat_box(s11, Inches(2.74), Inches(1.15), Inches(2.18), Inches(1.20), "85+", "Active Live Stations", "31 Indian Cities via OpenAQ v3", C_TEAL)
    add_stat_box(s11, Inches(5.08), Inches(1.15), Inches(2.18), Inches(1.20), "418", "Live Matched Pairs", "14-Day Audit (Nov 2026)", C_EMERALD)
    add_stat_box(s11, Inches(7.42), Inches(1.15), Inches(2.18), Inches(1.20), "2,150", "Paired Sky Images", "PM25Vision Multimodal Benchmark", C_PURPLE)

    # Bottom Row: Side-by-Side Feature Architectures
    f_w = Inches(4.45)
    f_h = Inches(2.65)

    # Card Left: LSTM Features (27)
    c_lstm, tf_l = add_glass_card(s11, Inches(0.4), Inches(2.50), f_w, Inches(2.72),
                                  title="LSTM Sequential Forecaster (27 Features)",
                                  subtitle="Ingests a 24-hour lookback matrix [24 x 27 channels]",
                                  top_accent=C_PURPLE)
    add_bullet(tf_l, "6 Criteria Pollutants", "Hourly PM2.5, PM10, NO2, SO2, CO, and O3 physical sensor concentrations.", 8.8, space_after=2.0)
    add_bullet(tf_l, "5 Weather Variables", "Temperature, Relative Humidity, Wind Speed, Wind Direction, and Surface Pressure.", 8.8, space_after=2.0)
    add_bullet(tf_l, "4 Cyclic Temporal Encodings", "Sin/Cos encodings of hour-of-day [sin/cos(2pi*h/24)] and month-of-year.", 8.8, space_after=2.0)
    add_bullet(tf_l, "12 Metadata & Quality Flags", "Station elevation, geographic coordinates, and missingness observation masks.", 8.8, space_after=2.0)

    # Card Right: LightGBM Features (49)
    c_lgb, tf_r = add_glass_card(s11, Inches(5.15), Inches(2.50), f_w, Inches(2.72),
                                 title="LightGBM Tabular Forecaster (49 Features)",
                                 subtitle="Ingests an engineered flat feature vector with temporal lags",
                                 top_accent=C_TEAL)
    add_bullet(tf_r, "Autoregressive Lag Features", "Explicit past values at t-1, t-2, t-3, t-6, t-12, and t-24 for PM2.5, PM10, and NO2.", 8.8, space_after=2.0)
    add_bullet(tf_r, "Rolling Moving Statistics", "3-hour, 6-hour, 12-hour, and 24-hour rolling means, rolling standard deviations, and min/max ranges.", 8.8, space_after=2.0)
    add_bullet(tf_r, "Weather Interaction Terms", "Wind vector components (u, v), cooling index, and dewpoint proxy.", 8.8, space_after=2.0)
    add_bullet(tf_r, "Spatial Station Metadata", "Station latitude, longitude, and historical baseline cluster averages.", 8.8, space_after=2.0)

    s11_notes = (
        "Here are the datasets and distinct feature engineering pipelines we designed: "
        "We trained on 2.45 million historical CPCB records across 31 cities, and evaluated live against 418 matched observation pairs. "
        "Notice the fundamental difference between our two model architectures: "
        "LSTM requires sequential time series, so it ingests a 24-hour sequence across 27 clean physical channels. "
        "LightGBM cannot ingest temporal sequences directly, so we engineer explicit time-lag columns (t-1, t-2, t-3, t-6, t-12, t-24) "
        "and rolling moving averages over 3, 6, 12, and 24 hours, yielding 49 engineered tabular inputs."
    )
    set_speaker_notes(s11, s11_notes)

    # =========================================================================
    # SLIDE 12: EXPERIMENTAL SETUP, TRAINING & SAFEGUARDS
    # =========================================================================
    print("Formatting Slide 12: Experimental Setup & Safeguards...")
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, C_DEEP_NAVY)
    add_header(s12, "11 / Experimental Setup", "Training Protocol, Hyperparameters & Operational Safeguards", 12)

    col_w = Inches(2.95)
    gap = Inches(0.18)
    s_top = Inches(1.15)
    s_h = Inches(4.05)

    # Card 1: Hyperparameters
    c1, tf1 = add_glass_card(s12, Inches(0.4), s_top, col_w, s_h,
                             title="Model Hyperparameters",
                             subtitle="Tuned architectures for CPU inference",
                             top_accent=C_PURPLE)
    add_bullet(tf1, "LSTM Sequence Model", "2 recurrent layers, 64 hidden units per layer, dropout rate 0.2, Adam optimizer, initial lr = 1e-3, batch size 64, 30 epochs with early stopping.", 9)
    add_bullet(tf1, "LightGBM Regressor", "150 boosted decision trees, max tree depth 6, learning rate 0.05, 31 leaves, Huber loss (robust to extreme particulate outliers).", 9)
    add_bullet(tf1, "ResNet-50 Projector", "Pre-trained ImageNet backbone, fine-tuned linear projection head to 32D, learned sigmoid quality gate m.", 9)

    # Card 2: Training Protocol
    c2, tf2 = add_glass_card(s12, Inches(0.4) + col_w + gap, s_top, col_w, s_h,
                             title="Data Protocol & Splits",
                             subtitle="Strict temporal chronological isolation",
                             top_accent=C_CYAN)
    add_bullet(tf2, "Chronological Splits", "70% Training (2015-2018), 15% Validation (2019), 15% Test (2020). Prevents future data leakage.", 9)
    add_bullet(tf2, "Spatial Holdout (LOSO)", "Leave-One-Station-Out cross-validation across 4 Delhi NCR stations to validate spatial IDW interpolation.", 9)
    add_bullet(tf2, "14-Day Live Audit", "Real-world test across 4 active monitors (418 matched prediction-observation pairs in Nov 2026).", 9)

    # Card 3: Safeguards
    c3, tf3 = add_glass_card(s12, Inches(0.4) + (col_w + gap)*2, s_top, col_w, s_h,
                             title="Operational Safeguards",
                             subtitle="Defensive boundaries for real-world reliability",
                             top_accent=C_EMERALD)
    add_bullet(tf3, "45 km Distance Boundary", "Halts spatial IDW if nearest monitor > 45 km, stopping unsupported continuous maps.", 9)
    add_bullet(tf3, "[-25, +25] Vision Clamp", "Caps image adjustments so camera noise never overrides physical ground-truth sensors.", 9)
    add_bullet(tf3, "5 Freshness Tiers", "Fresh (<2h), Delayed (2-6h), Stale (6-24h), Offline (>24h), Unknown. Prevents using outdated data.", 9)

    s12_notes = (
        "To ensure our results are scientifically rigorous and reproducible, we followed strict testing rules: "
        "Our models were tuned specifically for fast CPU inference. "
        "We used chronological train/validation/test splits so the models never saw future data during training. "
        "We also conducted a Leave-One-Station-Out spatial holdout in Delhi and a 14-day live audit on currently reporting monitors. "
        "Crucially, our three operational safeguards—the 45 km distance cutoff, the ±25 µg/m³ image limit, and 5 freshness tiers—ensure the system remains reliable in production."
    )
    set_speaker_notes(s12, s12_notes)

    # =========================================================================
    # SLIDE 13: EXPERIMENTAL RESULTS (Historical, Spatial & Live Benchmarks)
    # =========================================================================
    print("Formatting Slide 13: Experimental Results...")
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, C_DEEP_NAVY)
    add_header(s13, "12 / Results", "Comprehensive Empirical Evaluation & 14-Day Live Audit", 13)

    # Top Left: Performance Table
    tbl_shape = s13.shapes.add_table(6, 5, Inches(0.4), Inches(1.10), Inches(4.55), Inches(1.68))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.75) # Model
    tbl.columns[1].width = Inches(0.70) # PM2.5 MAE
    tbl.columns[2].width = Inches(0.70) # PM2.5 RMSE
    tbl.columns[3].width = Inches(0.70) # PM10 MAE
    tbl.columns[4].width = Inches(0.70) # PM10 RMSE

    t_headers = ["Model", "PM2.5 MAE", "PM2.5 RMSE", "PM10 MAE", "PM10 RMSE"]
    for j, h in enumerate(t_headers):
        c = tbl.cell(0, j)
        c.fill.solid()
        c.fill.fore_color.rgb = RGBColor(14, 116, 144)
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = h
        r.font.name = FONT_HEADING
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = C_WHITE

    res_rows = [
        ("1-Hour Persistence (Baseline)", "14.82", "22.40", "34.15", "51.20"),
        ("Multi-Layer Perceptron (MLP)", "11.20", "18.10", "27.50", "43.80"),
        ("LSTM (27 features)", "9.85", "16.20", "24.10", "39.50"),
        ("LightGBM (49 features)", "9.12", "15.04", "22.45", "37.12"),
        ("Weighted Ensemble (Best)", "8.94", "14.78", "21.90", "36.45")
    ]
    for i, row in enumerate(res_rows):
        is_best = (i == len(res_rows) - 1)
        bg = RGBColor(15, 23, 42) if not is_best else RGBColor(22, 78, 99)
        for j, val in enumerate(row):
            c = tbl.cell(i+1, j)
            c.fill.solid()
            c.fill.fore_color.rgb = bg
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = FONT_BODY
            r.font.size = Pt(8.2)
            if is_best:
                r.font.bold = True
                r.font.color.rgb = C_CYAN if j == 0 else C_WHITE
            else:
                r.font.color.rgb = C_ICE_BLUE

    # Top Right: High-Res Parity Plot
    plot_box = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.15), Inches(1.10), Inches(4.45), Inches(1.68))
    plot_box.fill.solid()
    plot_box.fill.fore_color.rgb = RGBColor(15, 23, 42)
    plot_box.line.color.rgb = C_CARD_BORDER
    plot_box.line.width = Pt(1)

    parity_img = PAPER_FIGS / "actual_vs_predicted.png"
    if parity_img.exists():
        s13.shapes.add_picture(str(parity_img), Inches(5.25), Inches(1.15), width=Inches(4.25), height=Inches(1.58))

    # Bottom Left: Spatial Holdout Card
    c_sh, tf_sh = add_glass_card(s13, Inches(0.4), Inches(2.92), Inches(4.55), Inches(2.30),
                                 title="Spatial Holdout in Delhi NCR (LOSO Test)",
                                 subtitle="Validation on completely unmonitored station coordinates",
                                 top_accent=C_CYAN)
    add_bullet(tf_sh, "Nearest Station Baseline", "Selecting the single nearest physical station yields MAE of 30.06 for PM2.5 and 70.68 for PM10.", 8.5, space_after=2.0)
    add_bullet(tf_sh, "IDW Weighted Gain", "Our IDW spatial interpolation reduces MAE to 24.07 (-20.0% error reduction) for PM2.5 and 52.76 (-25.4%) for PM10.", 8.5, space_after=2.0)
    add_bullet(tf_sh, "Key Finding", "Proves that combining multiple nearby monitors with distance weighting outperforms simple nearest-monitor substitution.", 8.5, space_after=2.0)

    # Bottom Right: 14-Day Live Test Card
    c_live, tf_lv = add_glass_card(s13, Inches(5.15), Inches(2.92), Inches(4.45), Inches(2.30),
                                   title="14-Day Live Test & Honest Findings (418 pairs)",
                                   subtitle="Real-world operational audit across 4 reporting stations",
                                   top_accent=C_EMERALD)
    add_bullet(tf_lv, "PM2.5 Live Audit", "LightGBM MAE was 9.64 vs Persistence MAE 9.86 (t = -0.71, p = 0.48, not statistically significant).", 8.5, space_after=2.0)
    add_bullet(tf_lv, "PM10 Live Audit", "LightGBM MAE was 22.13 vs Persistence MAE 22.35 (t = -0.50, p = 0.62, not statistically significant).", 8.5, space_after=2.0)
    add_bullet(tf_lv, "Honest Insight", "Hourly air quality is strongly autocorrelated. ML provides genuine value during rapid weather shifts and supplies 90.4% calibrated confidence bands.", 8.5, space_after=2.0)

    s13_notes = (
        "Here are our empirical results across historical, spatial, and live evaluations: "
        "On historical benchmarks, our weighted ensemble reduced PM2.5 error from 14.82 down to 8.94 MAE, with high correlation (R^2 = 0.949). "
        "In our Delhi spatial holdout, IDW reduced error by 20% compared to simply picking the nearest station. "
        "Most importantly, during our 14-day live audit across 418 matched pairs, LightGBM achieved 9.64 MAE versus 9.86 for persistence. "
        "We openly report that this difference is not statistically significant (p = 0.48). "
        "Consecutive hourly air quality is naturally autocorrelated, so simple persistence is already strong in normal weather; "
        "machine learning provides its real value during abrupt weather transitions and in supplying calibrated conformal uncertainty intervals."
    )
    set_speaker_notes(s13, s13_notes)

    # =========================================================================
    # SLIDE 14: FINDINGS & FUTURE RESEARCH ROADMAP
    # =========================================================================
    print("Formatting Slide 14: Findings & Future Work...")
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, C_DEEP_NAVY)
    add_header(s14, "13 / Conclusions", "Key Empirical Findings & Future Research Roadmap", 14)

    # Left: What We Learned
    c_learn, tf_ln = add_glass_card(s14, Inches(0.4), Inches(1.15), Inches(4.45), Inches(4.05),
                                    title="What We Learned (Empirical Insights)",
                                    subtitle="Key research conclusions from real-world testing",
                                    top_accent=C_CYAN)
    add_bullet(tf_ln, "1. Persistence Baseline is Strong", "Consecutive hourly pollution readings are naturally similar (high autocorrelation). ML models must be evaluated against persistence rather than trivial mean baselines.", 9.2)
    add_bullet(tf_ln, "2. Images Require Strict Safety Limits", "Outdoor sky photos provide useful auxiliary cues (haze), but unconstrained vision models are too vulnerable to fog and glare. Capping vision delta to ±25 µg/m³ is essential.", 9.2)
    add_bullet(tf_ln, "3. Spatial Maps Need Physical Anchors", "IDW reduces spatial error by 20%, but only within 45 km. Drawing continuous gradients beyond 45 km produces false reassurances in unmonitored communities.", 9.2)
    add_bullet(tf_ln, "4. Calibrated Uncertainty Adds Real Value", "Citizens make better health decisions when given calibrated 90% confidence bands (verified live at 90.4% coverage) rather than misleading single-point guesses.", 9.2)

    # Right: What We Plan to Improve (Roadmap)
    c_road, tf_rd = add_glass_card(s14, Inches(5.15), Inches(1.15), Inches(4.45), Inches(4.05),
                                   title="Future Research Roadmap",
                                   subtitle="Four planned technical upgrades",
                                   top_accent=C_TEAL)
    add_bullet(tf_rd, "1. Spatio-Temporal Graph Neural Networks", "Upgrade spatial IDW to Spatio-Temporal Graph Convolutional Networks (ST-GNNs) to explicitly learn urban wind advection and roadway corridors.", 9.2)
    add_bullet(tf_rd, "2. Satellite AOD Remote Sensing Integration", "Incorporate INSAT-3D and Sentinel-5P satellite Aerosol Optical Depth (AOD) grids to provide physical coverage across rural and unmonitored zones.", 9.2)
    add_bullet(tf_rd, "3. Extended Multi-Step Forecast Horizons", "Extend forecasting from next-hour (t+1) to multi-step 6-hour, 12-hour, and 24-hour diurnal forecasts with sequence-to-sequence transformers.", 9.2)
    add_bullet(tf_rd, "4. Low-Cost IoT Sensor Calibration", "Deploy lightweight Transfer Learning to calibrate low-cost laser dust monitors, expanding physical sensor density across university campuses.", 9.2)

    s14_notes = (
        "This slide brings together our core research takeaways and future roadmap. "
        "We learned that hourly air quality persistence is naturally strong, camera images require strict safety limits, "
        "spatial maps need physical anchors within 45 km, and conformal prediction intervals provide genuine value for public health decisions. "
        "Moving forward, our roadmap focuses on four high-impact extensions: "
        "Spatio-Temporal Graph Neural Networks, satellite remote sensing integration, multi-hour extended forecast horizons, "
        "and calibrating low-cost IoT sensors for dense university campus deployments."
    )
    set_speaker_notes(s14, s14_notes)

    # =========================================================================
    # SLIDE 15: ACADEMIC & REGULATORY REFERENCES
    # =========================================================================
    print("Formatting Slide 15: Academic References...")
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, C_DEEP_NAVY)
    add_header(s15, "14 / References", "Foundational Academic Literature & Regulatory Standards", 15)

    ref_w = Inches(4.45)
    ref_h = Inches(4.05)

    # Left Reference Column
    c_rl, tf_rl = add_glass_card(s15, Inches(0.4), Inches(1.15), ref_w, ref_h,
                                 title="Deep Learning & Vision Standards",
                                 subtitle="Foundational neural architectures and air quality standards",
                                 top_accent=C_PURPLE)
    add_bullet(tf_rl, "[1] Hochreiter & Schmidhuber (1997)", "Long Short-Term Memory. Neural Computation, 9(8), 1735-1780. (Foundational sequence learning architecture).", 8.8)
    add_bullet(tf_rl, "[2] He, Zhang, Ren & Sun (2016)", "Deep Residual Learning for Image Recognition. IEEE CVPR, 770-778. (ResNet-50 visual feature extractor).", 8.8)
    add_bullet(tf_rl, "[3] Shepard (1968)", "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data. ACM National Conference, 517-524. (IDW spatial formulation).", 8.8)
    add_bullet(tf_rl, "[4] CPCB (2009)", "National Ambient Air Quality Standards (NAAQS). Central Pollution Control Board, New Delhi, India. (Regulatory breakpoints).", 8.8)

    # Right Reference Column
    c_rr, tf_rr = add_glass_card(s15, Inches(5.15), Inches(1.15), ref_w, ref_h,
                                 title="Machine Learning, Spatial & Uncertainty",
                                 subtitle="Gradient boosting, nearest neighbor and conformal methods",
                                 top_accent=C_CYAN)
    add_bullet(tf_rr, "[5] Ke et al. (2017)", "LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30, 3146-3154. (Tabular feature engine).", 8.8)
    add_bullet(tf_rr, "[6] Angelopoulos & Bates (2021)", "A Gentle Introduction to Conformal Prediction. Foundations and Trends in Machine Learning. (Uncertainty intervals).", 8.8)
    add_bullet(tf_rr, "[7] Clark & Evans (1954)", "Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations. Ecology, 35(4), 445-453. (Clustering score R).", 8.8)
    add_bullet(tf_rr, "[8] OpenAQ Community (2026)", "OpenAQ: Global Real-Time Ambient Air Quality Platform API v3. (Live multi-city telemetry feed).", 8.8)

    s15_notes = (
        "Here are the eight foundational peer-reviewed references that anchor our methodology. "
        "They span recurrent deep learning, residual image perception, inverse distance spatial interpolation, "
        "official CPCB national standards, LightGBM gradient boosting, distribution-free conformal prediction, and spatial point pattern statistics."
    )
    set_speaker_notes(s15, s15_notes)

    # =========================================================================
    # SLIDE 16: CONCLUSION & VIVA DEFENSE
    # =========================================================================
    print("Formatting Slide 16: Conclusion & Viva Defense...")
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16, C_DEEP_NAVY)

    # Atmospheric hero background
    if hero_bg.exists():
        s16.shapes.add_picture(str(hero_bg), 0, 0, width=Inches(10.0), height=Inches(5.625))

    # Top White University Logo
    if vit_logo.exists():
        s16.shapes.add_picture(str(vit_logo), Inches(7.5), Inches(0.35), width=Inches(2.0))

    # Header Kicker
    cat_end = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(0.38), Inches(3.0), Inches(0.30))
    cat_end.fill.solid()
    cat_end.fill.fore_color.rgb = RGBColor(14, 116, 144)
    cat_end.line.fill.background()
    tf_ce = cat_end.text_frame
    tf_ce.margin_left = tf_ce.margin_right = tf_ce.margin_top = tf_ce.margin_bottom = 0
    p_ce = tf_ce.paragraphs[0]
    p_ce.alignment = PP_ALIGN.CENTER
    r_ce = p_ce.add_run()
    r_ce.text = "15 / CONCLUSION & VIVA DEFENSE"
    r_ce.font.name = FONT_HEADING
    r_ce.font.size = Pt(9)
    r_ce.font.bold = True
    r_ce.font.color.rgb = C_WHITE

    # Big Title
    end_title = s16.shapes.add_textbox(Inches(0.5), Inches(0.80), Inches(8.8), Inches(0.90))
    tf_et = end_title.text_frame
    tf_et.word_wrap = True
    tf_et.margin_left = tf_et.margin_top = tf_et.margin_right = tf_et.margin_bottom = 0
    p_et = tf_et.paragraphs[0]
    r_et = p_et.add_run()
    r_et.text = "VayuDrishti: Summary & Open Discussion"
    r_et.font.name = FONT_HEADING
    r_et.font.size = Pt(25)
    r_et.font.bold = True
    r_et.font.color.rgb = C_WHITE

    # Main Center Card: Project Deliverables & Links
    center_card, tf_cc = add_glass_card(s16, Inches(0.5), Inches(1.70), Inches(9.0), Inches(2.70),
                                        title="", top_accent=C_CYAN, bg_color=RGBColor(15, 23, 42))
    
    # Capstone ID Banner
    p_c_id = tf_cc.paragraphs[0]
    r_c_id_lbl = p_c_id.add_run()
    r_c_id_lbl.text = "CAPSTONE PROJECT ID: "
    r_c_id_lbl.font.name = FONT_HEADING
    r_c_id_lbl.font.size = Pt(12)
    r_c_id_lbl.font.bold = True
    r_c_id_lbl.font.color.rgb = C_RED
    r_c_id_val = p_c_id.add_run()
    r_c_id_val.text = "F5815A89"
    r_c_id_val.font.name = FONT_BODY
    r_c_id_val.font.size = Pt(12)
    r_c_id_val.font.bold = True
    r_c_id_val.font.color.rgb = C_CYAN
    p_c_id.space_after = Pt(4)

    add_bullet(tf_cc, "Project Summary", "VayuDrishti establishes an end-to-end urban air quality intelligence framework combining 85+ live CPCB monitors across 31 cities, dual next-hour forecasting (LSTM + LightGBM), bounded image gating ([-25, +25] µg/m³), 45 km spatial safeguards, and 90% conformal intervals.", 9.5)
    add_bullet(tf_cc, "Live Web Platform", "multimodal-air-quality-c38w.streamlit.app   (Fully operational on Streamlit Cloud with 6 interactive modules).", 9.5, C_TEAL)
    add_bullet(tf_cc, "Source Code Repository", "github.com/Rushyasree/Multimodal_airquality_prediction   (Complete pipelines, models & documentation).", 9.5, C_TEAL)
    add_bullet(tf_cc, "Project Team & Guide", "Teliki Rushya Sree (23BCE9562), Machireddy Swathi (23BCE9566), Thanuja Varakala (23BCE9608), Palli Shilpa (23BCE9567)   |   Supervised by Prof. Thota Ramathulsi (SCOPE, VIT-AP University).", 9.5, C_MUTED_SLATE)

    # Bottom Callout Bar: Thank You & Discussion
    thank_bar = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.55), Inches(9.0), Inches(0.65))
    thank_bar.fill.solid()
    thank_bar.fill.fore_color.rgb = RGBColor(14, 116, 144)
    thank_bar.line.color.rgb = C_CYAN
    thank_bar.line.width = Pt(1.5)
    tf_th = thank_bar.text_frame
    tf_th.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_th = tf_th.paragraphs[0]
    p_th.alignment = PP_ALIGN.CENTER
    r_th1 = p_th.add_run()
    r_th1.text = "Thank You! "
    r_th1.font.name = FONT_HEADING
    r_th1.font.size = Pt(15)
    r_th1.font.bold = True
    r_th1.font.color.rgb = C_WHITE
    r_th2 = p_th.add_run()
    r_th2.text = "We welcome questions, suggestions, and feedback from the review committee."
    r_th2.font.name = FONT_BODY
    r_th2.font.size = Pt(12)
    r_th2.font.color.rgb = C_ICE_BLUE

    s16_notes = (
        "To conclude, VayuDrishti provides a transparent, practical, and defensively engineered platform for urban air quality intelligence. "
        "Rather than over-promising on unconstrained deep learning, we demonstrated that machine learning delivers genuine value "
        "when paired with rigorous spatial safeguards, bounded multimodal fusion, and calibrated uncertainty intervals. "
        "Our complete system is live on Streamlit Cloud, and all code, weights, and documentation are open on GitHub. "
        "We thank our guide, Prof. Thota Ramathulsi, and the respected review committee members for their time and guidance. "
        "We are now open for questions and discussion."
    )
    set_speaker_notes(s16, s16_notes)

    # =========================================================================
    # SAVE & MIRROR
    # =========================================================================
    OUT_PPTX_PRES.parent.mkdir(parents=True, exist_ok=True)
    OUT_PPTX_DOCS.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {OUT_PPTX_PRES}...")
    prs.save(str(OUT_PPTX_PRES))

    print(f"Mirroring to {OUT_PPTX_DOCS}...")
    shutil.copy2(OUT_PPTX_PRES, OUT_PPTX_DOCS)

    print("Successfully built VayuDrishti_Final_Review_Professional_Design.pptx (16 slides)!")

if __name__ == "__main__":
    build_professional_deck()
