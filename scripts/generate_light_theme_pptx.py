"""
Generate VayuDrishti Light Theme Capstone Presentation
=====================================================
Professional, modern academic light-theme redesign for B.Tech CSE Capstone Final Review at VIT-AP University.

Design System:
- Pure white background (#FFFFFF)
- Alternate card backgrounds (#F8FAFD / #F3F7FC)
- Primary headings: Deep navy (#15345B)
- Secondary headings: Professional blue (#2463A6)
- Accent: Teal (#159A9C)
- Highlight: Soft sky blue (#DCEBFA)
- Success: Muted green (#31856A)
- Warning: Muted amber (#D49A36)
- Body text: Dark charcoal (#263445)
- Secondary text: Slate grey (#667085)
- Borders & dividers: Light grey (#DCE3EA)

Identity:
- Project Title: VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction in Indian Urban Environments
- Capstone ID: F5815A89
- Team: Teliki Rushya Sree (23BCE9562), Machireddy Swathi (23BCE9566), Thanuja Varakala (23BCE9608), Palli Shilpa (23BCE9567)
- Supervisor: Prof. Thota Ramathulsi, Assistant Professor, SCOPE, VIT-AP University
"""

import sys
import shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------
# CONSTANTS & PALETTE
# ---------------------------------------------------------
FONT_FAMILY = "Calibri"

BG_WHITE = RGBColor(255, 255, 255)
BG_LIGHT_BLUE = RGBColor(243, 247, 252)
CARD_BG = RGBColor(248, 250, 253)
NAVY_TITLE = RGBColor(21, 52, 91)
BLUE_HEADING = RGBColor(36, 99, 166)
TEAL_ACCENT = RGBColor(21, 154, 156)
HIGHLIGHT_BLUE = RGBColor(220, 235, 250)
GREEN_SUCCESS = RGBColor(49, 133, 106)
AMBER_WARN = RGBColor(212, 154, 54)
BODY_CHARCOAL = RGBColor(38, 52, 69)
MUTED_SLATE = RGBColor(102, 112, 133)
BORDER_GREY = RGBColor(220, 227, 234)

WORKSPACE_ROOT = Path("e:/Documents/multimodal_air_quality_vscode_starter")
NAVY_LOGO_PATH = WORKSPACE_ROOT / "scratch/vit_logo_navy.png"
MAP_SNIPPET_PATH = WORKSPACE_ROOT / "scratch/extracted_images/slide_1_shape_9_Picture 10.png"
ARCH_DIAGRAM_PATH = WORKSPACE_ROOT / "scratch/extracted_images/slide_8_shape_8_Picture 9.png"
DASHBOARD_SCREENSHOT_PATH = WORKSPACE_ROOT / "scratch/extracted_images/slide_9_shape_8_Picture 9.png"
PARITY_SCATTER_PATH = WORKSPACE_ROOT / "scratch/extracted_images/slide_13_shape_7_Picture 8.png"


def create_base_presentation():
    """Initializes widescreen 16:9 presentation."""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)
    return prs


def set_slide_background(slide, color):
    """Fills slide background with a solid color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, category_text, slide_title, slide_number_str):
    """Consistent, professional light-theme header."""
    # Category Tracker
    tb_cat = slide.shapes.add_textbox(Inches(0.6), Inches(0.24), Inches(6.5), Inches(0.22))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = False
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text
    p_cat.font.name = FONT_FAMILY
    p_cat.font.size = Pt(8.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL_ACCENT

    # Main Action Slide Title (single-line, fits before badge)
    tb_title = slide.shapes.add_textbox(Inches(0.6), Inches(0.44), Inches(6.6), Inches(0.44))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = False
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = slide_title
    p_title.font.name = FONT_FAMILY
    p_title.font.size = Pt(16.0)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY_TITLE

    # Header Divider Line
    divider = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.92), Inches(8.8), Inches(0.015)
    )
    divider.fill.solid()
    divider.fill.fore_color.rgb = BORDER_GREY
    divider.line.color.rgb = BORDER_GREY
    divider.line.width = Pt(0.5)

    # Right side: Slide Number Badge
    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.45), Inches(0.30), Inches(0.48), Inches(0.30)
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = HIGHLIGHT_BLUE
    badge.line.color.rgb = RGBColor(190, 215, 240)
    badge.line.width = Pt(1)
    tf_b = badge.text_frame
    tf_b.word_wrap = False
    tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
    p_b = tf_b.paragraphs[0]
    p_b.text = slide_number_str
    p_b.alignment = PP_ALIGN.CENTER
    p_b.font.name = FONT_FAMILY
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = NAVY_TITLE

    # Right side: VIT-AP University Navy Logo
    if NAVY_LOGO_PATH.exists():
        slide.shapes.add_picture(
            str(NAVY_LOGO_PATH), Inches(8.05), Inches(0.26), width=Inches(1.35)
        )


def add_card_with_text(slide, left, top, width, height, fill_color=CARD_BG, border_color=BORDER_GREY, pad_x=0.16, pad_y=0.12):
    """Creates a rounded rectangle background card and a perfectly aligned, top-anchored text box inside it."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1.0)
    else:
        card.line.fill.background()

    tb = slide.shapes.add_textbox(
        left + Inches(pad_x),
        top + Inches(pad_y),
        width - Inches(pad_x * 2),
        height - Inches(pad_y * 2)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    return card, tf


def add_bullet_point(text_frame, bold_prefix, text_body, pt_size=8.5, font_color=BODY_CHARCOAL, space_after=2.5, line_spacing=1.12):
    p = text_frame.add_paragraph() if len(text_frame.paragraphs[0].text) > 0 else text_frame.paragraphs[0]
    p.space_after = Pt(space_after)
    p.line_spacing = line_spacing
    if bold_prefix:
        r1 = p.add_run()
        r1.text = bold_prefix + " "
        r1.font.name = FONT_FAMILY
        r1.font.size = Pt(pt_size)
        r1.font.bold = True
        r1.font.color.rgb = BLUE_HEADING
    if text_body:
        r2 = p.add_run()
        r2.text = text_body
        r2.font.name = FONT_FAMILY
        r2.font.size = Pt(pt_size)
        r2.font.bold = False
        r2.font.color.rgb = font_color


def build_slide_1(prs):
    """Slide 1: Title Slide (Academic light theme)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)

    # Top Header Band with Logo & Review Badge
    if NAVY_LOGO_PATH.exists():
        slide.shapes.add_picture(str(NAVY_LOGO_PATH), Inches(0.8), Inches(0.40), width=Inches(2.4))

    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.8), Inches(0.42), Inches(3.4), Inches(0.36))
    badge.fill.solid()
    badge.fill.fore_color.rgb = HIGHLIGHT_BLUE
    badge.line.color.rgb = RGBColor(190, 215, 240)
    badge.line.width = Pt(1)
    tf_b = badge.text_frame
    tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
    p_b = tf_b.paragraphs[0]
    p_b.text = "CAPSTONE FINAL REVIEW  •  CSE 2026"
    p_b.alignment = PP_ALIGN.CENTER
    p_b.font.name = FONT_FAMILY
    p_b.font.size = Pt(10.5)
    p_b.font.bold = True
    p_b.font.color.rgb = NAVY_TITLE

    # Thin Divider
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.92), Inches(8.4), Inches(0.015))
    div.fill.solid()
    div.fill.fore_color.rgb = BORDER_GREY
    div.line.fill.background()

    # Title Area
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.02), Inches(8.4), Inches(1.45))
    tf = t_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_proj = tf.paragraphs[0]
    p_proj.text = "VayuDrishti: Multimodal Spatio-Temporal Air Quality Prediction\nin Indian Urban Environments"
    p_proj.font.name = FONT_FAMILY
    p_proj.font.size = Pt(19.5)
    p_proj.font.bold = True
    p_proj.font.color.rgb = NAVY_TITLE
    p_proj.line_spacing = 1.15
    p_proj.space_after = Pt(4)

    p_sub = tf.add_paragraph()
    p_sub.text = "Air Quality Forecasting and Location-Based Pollution Mapping Using Multiple Data Sources"
    p_sub.font.name = FONT_FAMILY
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = BLUE_HEADING
    p_sub.space_after = Pt(3)

    p_id = tf.add_paragraph()
    p_id.text = "CAPSTONE ID: F5815A89  •  ACADEMIC YEAR 2025–2026"
    p_id.font.name = FONT_FAMILY
    p_id.font.size = Pt(9.5)
    p_id.font.bold = True
    p_id.font.color.rgb = TEAL_ACCENT

    # Bottom Area: 2 Cards (Team Members & Supervisor)
    _, tf_team = add_card_with_text(slide, Inches(0.8), Inches(2.65), Inches(4.7), Inches(2.65), fill_color=CARD_BG)

    p_th = tf_team.paragraphs[0]
    p_th.text = "👥 Project Team Members"
    p_th.font.name = FONT_FAMILY
    p_th.font.size = Pt(11.5)
    p_th.font.bold = True
    p_th.font.color.rgb = NAVY_TITLE
    p_th.space_after = Pt(6)

    members = [
        "TELIKI RUSHYA SREE", "23BCE9562",
        "MACHIREDDY SWATHI", "23BCE9566",
        "THANUJA VARAKALA", "23BCE9608",
        "PALLI SHILPA", "23BCE9567"
    ]
    for i in range(0, len(members), 2):
        pm = tf_team.add_paragraph()
        pm.space_after = Pt(3)
        r_name = pm.add_run()
        r_name.text = f"•  {members[i]}  "
        r_name.font.name = FONT_FAMILY
        r_name.font.size = Pt(9.5)
        r_name.font.bold = True
        r_name.font.color.rgb = BODY_CHARCOAL
        r_id = pm.add_run()
        r_id.text = f"({members[i+1]})"
        r_id.font.name = FONT_FAMILY
        r_id.font.size = Pt(9.0)
        r_id.font.color.rgb = MUTED_SLATE

    p_dept = tf_team.add_paragraph()
    p_dept.space_before = Pt(8)
    p_dept.text = "School of Computer Science & Engineering (SCOPE), VIT-AP University"
    p_dept.font.name = FONT_FAMILY
    p_dept.font.size = Pt(8.5)
    p_dept.font.color.rgb = TEAL_ACCENT

    # Supervisor Card
    _, tf_sup = add_card_with_text(slide, Inches(5.7), Inches(2.65), Inches(3.5), Inches(2.65), fill_color=CARD_BG)

    p_sh = tf_sup.paragraphs[0]
    p_sh.text = "🎓 Project Supervisor"
    p_sh.font.name = FONT_FAMILY
    p_sh.font.size = Pt(11.5)
    p_sh.font.bold = True
    p_sh.font.color.rgb = NAVY_TITLE
    p_sh.space_after = Pt(3)

    p_sname = tf_sup.add_paragraph()
    p_sname.text = "Prof. Thota Ramathulsi"
    p_sname.font.name = FONT_FAMILY
    p_sname.font.size = Pt(11)
    p_sname.font.bold = True
    p_sname.font.color.rgb = BLUE_HEADING

    p_sdes = tf_sup.add_paragraph()
    p_sdes.text = "Assistant Professor, SCOPE\nVIT-AP University, Amaravati"
    p_sdes.font.name = FONT_FAMILY
    p_sdes.font.size = Pt(9)
    p_sdes.font.color.rgb = MUTED_SLATE
    p_sdes.space_after = Pt(4)

    # Frame Map snippet cleanly inside card
    if MAP_SNIPPET_PATH.exists():
        slide.shapes.add_picture(str(MAP_SNIPPET_PATH), Inches(5.85), Inches(3.80), width=Inches(3.20), height=Inches(1.40))


def build_slide_2(prs):
    """Slide 2: Roadmap (Two structured columns)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "01 / ROADMAP", "Presentation Outline & Research Trajectory", "02")

    # Part I Card
    _, tf1 = add_card_with_text(slide, Inches(0.6), Inches(1.12), Inches(4.25), Inches(4.25), fill_color=CARD_BG)

    p1 = tf1.paragraphs[0]
    p1.text = "Part I: Research Foundation & Formulation"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(12)
    p1.font.bold = True
    p1.font.color.rgb = NAVY_TITLE
    p1.space_after = Pt(1)

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Theoretical background, problem definition, and objectives"
    p1_sub.font.name = FONT_FAMILY
    p1_sub.font.size = Pt(8.5)
    p1_sub.font.color.rgb = MUTED_SLATE
    p1_sub.space_after = Pt(6)

    items1 = [
        ("03. Project Overview:", "Core problem, 3 distinct operational modes, and multimodal intelligence."),
        ("04. Problem Statement:", "Four critical urban bottlenecks: sparsity, downtime, image bias, and unconstrained maps."),
        ("05. Project Objectives:", "Four quantifiable engineering goals covering data, ML models, and safeguards."),
        ("06. Literature Review:", "Comparative analysis of 5 existing paradigms versus VayuDrishti's bounded safeguards."),
        ("07. Key Contributions:", "Bounded vision residual gate, 45 km spatial cutoff, conformal intervals, and web UI.")
    ]
    for tag, desc in items1:
        add_bullet_point(tf1, f"• {tag}", desc, pt_size=8.0, space_after=3.0)

    # Part II Card
    _, tf2 = add_card_with_text(slide, Inches(5.15), Inches(1.12), Inches(4.25), Inches(4.25), fill_color=CARD_BG)

    p2 = tf2.paragraphs[0]
    p2.text = "Part II: Engineering, Empirical Results & Viva"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = NAVY_TITLE
    p2.space_after = Pt(1)

    p2_sub = tf2.add_paragraph()
    p2_sub.text = "Architecture, real-time algorithms, benchmark results, and defense"
    p2_sub.font.name = FONT_FAMILY
    p2_sub.font.size = Pt(8.5)
    p2_sub.font.color.rgb = MUTED_SLATE
    p2_sub.space_after = Pt(6)

    items2 = [
        ("08. System Pipeline & Flow:", "Trace data from OpenAQ live sensors to LSTM/LightGBM, ResNet-50, and UI."),
        ("09. Web Dashboard (6 Tabs):", "Live telemetry monitor, city heatmap, station audit, forecast, camera, diagnostics."),
        ("10. Algorithms & Tech Stack:", "Formulations for IDW, Clark-Evans clustering, Conformal prediction, CPU setup."),
        ("11. Datasets & Feature Sets:", "1.37M CPCB records, 418 live pairs, 27 sequence vs 49 tabular features."),
        ("12. Setup & Operational Bounds:", "Training hyperparams, chronological splits, 45 km cutoff, 5 freshness tiers."),
        ("13. Empirical Benchmarks & Audit:", "Benchmark tables, parity scatter plots, Delhi NCR holdout (-20%), 14-day live audit."),
        ("14–16. Findings, Roadmap & Viva:", "Key takeaways, ST-GNN roadmap, academic references, and open Q&A defense.")
    ]
    for tag, desc in items2:
        add_bullet_point(tf2, f"• {tag}", desc, pt_size=7.8, space_after=2.5)


def build_slide_3(prs):
    """Slide 3: Three Operational Modes (3 distinct vertical cards + footer banner)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "02 / OVERVIEW", "Three Distinct Operational Working Modes", "03")

    modes = [
        ("Mode 1: Live Telemetry", "Real-Time Ingestion & Quality Audit", TEAL_ACCENT, [
            ("Live Ingestion:", "Automated hourly data ingestion from 85+ active CPCB monitors across 31 Indian cities using OpenAQ v3 API."),
            ("Multi-Pollutant:", "Monitors 6 criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3) plus meteorological variables."),
            ("Freshness Engine:", "Enforces 5 explicit freshness categories (Fresh, Delayed, Stale, Offline, Unknown) with true observation ages.")
        ]),
        ("Mode 2: 1-Hour Forecast", "Primary LSTM + LightGBM Hybrid (t+1)", BLUE_HEADING, [
            ("Primary Hybrid:", "Fuses 2-layer stacked unidirectional LSTM (27 features) with tuned LightGBM decision forest (49 features)."),
            ("Prediction Horizon:", "Uses 24h rolling lookback window to forecast strictly the next hour (t+1) particulate concentrations."),
            ("Calibrated Bounds:", "Generates distribution-free conformal prediction intervals for transparent uncertainty quantification.")
        ]),
        ("Mode 3: Location Mapping", "Spatial Interpolation with Hard Safeguards", GREEN_SUCCESS, [
            ("Geodesic IDW:", "Inverse Distance Weighting estimates air quality at arbitrary unmonitored coordinates between nearby stations."),
            ("45 km Distance Limit:", "Strictly halts spatial interpolation if nearest active station exceeds 45 km, preventing wild guesses."),
            ("Clustering Audit:", "Computes Clark-Evans Nearest Neighbor index (R) to detect sensor clustering versus geographic spread.")
        ])
    ]

    lefts = [0.6, 3.65, 6.7]
    card_width = 2.70
    for idx, (title, sub, color, bullets) in enumerate(modes):
        _, tf = add_card_with_text(slide, Inches(lefts[idx]), Inches(1.12), Inches(card_width), Inches(3.38), fill_color=CARD_BG)

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_FAMILY
        p_t.font.size = Pt(11.5)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY_TITLE

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.name = FONT_FAMILY
        p_s.font.size = Pt(8.0)
        p_s.font.bold = True
        p_s.font.color.rgb = color
        p_s.space_after = Pt(5)

        for b_tag, b_desc in bullets:
            add_bullet_point(tf, f"• {b_tag}", b_desc, pt_size=7.6, space_after=2.5)

    # Footer Principle Banner
    _, tf_ban = add_card_with_text(slide, Inches(0.6), Inches(4.62), Inches(8.8), Inches(0.68), fill_color=HIGHLIGHT_BLUE, border_color=RGBColor(190, 215, 240), pad_y=0.08)
    p_ban = tf_ban.paragraphs[0]
    r_bp = p_ban.add_run()
    r_bp.text = "Core Architectural Principle: "
    r_bp.font.name = FONT_FAMILY
    r_bp.font.size = Pt(9.0)
    r_bp.font.bold = True
    r_bp.font.color.rgb = NAVY_TITLE
    r_bt = p_ban.add_run()
    r_bt.text = "Physical sensor observations, machine learning forecasts, and spatial interpolations are kept strictly separate. The system never labels interpolated guesses as real sensor readings."
    r_bt.font.name = FONT_FAMILY
    r_bt.font.size = Pt(8.5)
    r_bt.font.color.rgb = BODY_CHARCOAL


def build_slide_4(prs):
    """Slide 4: Problem Statement (2x2 Balanced Cards)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "03 / PROBLEM STATEMENT", "Four Bottlenecks in Urban Air Quality Intelligence", "04")

    problems = [
        ("1. Sensor Sparsity & High Capital Costs", "Infrastructure Deficit", AMBER_WARN, [
            ("Severe Blindspots:", "A government CAAQMS monitoring station costs > ₹1.5 Crore. Consequently, only ~85 continuous stations serve 31 major Indian cities."),
            ("Suburban Voids:", "Large residential suburbs, peri-urban corridors, and university zones have zero physical sensor coverage.")
        ]),
        ("2. Frequent Sensor Downtime & Data Delays", "Data Reliability Gaps", AMBER_WARN, [
            ("Stale Observations:", "Frequent power interruptions, network drops, and maintenance cause >35% of stations to report stale or missing data."),
            ("Silent Failures:", "Most consumer apps silently display hours-old readings as 'current live air quality' without warning citizens.")
        ]),
        ("3. Camera-Based Prediction Vulnerabilities", "Computer Vision Failure Modes", TEAL_ACCENT, [
            ("Weather Confusion:", "End-to-end vision models confuse morning fog, rain droplets, camera lens glare, and dusk darkness for severe particulate pollution."),
            ("False Alarms:", "Unconstrained neural networks produce massive error spikes (>200 µg/m³) during harmless atmospheric changes.")
        ]),
        ("4. Misleading Continuous Spatial Extrapolation", "GIS Artifacts & Overconfidence", BLUE_HEADING, [
            ("Empty Extrapolation:", "Standard GIS systems draw smooth, continuous color gradients across 100+ km gaps where microclimates differ completely."),
            ("False Reassurance:", "Users are shown seemingly precise pollution readings at locations far beyond valid physical correlation range.")
        ])
    ]

    coords = [
        (0.6, 1.12), (5.15, 1.12),
        (0.6, 3.25), (5.15, 3.25)
    ]
    for idx, (title, tag, color, bullets) in enumerate(problems):
        left, top = coords[idx]
        _, tf = add_card_with_text(slide, Inches(left), Inches(top), Inches(4.25), Inches(2.05), fill_color=CARD_BG)

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_FAMILY
        p.font.size = Pt(11.0)
        p.font.bold = True
        p.font.color.rgb = NAVY_TITLE

        p_tag = tf.add_paragraph()
        p_tag.text = tag
        p_tag.font.name = FONT_FAMILY
        p_tag.font.size = Pt(8.0)
        p_tag.font.bold = True
        p_tag.font.color.rgb = color
        p_tag.space_after = Pt(3)

        for b_tag, b_desc in bullets:
            add_bullet_point(tf, f"• {b_tag}", b_desc, pt_size=7.8, space_after=2.5)


def build_slide_5(prs):
    """Slide 5: Objectives (4 Numbered Cards)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "04 / OBJECTIVES", "Four Core Engineering Goals & Deliverables", "05")

    goals = [
        ("GOAL 01", "Automate Telemetry & Freshness Engine", "Tech: OpenAQ v3 API + 5-Tier Engine", TEAL_ACCENT, [
            "Pulls hourly criteria pollutants across 31 cities automatically.",
            "Categorizes data into Fresh (<2h), Delayed (2-6h), Stale (6-24h), Offline (>24h), Unknown.",
            "Eliminates silent data staleness; tracks true observation age."
        ]),
        ("GOAL 02", "Primary LSTM + LightGBM Forecaster", "Tech: LSTM (27f) + LightGBM (49f)", BLUE_HEADING, [
            "Forecasts PM2.5 and PM10 concentrations strictly for next hour (t+1).",
            "Captures recurrent sequential patterns (27 features) and tabular rules (49 features).",
            "Generates distribution-free conformal prediction intervals."
        ]),
        ("GOAL 03", "Bounded Vision & Spatial Safeguards", "Tech: ResNet-50 Projector + 45 km Cutoff", GREEN_SUCCESS, [
            "Limits image adjustments to a safe range of ±25 µg/m³ around sensor baseline.",
            "Guarantees automatic zero adjustment (fallback) when sky photos are missing.",
            "Halts spatial IDW interpolation beyond 45 km to avoid unsupported continuous maps."
        ]),
        ("GOAL 04", "Operational Web Dashboard", "Tech: Streamlit Cloud + Folium + SciPy", NAVY_TITLE, [
            "Provides a public, interactive 6-module intelligence interface deployed 24/7.",
            "Integrates Clark-Evans clustering diagnostics (R) to evaluate station geometry.",
            "Displays transparent model diagnostic benchmarks and residual distributions."
        ])
    ]

    lefts = [0.6, 2.85, 5.10, 7.35]
    c_width = 2.05
    for idx, (gid, title, tech, color, bullets) in enumerate(goals):
        _, tf = add_card_with_text(slide, Inches(lefts[idx]), Inches(1.12), Inches(c_width), Inches(4.25), fill_color=CARD_BG, pad_x=0.14)

        p_gid = tf.paragraphs[0]
        p_gid.text = gid
        p_gid.font.name = FONT_FAMILY
        p_gid.font.size = Pt(11.5)
        p_gid.font.bold = True
        p_gid.font.color.rgb = color

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.name = FONT_FAMILY
        p_t.font.size = Pt(10.0)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY_TITLE
        p_t.space_after = Pt(2)

        p_tech = tf.add_paragraph()
        p_tech.text = tech
        p_tech.font.name = FONT_FAMILY
        p_tech.font.size = Pt(7.8)
        p_tech.font.bold = True
        p_tech.font.color.rgb = TEAL_ACCENT
        p_tech.space_after = Pt(6)

        for b in bullets:
            add_bullet_point(tf, "•", b, pt_size=7.6, space_after=4.0)


def build_slide_6(prs):
    """Slide 6: Literature Review (Comparative Paradigm Table)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "05 / LITERATURE REVIEW", "Comparative Paradigm Analysis: Prior Works vs VayuDrishti", "06")

    rows = 7
    cols = 6
    table_shape = slide.shapes.add_table(rows, cols, Inches(0.6), Inches(1.12), Inches(8.8), Inches(4.25))
    table = table_shape.table

    col_widths = [Inches(1.5), Inches(1.2), Inches(1.1), Inches(1.2), Inches(1.9), Inches(1.9)]
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = width

    headers = ["Approach / Paradigm", "Key Studies", "Modalities", "Spatial Limits", "Major Vulnerability", "VayuDrishti Distinction"]
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.margin_top = cell.margin_bottom = Inches(0.04)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_TITLE
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.0)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ("1. Classical Statistical", "Box-Jenkins (1970)\nZhang et al. (2018)", "Sensors Only\n(Single series)", "None (Point-only)", "Cannot capture non-linear weather interactions", "Uses deep LSTM + tuned LightGBM with 49 features"),
        ("2. Standard Deep Learning", "Hochreiter (1997)\nLi et al. (2020)", "Sensors Only\n(Time series)", "None (Point-only)", "Lacks spatial interpolation & image perception", "Integrates multi-station IDW + bounded vision"),
        ("3. Vision-Only Estimation", "DeadCardassian (2024)\nLiu et al. (2021)", "Images Only\n(Outdoor photos)", "None (Camera only)", "Extreme false spikes from morning fog, rain & glare", "Binds image delta to ±25 µg/m³; pure sensor baseline"),
        ("4. Unbounded Fusion", "Baltrušaitis (2018)\nChen et al. (2022)", "Concatenated\nSensor + Image", "Unconstrained", "High risk of image noise corrupting physical sensor data", "Gated residual architecture: zero fallback when m=0"),
        ("5. Continuous Spatial GIS", "Shepard (1968)\nKrige (1951)", "Spatial Sensors\n(Coordinates)", "Unbounded (Global GIS)", "Interpolates over empty zones without monitors", "Hard 45 km cutoff + Clark-Evans clustering audit"),
        ("★ VayuDrishti (Proposed)", "Our Capstone\nProject Framework", "Sensor + Image +\nWeather + Spatial", "Strict 45 km Cutoff\n+ Clark-Evans (R)", "Transparently reports baseline strength & live p-values", "Primary LSTM+GBDT + Bounded Vision (±25) + 6 UI tabs")
    ]

    for r_idx, row_data in enumerate(data, start=1):
        is_vayudrishti = (r_idx == 6)
        row_bg = HIGHLIGHT_BLUE if is_vayudrishti else (BG_LIGHT_BLUE if r_idx % 2 == 1 else BG_WHITE)
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bg
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_FAMILY
            p.font.size = Pt(7.8)
            p.font.bold = is_vayudrishti
            p.font.color.rgb = NAVY_TITLE if is_vayudrishti else BODY_CHARCOAL
            if c_idx in [1, 2, 3]:
                p.alignment = PP_ALIGN.CENTER


def build_slide_7(prs):
    """Slide 7: Key Contributions (4 distinct cards in 2x2 grid)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "06 / KEY CONTRIBUTIONS", "Four Methodological & Engineering Contributions", "07")

    contribs = [
        ("1. Bounded Vision Residual Gating", "Safe multimodal fusion with guaranteed sensor fallback", TEAL_ACCENT, [
            ("±25 µg/m³ Safety Limit:", "Vision projector outputs a strictly bounded correction delta in [-25, +25] on top of the physical sensor baseline."),
            ("Guaranteed Fallback:", "When outdoor sky photos are unavailable or low-quality, learned gate m=0 forces delta=0, guaranteeing pure sensor fallback.")
        ]),
        ("2. Spatial Distance Safeguards (45 km Limit)", "Halting unconstrained GIS interpolation", BLUE_HEADING, [
            ("Hard 45 km Cutoff:", "Based on LOSO cross-validation, spatial IDW interpolation is strictly prohibited if nearest monitor > 45 km away."),
            ("Clear UI Notice:", "The platform explicitly displays 'Out of Operational Range' rather than generating misleading continuous heatmaps.")
        ]),
        ("3. Distribution-Free Conformal Intervals", "Calibrated 90% uncertainty prediction bands", GREEN_SUCCESS, [
            ("90.4% Empirical Coverage:", "Evaluated on live sensor observations, exactly 90.4% of real pollutant readings fell inside our conformal band."),
            ("Risk-Aware UI:", "Users receive actionable health guidance based on realistic upper and lower uncertainty bounds instead of overconfident single numbers.")
        ]),
        ("4. Full-Stack Public Web Intelligence Platform", "Real-time multi-station deployment on Streamlit Cloud", NAVY_TITLE, [
            ("6 Interactive Modules:", "Deployed live with real-time station monitoring, spatial heatmaps, Clark-Evans clustering diagnostics, and camera tests."),
            ("CPU Efficiency:", "Full pipeline executes on standard Intel Core i7 CPU without requiring expensive cloud GPUs for inference.")
        ])
    ]

    coords = [
        (0.6, 1.12), (5.15, 1.12),
        (0.6, 3.25), (5.15, 3.25)
    ]
    for idx, (title, sub, color, bullets) in enumerate(contribs):
        left, top = coords[idx]
        _, tf = add_card_with_text(slide, Inches(left), Inches(top), Inches(4.25), Inches(2.05), fill_color=CARD_BG)

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_FAMILY
        p.font.size = Pt(11.0)
        p.font.bold = True
        p.font.color.rgb = NAVY_TITLE

        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = FONT_FAMILY
        p_sub.font.size = Pt(8.0)
        p_sub.font.bold = True
        p_sub.font.color.rgb = color
        p_sub.space_after = Pt(3)

        for b_tag, b_desc in bullets:
            add_bullet_point(tf, f"• {b_tag}", b_desc, pt_size=7.8, space_after=2.5)


def build_slide_8(prs):
    """Slide 8: System Architecture (Pipeline stages + Flowchart)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "07 / SYSTEM ARCHITECTURE", "End-to-End System Pipeline & Multimodal Information Flow", "08")

    # Left Card: Pipeline Stages
    _, tf = add_card_with_text(slide, Inches(0.6), Inches(1.12), Inches(3.50), Inches(4.25), fill_color=CARD_BG)

    p = tf.paragraphs[0]
    p.text = "Pipeline Stages & Data Flow"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_sub = tf.add_paragraph()
    p_sub.text = "Three structured computational tiers"
    p_sub.font.name = FONT_FAMILY
    p_sub.font.size = Pt(8.0)
    p_sub.font.bold = True
    p_sub.font.color.rgb = TEAL_ACCENT
    p_sub.space_after = Pt(5)

    tiers = [
        ("1. Ingestion & Quality Audit:", "OpenAQ v3 API pulls live hourly criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3) and weather across 31 cities. Handles missingness with 3h linear interpolation and assigns 5 freshness categories."),
        ("2. Dual Forecasters & Residual Gating:", "Primary forecast fuses 2-layer stacked unidirectional LSTM (24h x 27 channels) with tuned LightGBM (49 features). Separate experimental sky photos processed by ResNet-50 and gated to a safe [-25, +25] adjustment."),
        ("3. Spatial IDW & Conformal Delivery:", "Geodesic IDW interpolation estimates pollution between stations within 45 km. Conformal prediction adds 90% calibrated uncertainty intervals before serving outputs to 6 Streamlit modules.")
    ]
    for tag, desc in tiers:
        add_bullet_point(tf, f"• {tag}", desc, pt_size=7.6, space_after=3.5)

    # Right Card: Flowchart Frame
    add_card_with_text(slide, Inches(4.30), Inches(1.12), Inches(5.10), Inches(4.25), fill_color=BG_WHITE)

    # Title inside card
    t_cap_top = slide.shapes.add_textbox(Inches(4.38), Inches(1.18), Inches(4.94), Inches(0.28))
    tf_ct = t_cap_top.text_frame
    tf_ct.word_wrap = True
    tf_ct.margin_left = tf_ct.margin_top = tf_ct.margin_right = tf_ct.margin_bottom = 0
    p_ct = tf_ct.paragraphs[0]
    p_ct.text = "Figure 1: Multimodal Gated Residual Hybrid Architecture Flowchart"
    p_ct.alignment = PP_ALIGN.CENTER
    p_ct.font.name = FONT_FAMILY
    p_ct.font.size = Pt(9.0)
    p_ct.font.bold = True
    p_ct.font.color.rgb = NAVY_TITLE

    if ARCH_DIAGRAM_PATH.exists():
        slide.shapes.add_picture(str(ARCH_DIAGRAM_PATH), Inches(4.42), Inches(1.52), width=Inches(4.85), height=Inches(2.26))

    # Caption Box below image
    t_cap = slide.shapes.add_textbox(Inches(4.38), Inches(3.95), Inches(4.94), Inches(1.30))
    tf_c = t_cap.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
    p_cap = tf_c.paragraphs[0]
    p_cap.text = "• Primary Forecasting: Stacked 2-layer LSTM (27 features) + LightGBM (49 features) generate sensor baseline.\n• Bounded Gating: ResNet-50 vision projector is strictly bounded to ±25 µg/m³; zero fallback when m=0.\n• Operational Output: Conformal 90% confidence bands deliver calibrated risk-aware bounds to public UI."
    p_cap.line_spacing = 1.15
    p_cap.font.name = FONT_FAMILY
    p_cap.font.size = Pt(7.6)
    p_cap.font.color.rgb = BODY_CHARCOAL


def build_slide_9(prs):
    """Slide 9: User Interface (Six Operational Modules + Dashboard Screenshot)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "08 / USER INTERFACE", "Interactive Web Dashboard: Six Operational Modules", "09")

    # Left Card: Module descriptions
    _, tf = add_card_with_text(slide, Inches(0.6), Inches(1.12), Inches(4.20), Inches(4.25), fill_color=CARD_BG)

    p = tf.paragraphs[0]
    p.text = "Production Modules (Streamlit Cloud)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = NAVY_TITLE

    p_sub = tf.add_paragraph()
    p_sub.text = "Live platform available at vayudrishti-airquality.streamlit.app"
    p_sub.font.name = FONT_FAMILY
    p_sub.font.size = Pt(8.0)
    p_sub.font.bold = True
    p_sub.font.color.rgb = TEAL_ACCENT
    p_sub.space_after = Pt(5)

    modules = [
        ("M1: Live Monitor:", "Current pollutant levels, AQI sub-indices, observation age, and 5 freshness badges for 85+ stations."),
        ("M2: City Heatmap:", "Interactive geographic map displaying 45 km bounded IDW spatial particulate estimates."),
        ("M3: Station Audit:", "Audits 31 Indian cities, calculating Clark-Evans clustering scores (R) and offline station ratios."),
        ("M4: AI Sequence Forecaster:", "Primary LSTM + LightGBM Hybrid predictions with calibrated conformal confidence bands."),
        ("M5: Camera Studio:", "Inspect experimental bounded image adjustments ([-25, +25] µg/m³ limit)."),
        ("M6: Model Diagnostics:", "Historical training curves, feature importance rankings, and empirical test benchmarks.")
    ]
    for tag, desc in modules:
        add_bullet_point(tf, f"• {tag}", desc, pt_size=7.6, space_after=2.5)

    # Right Card: Screenshot Frame
    add_card_with_text(slide, Inches(5.00), Inches(1.12), Inches(4.40), Inches(4.25), fill_color=BG_WHITE)

    # Title inside card
    t_cap_top = slide.shapes.add_textbox(Inches(5.10), Inches(1.18), Inches(4.20), Inches(0.28))
    tf_ct = t_cap_top.text_frame
    tf_ct.word_wrap = True
    tf_ct.margin_left = tf_ct.margin_top = tf_ct.margin_right = tf_ct.margin_bottom = 0
    p_ct = tf_ct.paragraphs[0]
    p_ct.text = "Figure 2: Real-Time Air Quality Intelligence Interface"
    p_ct.alignment = PP_ALIGN.CENTER
    p_ct.font.name = FONT_FAMILY
    p_ct.font.size = Pt(9.0)
    p_ct.font.bold = True
    p_ct.font.color.rgb = NAVY_TITLE

    if DASHBOARD_SCREENSHOT_PATH.exists():
        slide.shapes.add_picture(str(DASHBOARD_SCREENSHOT_PATH), Inches(5.25), Inches(1.50), width=Inches(3.90), height=Inches(3.25))

    # Caption Box below image
    t_cap = slide.shapes.add_textbox(Inches(5.10), Inches(4.82), Inches(4.20), Inches(0.30))
    tf_c = t_cap.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0
    p_cap = tf_c.paragraphs[0]
    p_cap.text = "Public Deployment: vayudrishti-airquality.streamlit.app"
    p_cap.alignment = PP_ALIGN.CENTER
    p_cap.font.name = FONT_FAMILY
    p_cap.font.size = Pt(7.5)
    p_cap.font.bold = True
    p_cap.font.color.rgb = BLUE_HEADING


def build_slide_10(prs):
    """Slide 10: Algorithms & Stack (4 mathematical formulations + stack footer)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "09 / ALGORITHMS & STACK", "Mathematical Formulations & Production Technology Stack", "10")

    algos = [
        ("1. Inverse Distance Weighting (IDW)", "Spatial Geodesic Interpolation with 45 km Limit", BLUE_HEADING, [
            ("Formula:", "z(x) = sum(w_i * z_i) / sum(w_i), where w_i = d_i^(-p) with power parameter p=2.0."),
            ("Cutoff:", "Calculates Haversine distance d_i; strictly requires min(d_i) <= 45 km, else returns Out of Range.")
        ]),
        ("2. Clark-Evans Nearest Neighbor Index (R)", "Spatial Monitor Geometry & Clustering Audit", TEAL_ACCENT, [
            ("Formula:", "R = r_observed / r_expected = 2 * r_A * sqrt(rho), where rho is station spatial density."),
            ("Interpretation:", "R < 1 indicates clustered monitors (e.g., Hyderabad R=0.58); R approx 1 indicates random distribution.")
        ]),
        ("3. Distribution-Free Conformal Prediction", "Calibrated Uncertainty Prediction Intervals", GREEN_SUCCESS, [
            ("Formula:", "C(X) = [y_hat - q_(1-alpha), y_hat + q_(1-alpha)], calibrated on held-out residual quantiles."),
            ("Guarantee:", "Guarantees finite-sample coverage P(Y in C(X)) >= 1 - alpha without assuming Gaussian errors.")
        ]),
        ("4. Residual Hybrid Vision Gate", "Safe Multimodal Bounded Fusion", NAVY_TITLE, [
            ("Formula:", "y_hybrid = max(0, y_sensor + delta), where delta = m * alpha_eff * Delta in [-25, +25] µg/m³."),
            ("Fallback:", "If image is missing or corrupted, binary mask m=0 forces delta=0, guaranteeing pure sensor baseline.")
        ])
    ]

    coords = [
        (0.6, 1.12), (5.15, 1.12),
        (0.6, 2.86), (5.15, 2.86)
    ]
    for idx, (title, sub, color, bullets) in enumerate(algos):
        left, top = coords[idx]
        _, tf = add_card_with_text(slide, Inches(left), Inches(top), Inches(4.25), Inches(1.68), fill_color=CARD_BG)

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_FAMILY
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = NAVY_TITLE

        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = FONT_FAMILY
        p_sub.font.size = Pt(8.0)
        p_sub.font.bold = True
        p_sub.font.color.rgb = color
        p_sub.space_after = Pt(2)

        for b_tag, b_desc in bullets:
            add_bullet_point(tf, f"• {b_tag}", b_desc, pt_size=7.8, space_after=1.5)

    # Footer stack banner
    _, tf_b = add_card_with_text(slide, Inches(0.6), Inches(4.65), Inches(8.8), Inches(0.65), fill_color=HIGHLIGHT_BLUE, border_color=RGBColor(190, 215, 240), pad_y=0.08)
    pb = tf_b.paragraphs[0]
    pb.text = "Production Software Stack & Hardware: Python 3.11  |  PyTorch (LSTM & ResNet-50)  |  LightGBM (GBDT)  |  Scikit-Learn  |  Streamlit Cloud  |  Folium  |  Intel Core i7 CPU (Zero GPU Required)"
    pb.alignment = PP_ALIGN.CENTER
    pb.font.name = FONT_FAMILY
    pb.font.size = Pt(8.5)
    pb.font.bold = True
    pb.font.color.rgb = NAVY_TITLE


def build_slide_11(prs):
    """Slide 11: Datasets & Features (Metric cards + LSTM 27 vs LightGBM 49)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "10 / DATASETS & FEATURES", "Dataset Scale & Dual Feature Engineering Architectures", "11")

    # 4 Top Metric Cards
    metrics = [
        ("1,369,779", "CPCB Hourly Records", "2015-2020 Multi-City Benchmark", TEAL_ACCENT),
        ("85+", "Active Live Stations", "31 Indian Cities via OpenAQ v3", BLUE_HEADING),
        ("418", "Live Matched Pairs", "14-Day Audit (Nov 2026)", GREEN_SUCCESS),
        ("1,235 / 199", "PM25Vision Images / Pairs", "18 Synchronous Monitors", NAVY_TITLE)
    ]
    lefts = [0.6, 2.85, 5.10, 7.35]
    for idx, (val, label, sub, color) in enumerate(metrics):
        _, tf_m = add_card_with_text(slide, Inches(lefts[idx]), Inches(1.12), Inches(2.05), Inches(0.88), fill_color=HIGHLIGHT_BLUE, border_color=RGBColor(190, 215, 240), pad_y=0.06)

        p_v = tf_m.paragraphs[0]
        p_v.text = val
        p_v.alignment = PP_ALIGN.CENTER
        p_v.font.name = FONT_FAMILY
        p_v.font.size = Pt(12.5)
        p_v.font.bold = True
        p_v.font.color.rgb = color

        p_l = tf_m.add_paragraph()
        p_l.text = label
        p_l.alignment = PP_ALIGN.CENTER
        p_l.font.name = FONT_FAMILY
        p_l.font.size = Pt(8.0)
        p_l.font.bold = True
        p_l.font.color.rgb = NAVY_TITLE

        p_s = tf_m.add_paragraph()
        p_s.text = sub
        p_s.alignment = PP_ALIGN.CENTER
        p_s.font.name = FONT_FAMILY
        p_s.font.size = Pt(7.2)
        p_s.font.color.rgb = MUTED_SLATE

    # 2 Bottom Cards: LSTM 27 vs LightGBM 49
    _, tf_l = add_card_with_text(slide, Inches(0.6), Inches(2.12), Inches(4.25), Inches(3.25), fill_color=CARD_BG)

    pl = tf_l.paragraphs[0]
    pl.text = "LSTM Sequential Forecaster (27 Features)"
    pl.font.name = FONT_FAMILY
    pl.font.size = Pt(11.5)
    pl.font.bold = True
    pl.font.color.rgb = NAVY_TITLE

    pl_sub = tf_l.add_paragraph()
    pl_sub.text = "Ingests a 24-hour lookback matrix [24 x 27 channels]:"
    pl_sub.font.name = FONT_FAMILY
    pl_sub.font.size = Pt(8.0)
    pl_sub.font.bold = True
    pl_sub.font.color.rgb = BLUE_HEADING
    pl_sub.space_after = Pt(3)

    lstm_feats = [
        ("2 Particulates:", "Hourly PM2.5 and PM10 physical sensor concentrations (channels 0-1)."),
        ("10 Trace Gases:", "NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene (channels 2-11)."),
        ("7 Weather Variables:", "Temperature, Dewpoint, Relative Humidity, Wind Speed, Precipitation, sin/cos Wind Dir (channels 12-18)."),
        ("6 Cyclical Harmonics:", "Sin/Cos encodings of hour-of-day, month, and day-of-week (channels 19-24)."),
        ("2 Coordinates:", "Station geographic Latitude and Longitude (channels 25-26).")
    ]
    for tag, desc in lstm_feats:
        add_bullet_point(tf_l, f"• {tag}", desc, pt_size=7.4, space_after=2.0)

    # LightGBM Card
    _, tf_g = add_card_with_text(slide, Inches(5.15), Inches(2.12), Inches(4.25), Inches(3.25), fill_color=CARD_BG)

    pg = tf_g.paragraphs[0]
    pg.text = "LightGBM Tabular Forecaster (49 Features)"
    pg.font.name = FONT_FAMILY
    pg.font.size = Pt(11.5)
    pg.font.bold = True
    pg.font.color.rgb = NAVY_TITLE

    pg_sub = tf_g.add_paragraph()
    pg_sub.text = "Ingests an engineered flat feature vector (n_features_in_ = 49):"
    pg_sub.font.name = FONT_FAMILY
    pg_sub.font.size = Pt(8.0)
    pg_sub.font.bold = True
    pg_sub.font.color.rgb = TEAL_ACCENT
    pg_sub.space_after = Pt(3)

    gbdt_feats = [
        ("2 Instantaneous:", "Observed PM2.5 and PM10 concentrations at forecast issue time t."),
        ("6 Calendar Harmonics:", "Sin/Cos encodings for Hour, Month, and Day of Week."),
        ("2 Coordinates:", "Station geographic Latitude and Longitude."),
        ("12 Autoregressive Lags:", "Past values at t-1, t-2, t-3, t-5, t-11, t-23 for both PM2.5 and PM10."),
        ("16 Rolling Statistics:", "3h, 6h, 12h, 24h rolling means and standard deviations for both pollutants."),
        ("10 Momentum & Accel:", "1h, 3h, 6h, 24h rate-of-change differences + particulate acceleration."),
        ("1 Domain Mass Ratio:", "Particulate ratio clip(PM2.5 / (PM10 + 1e-4), -5.0, 5.0).")
    ]
    for tag, desc in gbdt_feats:
        add_bullet_point(tf_g, f"• {tag}", desc, pt_size=7.0, space_after=1.2, line_spacing=1.08)


def build_slide_12(prs):
    """Slide 12: Experimental Setup (3 Cards: Hyperparams, Data Protocol, Safeguards)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "11 / EXPERIMENTAL SETUP", "Training Setup, Hyperparameters & Operational Safeguards", "12")

    sections = [
        ("Model Hyperparameters", "Tuned architectures for CPU inference", BLUE_HEADING, [
            ("LSTM Sequence Model:", "2 stacked unidirectional layers, 64 hidden units per layer, dropout rate 0.2, Adam optimizer, initial lr = 1e-3, batch size 64, 30 epochs with early stopping."),
            ("LightGBM Regressor:", "150 boosted decision trees, max tree depth 6, learning rate 0.05, 31 leaves, Huber loss (robust to extreme particulate outliers)."),
            ("ResNet-50 Projector:", "Pre-trained ImageNet backbone, fine-tuned linear projection head to 32D, learned sigmoid quality gate m.")
        ]),
        ("Data Protocol & Splits", "Strict temporal chronological isolation", TEAL_ACCENT, [
            ("Chronological Splits:", "70% Training (2015-2018), 15% Validation (2019), 15% Test (2020). Strictly terminates future data leakage."),
            ("Recent Holdout Split:", "2021-2023 Adaptation training, H1 2024 Validation, Apr-Aug 2025 untouched holdout (236,289 sequences across 75 stations)."),
            ("Spatial Holdout (LOSO):", "Leave-One-Station-Out cross-validation across 4 Delhi NCR stations to validate spatial IDW interpolation."),
            ("14-Day Live Audit:", "Real-world test across 4 active monitors (418 matched prediction-observation pairs in Nov 2026).")
        ]),
        ("Operational Safeguards", "Defensive boundaries for real-world reliability", GREEN_SUCCESS, [
            ("45 km Distance Boundary:", "Halts spatial IDW if nearest monitor > 45 km, stopping unsupported continuous maps."),
            ("[-25, +25] Vision Clamp:", "Caps image adjustments so camera noise never overrides physical ground-truth sensors."),
            ("5 Freshness Tiers:", "Fresh (<2h), Delayed (2-6h), Stale (6-24h), Offline (>24h), Unknown. Prevents using outdated data."),
            ("Severe Episode Gating:", "Automatically shifts ensemble to 100% LSTM during severe episodes (PM2.5 >= 250, PM10 >= 400).")
        ])
    ]

    lefts = [0.6, 3.65, 6.70]
    c_width = 2.70
    for idx, (title, sub, color, bullets) in enumerate(sections):
        _, tf = add_card_with_text(slide, Inches(lefts[idx]), Inches(1.12), Inches(c_width), Inches(4.25), fill_color=CARD_BG, pad_x=0.14)

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_FAMILY
        p.font.size = Pt(11.0)
        p.font.bold = True
        p.font.color.rgb = NAVY_TITLE

        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = FONT_FAMILY
        p_sub.font.size = Pt(7.8)
        p_sub.font.bold = True
        p_sub.font.color.rgb = color
        p_sub.space_after = Pt(5)

        for b_tag, b_desc in bullets:
            add_bullet_point(tf, f"• {b_tag}", b_desc, pt_size=7.4, space_after=3.0, line_spacing=1.10)


def build_slide_13(prs):
    """Slide 13: Results (Benchmark table with separated cohorts + Full Parity Hexbin Scatter + Holdouts)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "12 / RESULTS", "Comprehensive Empirical Benchmarks & 14-Day Live Audit", "13")

    # Left Column Top: Benchmark Table (width 5.2 in)
    rows = 7
    cols = 5
    t_shape = slide.shapes.add_table(rows, cols, Inches(0.6), Inches(1.12), Inches(5.20), Inches(1.80))
    table = t_shape.table
    c_widths = [Inches(2.00), Inches(0.80), Inches(0.80), Inches(0.80), Inches(0.80)]
    for idx, w in enumerate(c_widths):
        table.columns[idx].width = w

    t_headers = ["Evaluation Cohort / Model", "PM2.5 MAE", "PM2.5 RMSE", "PM10 MAE", "PM10 RMSE"]
    for c_idx, h_text in enumerate(t_headers):
        cell = table.cell(0, c_idx)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_TITLE
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.alignment = PP_ALIGN.CENTER
        p.font.name = FONT_FAMILY
        p.font.size = Pt(7.8)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ("Historical CPCB Holdout (404K seqs)", "", "", "", ""),
        ("  1-Hour Persistence Baseline", "10.68", "19.81", "19.74", "36.24"),
        ("  Pretrained LSTM (27 feats)", "10.57", "18.58", "18.86", "32.20"),
        ("  ★ Severity-Gated Hybrid", "10.35", "18.38", "18.14", "31.64"),
        ("Recent 2025 Holdout (236K seqs)", "", "", "", ""),
        ("  ★ Adapted Hybrid (0.40/0.45)", "6.62", "13.04", "16.14", "34.53")
    ]
    for r_idx, row_vals in enumerate(data, start=1):
        is_sec_header = row_vals[1] == ""
        is_hybrid = "★" in row_vals[0]
        bg = HIGHLIGHT_BLUE if is_hybrid else (BG_LIGHT_BLUE if is_sec_header else BG_WHITE)
        for c_idx, val in enumerate(row_vals):
            cell = table.cell(r_idx, c_idx)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_FAMILY
            p.font.size = Pt(7.4)
            p.font.bold = is_sec_header or is_hybrid
            p.font.color.rgb = NAVY_TITLE if (is_sec_header or is_hybrid) else BODY_CHARCOAL
            if c_idx > 0:
                p.alignment = PP_ALIGN.CENTER

    # Left Column Bottom: 2 Cards Side-by-Side (LOSO + Live Audit)
    # Card 1: Spatial Holdout LOSO
    _, tf_lo = add_card_with_text(slide, Inches(0.6), Inches(3.08), Inches(2.55), Inches(2.28), fill_color=CARD_BG, pad_x=0.12)

    plo = tf_lo.paragraphs[0]
    plo.text = "Spatial Holdout (LOSO)"
    plo.font.name = FONT_FAMILY
    plo.font.size = Pt(10.0)
    plo.font.bold = True
    plo.font.color.rgb = NAVY_TITLE

    plo_s = tf_lo.add_paragraph()
    plo_s.text = "Delhi NCR unmonitored test:"
    plo_s.font.name = FONT_FAMILY
    plo_s.font.size = Pt(7.5)
    plo_s.font.bold = True
    plo_s.font.color.rgb = BLUE_HEADING
    plo_s.space_after = Pt(2)

    loso_b = [
        ("Nearest Baseline:", "Single nearest monitor yields MAE 30.06 (PM2.5) / 70.68 (PM10)."),
        ("IDW Gain:", "Our IDW reduces MAE to 24.07 (-20.0%) and 52.76 (-25.4%)."),
        ("Finding:", "Multi-station distance weighting strongly outperforms nearest monitor.")
    ]
    for tag, desc in loso_b:
        add_bullet_point(tf_lo, f"• {tag}", desc, pt_size=7.2, space_after=2.0)

    # Card 2: 14-Day Live Test
    _, tf_li = add_card_with_text(slide, Inches(3.25), Inches(3.08), Inches(2.55), Inches(2.28), fill_color=CARD_BG, pad_x=0.12)

    pli = tf_li.paragraphs[0]
    pli.text = "14-Day Live Audit (418 pairs)"
    pli.font.name = FONT_FAMILY
    pli.font.size = Pt(10.0)
    pli.font.bold = True
    pli.font.color.rgb = NAVY_TITLE

    pli_s = tf_li.add_paragraph()
    pli_s.text = "Operational audit (Nov 2026):"
    pli_s.font.name = FONT_FAMILY
    pli_s.font.size = Pt(7.5)
    pli_s.font.bold = True
    pli_s.font.color.rgb = TEAL_ACCENT
    pli_s.space_after = Pt(2)

    live_b = [
        ("PM2.5 Audit:", "GBDT MAE 9.64 vs Persistence 9.86 (t = -0.71, p = 0.48)."),
        ("PM10 Audit:", "GBDT MAE 22.13 vs Persistence 22.35 (t = -0.50, p = 0.62)."),
        ("Insight:", "High autocorrelation; ML delivers 90.4% calibrated confidence bands.")
    ]
    for tag, desc in live_b:
        add_bullet_point(tf_li, f"• {tag}", desc, pt_size=7.2, space_after=2.0)

    # Right Column: Full-Height Parity Scatter Hexbin Card
    add_card_with_text(slide, Inches(5.95), Inches(1.12), Inches(3.45), Inches(4.25), fill_color=BG_WHITE)

    t_sc_top = slide.shapes.add_textbox(Inches(6.00), Inches(1.18), Inches(3.35), Inches(0.24))
    tf_sct = t_sc_top.text_frame
    tf_sct.word_wrap = True
    tf_sct.margin_left = tf_sct.margin_top = tf_sct.margin_right = tf_sct.margin_bottom = 0
    p_sct = tf_sct.paragraphs[0]
    p_sct.text = "Figure 3: Dual Parity Hexbin Scatter"
    p_sct.alignment = PP_ALIGN.CENTER
    p_sct.font.name = FONT_FAMILY
    p_sct.font.size = Pt(9.5)
    p_sct.font.bold = True
    p_sct.font.color.rgb = NAVY_TITLE

    if PARITY_SCATTER_PATH.exists():
        slide.shapes.add_picture(str(PARITY_SCATTER_PATH), Inches(6.05), Inches(1.48), width=Inches(3.25), height=Inches(3.06))

    # Caption Box below hexbin
    t_cap_h = slide.shapes.add_textbox(Inches(6.00), Inches(4.65), Inches(3.35), Inches(0.60))
    tf_ch = t_cap_h.text_frame
    tf_ch.word_wrap = True
    tf_ch.margin_left = tf_ch.margin_top = tf_ch.margin_right = tf_ch.margin_bottom = 0
    p_ch = tf_ch.paragraphs[0]
    p_ch.text = "Held-out test set: PM2.5 R² = 0.9492, PM10 R² = 0.9301. Strong adherence to 1:1 ideal diagonal across all concentration levels."
    p_ch.alignment = PP_ALIGN.CENTER
    p_ch.font.name = FONT_FAMILY
    p_ch.font.size = Pt(7.2)
    p_ch.font.color.rgb = MUTED_SLATE


def build_slide_14(prs):
    """Slide 14: Conclusions (Two structured columns: Insights vs Roadmap)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "13 / CONCLUSIONS", "Key Empirical Findings & Future Research Roadmap", "14")

    # Left: Insights
    _, tf1 = add_card_with_text(slide, Inches(0.6), Inches(1.12), Inches(4.25), Inches(4.25), fill_color=CARD_BG)

    p1 = tf1.paragraphs[0]
    p1.text = "What We Learned (Empirical Insights)"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(12)
    p1.font.bold = True
    p1.font.color.rgb = NAVY_TITLE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Key research conclusions from real-world testing"
    p1_sub.font.name = FONT_FAMILY
    p1_sub.font.size = Pt(8.5)
    p1_sub.font.bold = True
    p1_sub.font.color.rgb = BLUE_HEADING
    p1_sub.space_after = Pt(5)

    insights = [
        ("1. Persistence Baseline is Strong:", "Consecutive hourly pollution readings are naturally similar (high autocorrelation). ML models must be evaluated against persistence rather than trivial mean baselines."),
        ("2. Images Require Strict Safety Limits:", "Outdoor sky photos provide useful auxiliary cues (haze), but unconstrained vision models are too vulnerable to fog and glare. Capping vision delta to ±25 µg/m³ is essential."),
        ("3. Spatial Maps Need Physical Anchors:", "IDW reduces spatial error by 20%, but only within 45 km. Drawing continuous gradients beyond 45 km produces false reassurances in unmonitored communities."),
        ("4. Calibrated Uncertainty Adds Real Value:", "Citizens make better health decisions when given calibrated 90% confidence bands (verified live at 90.4% coverage) rather than misleading single-point guesses.")
    ]
    for tag, desc in insights:
        add_bullet_point(tf1, f"• {tag}", desc, pt_size=7.6, space_after=3.5)

    # Right: Roadmap
    _, tf2 = add_card_with_text(slide, Inches(5.15), Inches(1.12), Inches(4.25), Inches(4.25), fill_color=CARD_BG)

    p2 = tf2.paragraphs[0]
    p2.text = "Future Research Roadmap"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = NAVY_TITLE

    p2_sub = tf2.add_paragraph()
    p2_sub.text = "Four planned technical upgrades"
    p2_sub.font.name = FONT_FAMILY
    p2_sub.font.size = Pt(8.5)
    p2_sub.font.bold = True
    p2_sub.font.color.rgb = TEAL_ACCENT
    p2_sub.space_after = Pt(5)

    roadmap = [
        ("1. Spatio-Temporal Graph Neural Networks:", "Upgrade spatial IDW to Spatio-Temporal Graph Convolutional Networks (ST-GNNs) to explicitly learn urban wind advection and roadway corridors."),
        ("2. Satellite AOD Remote Sensing Integration:", "Incorporate INSAT-3D and Sentinel-5P satellite Aerosol Optical Depth (AOD) grids to provide physical coverage across rural and unmonitored zones."),
        ("3. Extended Multi-Step Forecast Horizons:", "Extend forecasting from next-hour (t+1) to multi-step 6-hour, 12-hour, and 24-hour diurnal forecasts with sequence-to-sequence transformers."),
        ("4. Low-Cost IoT Sensor Calibration:", "Deploy lightweight Transfer Learning to calibrate low-cost laser dust monitors, expanding physical sensor density across university campuses.")
    ]
    for tag, desc in roadmap:
        add_bullet_point(tf2, f"• {tag}", desc, pt_size=7.6, space_after=3.5)


def build_slide_15(prs):
    """Slide 15: References (Academic citations in 2 columns)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "14 / REFERENCES", "Foundational Academic Literature & Regulatory Standards", "15")

    # Column 1
    _, tf1 = add_card_with_text(slide, Inches(0.6), Inches(1.12), Inches(4.25), Inches(4.25), fill_color=CARD_BG)

    p1 = tf1.paragraphs[0]
    p1.text = "Deep Learning & Vision Standards"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(12)
    p1.font.bold = True
    p1.font.color.rgb = NAVY_TITLE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Foundational neural architectures and air quality standards"
    p1_sub.font.name = FONT_FAMILY
    p1_sub.font.size = Pt(8.0)
    p1_sub.font.bold = True
    p1_sub.font.color.rgb = BLUE_HEADING
    p1_sub.space_after = Pt(5)

    refs1 = [
        ("[1] Hochreiter & Schmidhuber (1997):", "Long Short-Term Memory. Neural Computation, 9(8), 1735-1780. (Foundational sequence learning architecture)."),
        ("[2] He, Zhang, Ren & Sun (2016):", "Deep Residual Learning for Image Recognition. IEEE CVPR, 770-778. (ResNet-50 visual feature extractor)."),
        ("[3] Shepard (1968):", "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data. ACM National Conference, 517-524. (IDW spatial formulation)."),
        ("[4] CPCB (2009):", "National Ambient Air Quality Standards (NAAQS). Central Pollution Control Board, New Delhi, India. (Regulatory breakpoints).")
    ]
    for tag, desc in refs1:
        add_bullet_point(tf1, tag, desc, pt_size=7.4, space_after=4.0)

    # Column 2
    _, tf2 = add_card_with_text(slide, Inches(5.15), Inches(1.12), Inches(4.25), Inches(4.25), fill_color=CARD_BG)

    p2 = tf2.paragraphs[0]
    p2.text = "Machine Learning, Spatial & Uncertainty"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = NAVY_TITLE

    p2_sub = tf2.add_paragraph()
    p2_sub.text = "Gradient boosting, nearest neighbor and conformal methods"
    p2_sub.font.name = FONT_FAMILY
    p2_sub.font.size = Pt(8.0)
    p2_sub.font.bold = True
    p2_sub.font.color.rgb = TEAL_ACCENT
    p2_sub.space_after = Pt(5)

    refs2 = [
        ("[5] Ke et al. (2017):", "LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30, 3146-3154. (Tabular feature engine)."),
        ("[6] Angelopoulos & Bates (2021):", "A Gentle Introduction to Conformal Prediction. Foundations and Trends in Machine Learning. (Uncertainty intervals)."),
        ("[7] Clark & Evans (1954):", "Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations. Ecology, 35(4), 445-453. (Clustering score R)."),
        ("[8] OpenAQ Community (2026):", "OpenAQ: Global Real-Time Ambient Air Quality Platform API v3. (Live multi-city telemetry feed).")
    ]
    for tag, desc in refs2:
        add_bullet_point(tf2, tag, desc, pt_size=7.4, space_after=4.0)


def build_slide_16(prs):
    """Slide 16: Conclusion & Viva Defense (Closing summary card + Thank You banner)."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    set_slide_background(slide, BG_WHITE)
    add_header(slide, "15 / CONCLUSION & VIVA DEFENSE", "VayuDrishti: Summary & Open Discussion", "16")

    # Top Card: Project Identity & Summary
    _, tf = add_card_with_text(slide, Inches(0.8), Inches(1.12), Inches(8.4), Inches(2.95), fill_color=CARD_BG)

    p_id = tf.paragraphs[0]
    p_id.text = "CAPSTONE PROJECT ID: F5815A89"
    p_id.alignment = PP_ALIGN.CENTER
    p_id.font.name = FONT_FAMILY
    p_id.font.size = Pt(11)
    p_id.font.bold = True
    p_id.font.color.rgb = TEAL_ACCENT
    p_id.space_after = Pt(6)

    bullets = [
        ("• Project Summary:", "VayuDrishti establishes an end-to-end urban air quality intelligence framework combining 85+ live CPCB monitors across 31 cities, primary next-hour forecasting (LSTM + LightGBM), bounded image gating ([-25, +25] µg/m³), 45 km spatial safeguards, and 90% conformal intervals."),
        ("• Live Web Platform:", "vayudrishti-airquality.streamlit.app   (Fully operational on Streamlit Cloud with 6 interactive modules)."),
        ("• Source Code Repository:", "github.com/Rushyasree/Multimodal_airquality_prediction   (Complete pipelines, models & documentation)."),
        ("• Project Team & Guide:", "Teliki Rushya Sree (23BCE9562), Machireddy Swathi (23BCE9566), Thanuja Varakala (23BCE9608), Palli Shilpa (23BCE9567)   |   Supervised by Prof. Thota Ramathulsi (SCOPE, VIT-AP University).")
    ]
    for tag, desc in bullets:
        add_bullet_point(tf, tag, desc, pt_size=8.5, space_after=3.5)

    # Bottom Banner: Thank You
    _, tf_ty = add_card_with_text(slide, Inches(0.8), Inches(4.25), Inches(8.4), Inches(1.15), fill_color=HIGHLIGHT_BLUE, border_color=RGBColor(190, 215, 240))

    pty = tf_ty.paragraphs[0]
    pty.text = "Thank You! We welcome questions, suggestions, and feedback."
    pty.alignment = PP_ALIGN.CENTER
    pty.font.name = FONT_FAMILY
    pty.font.size = Pt(15)
    pty.font.bold = True
    pty.font.color.rgb = NAVY_TITLE
    pty.space_after = Pt(3)

    p_sub = tf_ty.add_paragraph()
    p_sub.text = "VayuDrishti Capstone Review Viva Defense  •  SCOPE  •  VIT-AP University"
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.font.name = FONT_FAMILY
    p_sub.font.size = Pt(10.5)
    p_sub.font.color.rgb = BLUE_HEADING


def main():
    print("Initializing PowerPoint presentation...")
    prs = create_base_presentation()

    print("Building Slide 1: Title Slide...")
    build_slide_1(prs)

    print("Building Slide 2: Roadmap...")
    build_slide_2(prs)

    print("Building Slide 3: Three Operational Modes...")
    build_slide_3(prs)

    print("Building Slide 4: Problem Statement...")
    build_slide_4(prs)

    print("Building Slide 5: Objectives...")
    build_slide_5(prs)

    print("Building Slide 6: Literature Review...")
    build_slide_6(prs)

    print("Building Slide 7: Key Contributions...")
    build_slide_7(prs)

    print("Building Slide 8: System Architecture...")
    build_slide_8(prs)

    print("Building Slide 9: User Interface...")
    build_slide_9(prs)

    print("Building Slide 10: Algorithms & Stack...")
    build_slide_10(prs)

    print("Building Slide 11: Datasets & Features...")
    build_slide_11(prs)

    print("Building Slide 12: Experimental Setup...")
    build_slide_12(prs)

    print("Building Slide 13: Results...")
    build_slide_13(prs)

    print("Building Slide 14: Conclusions...")
    build_slide_14(prs)

    print("Building Slide 15: References...")
    build_slide_15(prs)

    print("Building Slide 16: Conclusion & Viva Defense...")
    build_slide_16(prs)

    out_file = WORKSPACE_ROOT / "presentation/VayuDrishti_Final_Review_Light_Theme.pptx"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_file))
    print(f"Presentation successfully saved to: {out_file}")

    # Mirror copies to docs/ and workspace root
    docs_copy = WORKSPACE_ROOT / "docs/VayuDrishti_Final_Review_Light_Theme.pptx"
    root_copy = WORKSPACE_ROOT / "VayuDrishti_Final_Review_Light_Theme.pptx"
    shutil.copyfile(str(out_file), str(docs_copy))
    shutil.copyfile(str(out_file), str(root_copy))
    print("Copies saved to docs/ and workspace root.")


if __name__ == "__main__":
    main()
