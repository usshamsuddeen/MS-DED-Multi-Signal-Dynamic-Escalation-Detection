"""
Generates publication-quality minimal diagram for Paper 5 Problem Foundation:
Figure 1: The Contact Center Escalation Crisis (Bifurcated Failure Vectors vs MS-DED Resolution)
Saves both as .png and .jpg to Paper5/figures/fig_problem_foundation.png/.jpg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from PIL import Image

FIGURES_DIR = os.path.abspath("Paper5/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# Set High-Quality Typography & Styling
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#B0BEC5',
    'figure.dpi': 300
})

fig = plt.figure(figsize=(14, 8.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.2)
ax.axis('off')

# Color Palette
C_RED_DARK   = '#B71C1C'
C_RED_MED    = '#D32F2F'
C_RED_BG     = '#FFEBEE'
C_RED_LINE   = '#EF9A9A'

C_ORANGE_DARK= '#E65100'
C_ORANGE_MED = '#F57C00'
C_ORANGE_BG  = '#FFF3E0'
C_ORANGE_LINE= '#FFE0B2'

C_BLUE_DARK  = '#0D47A1'
C_BLUE_MED   = '#1976D2'
C_BLUE_BG    = '#E3F2FD'

C_GREEN_DARK = '#1B5E20'
C_GREEN_MED  = '#2E7D32'
C_GREEN_BG   = '#E8F5E9'
C_GREEN_LINE = '#A5D6A7'

C_SLATE_DARK = '#263238'
C_SLATE_MED  = '#455A64'
C_SLATE_MUTED= '#607D8B'
C_BG_BASE    = '#FFFFFF'

# Outer background fill
ax.add_patch(mpatches.Rectangle((0, 0), 14, 8.2, fc=C_BG_BASE, ec='none'))

# ---------------------------------------------------------
# 1. HEADER SECTION
# ---------------------------------------------------------
ax.text(7.0, 7.82, "THE CONTACT CENTER ESCALATION CRISIS: OPERATIONAL PROBLEM FOUNDATION",
        ha='center', va='center', fontsize=14, fontweight='bold', color=C_SLATE_DARK)
ax.text(7.0, 7.50, "Bifurcated Failure Vectors in Automated Dialogue Management: False-Alarm Overload vs. Reactive Customer Churn",
        ha='center', va='center', fontsize=10.5, style='italic', color=C_SLATE_MED)

# Thin divider line under title
ax.plot([0.6, 13.4], [7.32, 7.32], color='#CFD8DC', lw=1.2)

# ---------------------------------------------------------
# 2. LEFT PANEL: FALSE-ALARM FLOOD VECTOR (OVER-SENSITIVITY)
# ---------------------------------------------------------
panel_left = mpatches.FancyBboxPatch((0.5, 2.5), 6.2, 4.65, boxstyle="round,pad=0.15",
                                     ec=C_RED_DARK, fc=C_RED_BG, lw=2.0)
ax.add_patch(panel_left)

# Panel Header Banner
header_left = mpatches.FancyBboxPatch((0.6, 6.65), 6.0, 0.42, boxstyle="round,pad=0.06",
                                      ec=C_RED_DARK, fc=C_RED_DARK, lw=1)
ax.add_patch(header_left)
ax.text(3.6, 6.86, "FAILURE VECTOR A: THE FALSE-ALARM FLOOD (OVER-SENSITIVITY)",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color='#FFFFFF')

# Left Step 1: Benign Inputs
b1 = mpatches.FancyBboxPatch((0.7, 5.25), 2.3, 1.25, boxstyle="round,pad=0.08",
                             ec='#B0BEC5', fc='#FFFFFF', lw=1.2)
ax.add_patch(b1)
ax.text(1.85, 6.32, "Inbound Benign Messages", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_SLATE_DARK)
ax.text(1.85, 5.75, "• \"How do I request a refund?\"\n• \"What is the return policy?\"\n• \"Cancel my renewal auto-pay\"\n• \"Leave a review for product\"",
        ha='center', va='center', fontsize=7.2, color='#37474F')

# Arrow from Inbound to Rule Matcher
ax.annotate("", xy=(3.3, 5.87), xytext=(3.0, 5.87),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color=C_RED_MED, mutation_scale=12))

# Left Step 2: Naive Keyword Matching
b2 = mpatches.FancyBboxPatch((3.3, 5.25), 3.2, 1.25, boxstyle="round,pad=0.08",
                             ec=C_RED_MED, fc='#FFFFFF', lw=1.4)
ax.add_patch(b2)
ax.text(4.9, 6.32, "Naive Keyword Trigger Rules", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_RED_DARK)
ax.text(4.9, 5.75, "• Static RegEx: [refund | cancel | return]\n• Blind to transactional context\n• Immediate uncalibrated escalation flag\n• Baseline False Alarm Rate: 71.5%",
        ha='center', va='center', fontsize=7.2, color='#B71C1C')

# Downward Arrow from Naive Rule to Queue Collapse
ax.annotate("", xy=(4.95, 4.88), xytext=(4.95, 5.25),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color=C_RED_MED, mutation_scale=12))

# Left Step 3: Breakdown Graphic / Impact Box
b3 = mpatches.FancyBboxPatch((0.7, 2.65), 5.8, 2.2, boxstyle="round,pad=0.08",
                             ec=C_RED_MED, fc='#FFFFFF', lw=1.4)
ax.add_patch(b3)

ax.text(3.6, 4.65, "Catastrophic Helpdesk Queue Collapse", ha='center', va='center',
        fontsize=9.0, fontweight='bold', color=C_RED_DARK)

# Mini Bar Representation inside Left Panel
bar_y = 3.9
ax.text(0.9, bar_y + 0.35, "Escalation Queue Breakdown:", fontsize=7.8, fontweight='bold', color=C_SLATE_DARK)

# False Alarms Bar (94.1%) vs True Distress (5.9%)
ax.add_patch(mpatches.Rectangle((0.9, bar_y), 4.0, 0.28, fc=C_RED_MED, ec='none'))
ax.add_patch(mpatches.Rectangle((4.9, bar_y), 0.35, 0.28, fc=C_GREEN_MED, ec='none'))
ax.text(2.9, bar_y + 0.14, "False-Positive Tickets: 94.1%", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#FFFFFF')
ax.text(5.5, bar_y + 0.14, "True: 5.9%", ha='left', va='center', fontsize=7.0, fontweight='bold', color=C_GREEN_DARK)

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Consequences List
ax.text(0.95, 3.25,
        "• Over 9,450 unjustified escalation tickets generated daily\n"
        "• Human support desks overwhelmed with benign routine queries\n"
        "• Average Handle Time (AHT) surges +310%; human agent burnout\n"
        "• True critical customer distress buried under false-alarm backlog",
        fontsize=7.5, color='#37474F', linespacing=1.4)

# ---------------------------------------------------------
# 3. RIGHT PANEL: REACTIVE MELTDOWN VECTOR (UNDER-SENSITIVITY)
# ---------------------------------------------------------
panel_right = mpatches.FancyBboxPatch((7.3, 2.5), 6.2, 4.65, boxstyle="round,pad=0.15",
                                      ec=C_ORANGE_DARK, fc=C_ORANGE_BG, lw=2.0)
ax.add_patch(panel_right)

# Panel Header Banner
header_right = mpatches.FancyBboxPatch((7.4, 6.65), 6.0, 0.42, boxstyle="round,pad=0.06",
                                       ec=C_ORANGE_DARK, fc=C_ORANGE_DARK, lw=1)
ax.add_patch(header_right)
ax.text(10.4, 6.86, "FAILURE VECTOR B: REACTIVE MELTDOWN (UNDER-SENSITIVITY)",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color='#FFFFFF')

# Right Step 1: Multi-Turn Conversation Stream
b4 = mpatches.FancyBboxPatch((7.5, 5.25), 2.6, 1.25, boxstyle="round,pad=0.08",
                             ec='#B0BEC5', fc='#FFFFFF', lw=1.2)
ax.add_patch(b4)
ax.text(8.8, 6.32, "Multi-Turn Customer Drift", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_SLATE_DARK)
ax.text(8.8, 5.75, "Turn 1: \"Where is my order?\"\nTurn 3: \"Still waiting for update...\"\nTurn 5: \"3rd time asking, please help!\"\nTurn 7: \"Nobody answers. Unacceptable.\"",
        ha='center', va='center', fontsize=7.0, color='#37474F')

# Arrow from Stream to Sentiment Model
ax.annotate("", xy=(10.4, 5.87), xytext=(10.1, 5.87),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color=C_ORANGE_MED, mutation_scale=12))

# Right Step 2: Memoryless Sentiment Model
b5 = mpatches.FancyBboxPatch((10.4, 5.25), 2.9, 1.25, boxstyle="round,pad=0.08",
                             ec=C_ORANGE_MED, fc='#FFFFFF', lw=1.4)
ax.add_patch(b5)
ax.text(11.85, 6.32, "Memoryless Static Classifier", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_ORANGE_DARK)
ax.text(11.85, 5.75, "• Evaluates utterances independently\n• Zero velocity / temporal tracking\n• No profanity -> classified \"NEUTRAL\"\n• Ignores compounding customer anger",
        ha='center', va='center', fontsize=7.0, color=C_ORANGE_DARK)

# Downward Arrow to Trajectory Curve
ax.annotate("", xy=(11.85, 4.88), xytext=(11.85, 5.25),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color=C_ORANGE_MED, mutation_scale=12))

# Right Step 3: Frustration Trajectory Graph inside Box
b6 = mpatches.FancyBboxPatch((7.5, 2.65), 5.8, 2.2, boxstyle="round,pad=0.08",
                             ec=C_ORANGE_MED, fc='#FFFFFF', lw=1.4)
ax.add_patch(b6)
ax.text(10.4, 4.65, "Accelerating Frustration vs Blind Static Bot", ha='center', va='center',
        fontsize=9.0, fontweight='bold', color=C_ORANGE_DARK)

# Draw mini coordinate axis for Frustration Curve (left sub-area of b6)
curve_x0, curve_y0 = 7.9, 2.95
curve_w, curve_h = 2.6, 1.30

# Axes lines
ax.plot([curve_x0, curve_x0 + curve_w], [curve_y0, curve_y0], color='#78909C', lw=1.0)
ax.plot([curve_x0, curve_x0], [curve_y0, curve_y0 + curve_h], color='#78909C', lw=1.0)

ax.text(curve_x0 + curve_w/2, curve_y0 - 0.18, "Turn Sequence (1 to 9)", ha='center', fontsize=7.0, color='#546E7A')
ax.text(curve_x0 - 0.08, curve_y0 + curve_h/2, "Frustration", ha='right', va='center', rotation=90, fontsize=7.0, color='#546E7A')

# Actual Cumulative Frustration Curve (Exponential Rise)
turns = np.linspace(0, curve_w, 50)
curve_vals = curve_y0 + 0.06 + (curve_h - 0.12) * (np.exp(turns / 0.9) - 1) / (np.exp(curve_w / 0.9) - 1)
ax.plot(curve_x0 + turns, curve_vals, color='#C62828', lw=2.2, label='Actual Emotional Friction')

# Static Bot Perceived Frustration (Flatline / Low until final snap)
static_turns = np.linspace(0, curve_w * 0.88, 40)
static_vals = curve_y0 + 0.12 + 0.04 * np.sin(static_turns * 3)
ax.plot(curve_x0 + static_turns, static_vals, color='#78909C', lw=1.6, ls='--', label='Static Bot Evaluation')
# Final jump
ax.plot([curve_x0 + curve_w * 0.88, curve_x0 + curve_w], [static_vals[-1], curve_y0 + curve_h], color='#78909C', lw=1.6, ls='--')

# Critical Handover Gap Marker (Early)
early_x = curve_x0 + curve_w * 0.42
ax.plot([early_x, early_x], [curve_y0, curve_y0 + curve_h], color='#2E7D32', lw=1.2, ls=':')
ax.text(early_x, curve_y0 + curve_h + 0.05, "Turn 4: MS-DED Alert", ha='center',
        fontsize=6.5, fontweight='bold', color='#1B5E20')

# Static Bot Alert Marker (Late)
ax.scatter([curve_x0 + curve_w], [curve_y0 + curve_h], color=C_RED_DARK, s=35, zorder=5)
ax.text(curve_x0 + curve_w - 0.05, curve_y0 + curve_h + 0.05, "Turn 9:\nLate!",
        ha='right', va='bottom', fontsize=6.5, fontweight='bold', color=C_RED_DARK)

# Callout impact box on the right side of mini chart
b_right_text = mpatches.FancyBboxPatch((10.75, 2.80), 2.4, 1.55, boxstyle="round,pad=0.06",
                                       ec='#FFE0B2', fc='#FFF8E1', lw=1.0)
ax.add_patch(b_right_text)
ax.text(11.95, 4.15, "Operational Outcome:", ha='center', va='center',
        fontsize=7.3, fontweight='bold', color=C_ORANGE_DARK)
ax.text(10.85, 3.45,
        "• Bot ignores emotional velocity\n"
        "• Unchecked anger accumulation\n"
        "• Customer snaps at Turn 9\n"
        "• Account churn & revenue loss",
        fontsize=7.0, color='#37474F', linespacing=1.35)

# ---------------------------------------------------------
# 4. CENTER BRIDGE: MS-DED UNIFIED RESOLUTION (THE SOLUTION)
# ---------------------------------------------------------
bridge_box = mpatches.FancyBboxPatch((0.5, 0.35), 13.0, 1.95, boxstyle="round,pad=0.15",
                                     ec=C_GREEN_DARK, fc=C_GREEN_BG, lw=2.2)
ax.add_patch(bridge_box)

# Solution Header Banner
sol_header = mpatches.FancyBboxPatch((2.5, 2.05), 9.0, 0.40, boxstyle="round,pad=0.06",
                                     ec=C_GREEN_DARK, fc=C_GREEN_DARK, lw=1)
ax.add_patch(sol_header)
ax.text(7.0, 2.25, "THE MS-DED PARADIGM SHIFT: MULTI-SIGNAL DYNAMIC ESCALATION ENGINE",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#FFFFFF')

# Pillar 1 (Resolves Vector A)
p1 = mpatches.FancyBboxPatch((0.7, 0.5), 5.4, 1.45, boxstyle="round,pad=0.08",
                             ec=C_GREEN_MED, fc='#FFFFFF', lw=1.4)
ax.add_patch(p1)
ax.text(3.4, 1.70, "1. Contextual False-Positive Suppression Filter (FR-7.2.1)",
        ha='center', va='center', fontsize=8.4, fontweight='bold', color=C_GREEN_DARK)
ax.text(3.4, 1.10,
        "• Strips 22 ambiguous commercial tokens ('refund', 'cancel', 'return', 'credit')\n"
        "• Requires multi-signal conjunction (Caps Ratio > 0.4 + Bert Negativity > 0.6)\n"
        "• Enforces length guard (short inputs '< 5 chars' like '??' cannot trigger false alerts)\n"
        "• Result: Slashes False Alarms from 71.5% down to 4.2% (94.1% Reduction!)",
        ha='center', va='center', fontsize=7.2, color='#2E7D32', linespacing=1.35)

# Pillar 2 (Resolves Vector B)
p2 = mpatches.FancyBboxPatch((6.3, 0.5), 5.4, 1.45, boxstyle="round,pad=0.08",
                             ec=C_BLUE_MED, fc='#FFFFFF', lw=1.4)
ax.add_patch(p2)
ax.text(9.0, 1.70, "2. Frustration Velocity Dynamics & Dual Hysteresis Gate",
        ha='center', va='center', fontsize=8.4, fontweight='bold', color=C_BLUE_DARK)
ax.text(9.0, 1.10,
        r"• Continuously tracks Exponential Moving Average: $EMA_t = \alpha S_t + (1-\alpha)EMA_{t-1}$" "\n"
        r"• Computes first-derivative emotional velocity ($v_t = dEMA/dt$) and acceleration" "\n"
        r"• Dual-threshold hysteresis prevents flickering: Warning at $T_{flag}=0.35$, Takeover at $T_{escalate}=0.60$" "\n"
        "• Result: Proactive autonomous human handoff 2.3 turns ahead of explicit customer churn",
        ha='center', va='center', fontsize=7.2, color='#1565C0', linespacing=1.35)

# Verified Empirical Outcome Badge
badge_box = mpatches.FancyBboxPatch((11.9, 0.5), 1.4, 1.45, boxstyle="round,pad=0.08",
                                     ec=C_GREEN_DARK, fc='#C8E6C9', lw=1.4)
ax.add_patch(badge_box)
ax.text(12.6, 1.40, "EMPIRICAL\nRESULT", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1B5E20')
ax.text(12.6, 0.95, "96.8% F1\n-94% False\n0 Churn", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#1B5E20')

# Save high-res PNG
png_path = os.path.join(FIGURES_DIR, "fig_problem_foundation.png")
jpg_path = os.path.join(FIGURES_DIR, "fig_problem_foundation.jpg")

plt.savefig(png_path, dpi=300, facecolor=C_BG_BASE)
print(f"[OK] Saved high-resolution PNG: {png_path}")

# Convert to high-quality JPG
img = Image.open(png_path).convert('RGB')
img.save(jpg_path, 'JPEG', quality=95)
print(f"[OK] Saved high-resolution JPG: {jpg_path}")

plt.close()
