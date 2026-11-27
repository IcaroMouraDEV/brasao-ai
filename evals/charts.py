"""Offline pure-Python SVG chart renderer for Brasão AI benchmark reports."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape

from evals.rubrics import ComparisonReport, ProfileSummary

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Colorblind-friendly palette for academic/presentation charts.
COLOR_BG = "#FFFFFF"
COLOR_SURFACE = "#F8FAFC"
COLOR_BORDER = "#CBD5E1"
COLOR_GRID = "#E2E8F0"
COLOR_AXIS = "#64748B"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#475569"

COLOR_REMOTE = "#2563EB"  # Blue
COLOR_LOCAL = "#059669"  # Emerald
COLOR_ACCENT_AMBER = "#D97706"  # Amber
COLOR_ACCENT_VIOLET = "#7C3AED"  # Violet
COLOR_ACCENT_ROSE = "#E11D48"  # Rose
COLOR_ACCENT_TEAL = "#0D9488"  # Teal

PROFILE_COLORS: dict[str, str] = {
    "remote": COLOR_REMOTE,
    "local": COLOR_LOCAL,
}


def _xml(text: str) -> str:
    """Escape text for safe embedding in XML/SVG elements and attributes."""
    return escape(text, {'"': "&quot;", "'": "&apos;"})


def _profile_label(summary: ProfileSummary) -> str:
    """Format a concise human-readable label for a profile summary."""
    mode_tag = "" if summary.tools_enabled else " [no-tools]"
    return f"{summary.profile}: {summary.model_id}{mode_tag}"


def _short_model_label(summary: ProfileSummary, max_len: int = 26) -> str:
    """Return a compact display label that fits cleanly beneath chart axes."""
    label = _profile_label(summary)
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 1]}…"


def _profile_color(profile: str, index: int) -> str:
    """Return a deterministic color for a profile."""
    if profile in PROFILE_COLORS:
        return PROFILE_COLORS[profile]
    fallback = [COLOR_REMOTE, COLOR_LOCAL, COLOR_ACCENT_VIOLET, COLOR_ACCENT_AMBER]
    return fallback[index % len(fallback)]


def _nice_upper_bound(max_value: float, default_max: float = 1.0) -> float:
    """Compute a clean axis upper bound using 1-2-5 nice-number rounding."""
    if max_value <= 0.0:
        return default_max
    magnitude = 10.0 ** math.floor(math.log10(max_value))
    normalized = max_value / magnitude
    if normalized <= 1.0:
        nice = 1.0
    elif normalized <= 2.0:
        nice = 2.0
    elif normalized <= 2.5:
        nice = 2.5
    elif normalized <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def _format_latency(ms: float) -> str:
    """Format millisecond latency values cleanly for axis ticks and callouts."""
    if ms >= 10_000:
        return f"{ms / 1000.0:.1f}s"
    if ms >= 1_000:
        return f"{ms / 1000.0:.2f}s"
    return f"{ms:.0f}ms"


def _format_usd(usd: float) -> str:
    """Format USD cost per 1k queries cleanly."""
    if usd == 0.0:
        return "$0.00"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1.0:
        return f"${usd:.3f}"
    return f"${usd:.2f}"


def _completion_rates(summary: ProfileSummary) -> tuple[float, float]:
    """Return (overall_completion_pct, offline_completion_pct) in [0, 100]."""
    if summary.total_cases <= 0:
        return 0.0, 0.0
    completed = max(0, summary.total_cases - summary.error_count)
    overall_pct = round((completed / summary.total_cases) * 100.0, 1)
    offline_pct = overall_pct if summary.offline_capable else 0.0
    return overall_pct, offline_pct


def _svg_header(width: int, height: int, title: str, description: str) -> list[str]:
    """Build standard accessible SVG opening tags, styles, and outer card."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_xml(title)}">',
        f"  <title>{_xml(title)}</title>",
        f"  <desc>{_xml(description)}</desc>",
        "  <style>",
        f"    text {{ font-family: {FONT_STACK}; fill: {COLOR_TEXT}; }}",
        f"    .title {{ font-size: 20px; font-weight: 700; fill: {COLOR_TEXT}; }}",
        f"    .subtitle {{ font-size: 12px; fill: {COLOR_MUTED}; }}",
        f"    .panel-title {{ font-size: 15px; font-weight: 700; fill: {COLOR_TEXT}; }}",
        f"    .axis-label {{ font-size: 12px; font-weight: 600; fill: {COLOR_MUTED}; }}",
        f"    .tick-label {{ font-size: 11px; fill: {COLOR_MUTED}; }}",
        f"    .bar-label {{ font-size: 11px; font-weight: 700; fill: {COLOR_TEXT}; }}",
        f"    .legend-text {{ font-size: 12px; fill: {COLOR_TEXT}; }}",
        f"    .note {{ font-size: 11px; fill: {COLOR_MUTED}; }}",
        "  </style>",
        f'  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" '
        f'fill="{COLOR_BG}" stroke="{COLOR_BORDER}" stroke-width="1"/>',
    ]


def _render_empty_chart(title: str, subtitle: str) -> str:
    """Render a valid placeholder SVG when a report contains no profile summaries."""
    width, height = 840, 320
    lines = _svg_header(width, height, title, subtitle)
    lines.extend(
        [
            f'  <text x="36" y="46" class="title">{_xml(title)}</text>',
            f'  <text x="36" y="68" class="subtitle">{_xml(subtitle)}</text>',
            f'  <rect x="36" y="96" width="{width - 72}" height="184" rx="8" '
            f'fill="{COLOR_SURFACE}" stroke="{COLOR_GRID}"/>',
            f'  <text x="{width / 2:.1f}" y="194" text-anchor="middle" class="axis-label">'
            "No profile summaries available in this benchmark report.</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_metadata_subtitle(report: ComparisonReport) -> str:
    """Format concise metadata subtitle shared across all report charts."""
    mode = "dry-run" if report.dry_run else "live"
    tools = "tools/RAG enabled" if report.tools_enabled else "no-tools ablation"
    judge = (
        f"judge: {report.judge_model or 'enabled'}"
        if report.use_judge
        else "deterministic + heuristic"
    )
    return f"Prompt {report.prompt_version} · Mode: {mode} · {tools} · {judge}"


def render_accuracy_pedagogy_svg(report: ComparisonReport) -> str:
    """Render grouped bar chart of Accuracy and SDG 4 Pedagogy scores (0-5) by profile."""
    title = "Historical Accuracy & SDG 4 Pedagogy Scores by Profile"
    subtitle = _report_metadata_subtitle(report)
    if not report.summaries:
        return _render_empty_chart(title, subtitle)

    has_judge_scores = any(
        s.avg_judge_accuracy_5 is not None or s.avg_judge_pedagogy_5 is not None
        for s in report.summaries
    )

    series: list[tuple[str, str]] = [
        ("Det. Accuracy (/5)", COLOR_REMOTE),
        ("SDG 4 Pedagogy (/5)", COLOR_LOCAL),
    ]
    if has_judge_scores:
        series.extend(
            [
                ("Judge Accuracy (/5)", COLOR_ACCENT_VIOLET),
                ("Judge Pedagogy (/5)", COLOR_ACCENT_AMBER),
            ]
        )

    width = max(860, 260 + len(report.summaries) * 260)
    height = 480
    margin_left = 72
    margin_right = 40
    margin_top = 128
    margin_bottom = 104
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    lines = _svg_header(
        width,
        height,
        title,
        "Grouped bar chart comparing historical accuracy and SDG 4 pedagogical scores across profiles.",
    )
    lines.append(f'  <text x="36" y="42" class="title">{_xml(title)}</text>')
    lines.append(f'  <text x="36" y="64" class="subtitle">{_xml(subtitle)}</text>')

    # Legend row
    legend_x = 36
    legend_y = 92
    for label, color in series:
        lines.append(
            f'  <rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        lines.append(
            f'  <text x="{legend_x + 18}" y="{legend_y}" class="legend-text">{_xml(label)}</text>'
        )
        legend_x += 172

    # Horizontal gridlines & Y ticks (0.0 to 5.0)
    for tick in range(6):
        y = margin_top + plot_h - (tick / 5.0) * plot_h
        stroke = COLOR_AXIS if tick == 0 else COLOR_GRID
        lines.append(
            f'  <line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick-label">{tick}.0</text>'
        )

    # Y-axis title
    y_mid = margin_top + plot_h / 2.0
    lines.append(
        f'  <text x="22" y="{y_mid:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 22 {y_mid:.1f})">Score (0-5 scale)</text>'
    )

    # Bars per profile group
    n_groups = len(report.summaries)
    group_slot = plot_w / n_groups
    n_bars = len(series)
    bar_gap = 10.0
    max_group_w = min(group_slot * 0.72, n_bars * 56.0 + (n_bars - 1) * bar_gap)
    bar_w = max(20.0, (max_group_w - (n_bars - 1) * bar_gap) / n_bars)
    actual_group_w = n_bars * bar_w + (n_bars - 1) * bar_gap

    for idx, summary in enumerate(report.summaries):
        group_center = margin_left + (idx + 0.5) * group_slot
        start_x = group_center - actual_group_w / 2.0

        values: list[float | None] = [summary.avg_accuracy_5, summary.avg_pedagogy_5]
        if has_judge_scores:
            values.extend([summary.avg_judge_accuracy_5, summary.avg_judge_pedagogy_5])

        for b_idx, (val, (_, color)) in enumerate(zip(values, series, strict=True)):
            bx = start_x + b_idx * (bar_w + bar_gap)
            if val is None:
                lines.append(
                    f'  <text x="{bx + bar_w / 2:.1f}" y="{margin_top + plot_h - 8:.1f}" '
                    f'text-anchor="middle" class="tick-label">N/A</text>'
                )
                continue
            clamped = max(0.0, min(5.0, val))
            bar_h = (clamped / 5.0) * plot_h
            by = margin_top + plot_h - bar_h
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(1.0, bar_h):.1f}" '
                f'rx="4" fill="{color}"/>'
            )
            lines.append(
                f'  <text x="{bx + bar_w / 2:.1f}" y="{max(margin_top - 4, by - 6):.1f}" '
                f'text-anchor="middle" class="bar-label">{clamped:.2f}</text>'
            )

        # X-axis labels (profile + model + recall/hallucination note)
        lines.append(
            f'  <text x="{group_center:.1f}" y="{margin_top + plot_h + 24:.1f}" '
            f'text-anchor="middle" class="axis-label">{_xml(_short_model_label(summary, 32))}</text>'
        )
        meta_note = (
            f"Recall: {summary.avg_fact_recall:.0%} · "
            f"Hallucinations: {summary.hallucination_case_count}/{summary.total_cases} · "
            f"Errors: {summary.error_count}"
        )
        lines.append(
            f'  <text x="{group_center:.1f}" y="{margin_top + plot_h + 44:.1f}" '
            f'text-anchor="middle" class="note">{_xml(meta_note)}</text>'
        )

    lines.append(
        f'  <text x="36" y="{height - 18}" class="note">'
        "Higher is better (max 5.0). Deterministic scores evaluate expected facts, hallucination traps, and SDG 4 heuristics."
        "</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_latency_accuracy_svg(report: ComparisonReport) -> str:
    """Render a Mean/p95 Latency vs. Accuracy trade-off chart."""
    title = "Mean & p95 Latency vs. Historical Accuracy Trade-off"
    subtitle = _report_metadata_subtitle(report)
    if not report.summaries:
        return _render_empty_chart(title, subtitle)

    width = 880
    height = 500
    margin_left = 78
    margin_right = 56
    margin_top = 128
    margin_bottom = 86
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_lat = max(
        (max(s.mean_latency_ms, s.p95_latency_ms) for s in report.summaries),
        default=0.0,
    )
    x_max = _nice_upper_bound(max_lat * 1.2 if max_lat > 0 else 1000.0, 1000.0)

    lines = _svg_header(
        width,
        height,
        title,
        "Scatter and range plot comparing mean and p95 latency against historical accuracy score.",
    )
    lines.append(f'  <text x="36" y="42" class="title">{_xml(title)}</text>')
    lines.append(f'  <text x="36" y="64" class="subtitle">{_xml(subtitle)}</text>')

    # Legend explaining Mean circle, p95 diamond, and profile colors
    lines.append(f'  <circle cx="44" cy="90" r="6" fill="{COLOR_MUTED}"/>')
    lines.append('  <text x="58" y="94" class="legend-text">Mean latency</text>')
    lines.append(
        f'  <polygon points="164,84 170,90 164,96 158,90" fill="{COLOR_BG}" '
        f'stroke="{COLOR_MUTED}" stroke-width="2.2"/>'
    )
    lines.append('  <text x="178" y="94" class="legend-text">p95 latency</text>')
    lines.append(
        f'  <line x1="274" y1="90" x2="302" y2="90" stroke="{COLOR_MUTED}" '
        f'stroke-width="2" stroke-dasharray="4,3"/>'
    )
    lines.append('  <text x="310" y="94" class="legend-text">Mean → p95 spread</text>')

    # Subtle shaded "ideal zone" callout in top-left of plot area
    lines.append(
        f'  <rect x="{margin_left + 8}" y="{margin_top + 8}" width="214" height="28" rx="6" '
        f'fill="#ECFDF5" stroke="#A7F3D0"/>'
    )
    lines.append(
        f'  <text x="{margin_left + 18}" y="{margin_top + 26}" class="note" '
        'style="fill:#065F46;font-weight:600">↖ Ideal: Lower Latency, Higher Accuracy</text>'
    )

    # Horizontal gridlines (Accuracy 0..5)
    for tick in range(6):
        y = margin_top + plot_h - (tick / 5.0) * plot_h
        stroke = COLOR_AXIS if tick == 0 else COLOR_GRID
        lines.append(
            f'  <line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick-label">{tick}.0</text>'
        )

    # Vertical gridlines (5 intervals from 0 to x_max)
    n_x_ticks = 5
    for i in range(n_x_ticks + 1):
        lat_val = (x_max / n_x_ticks) * i
        x = margin_left + (lat_val / x_max) * plot_w
        stroke = COLOR_AXIS if i == 0 else COLOR_GRID
        lines.append(
            f'  <line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{x:.1f}" y="{margin_top + plot_h + 20:.1f}" text-anchor="middle" '
            f'class="tick-label">{_xml(_format_latency(lat_val))}</text>'
        )

    # Axis titles
    lines.append(
        f'  <text x="{margin_left + plot_w / 2:.1f}" y="{margin_top + plot_h + 46:.1f}" '
        'text-anchor="middle" class="axis-label">End-to-End Latency (lower is faster)</text>'
    )
    y_mid = margin_top + plot_h / 2.0
    lines.append(
        f'  <text x="24" y="{y_mid:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 24 {y_mid:.1f})">Historical Accuracy Score (0-5)</text>'
    )

    # Plot each profile's Mean -> p95 whisker and markers
    for idx, summary in enumerate(report.summaries):
        color = _profile_color(summary.profile, idx)
        acc = max(0.0, min(5.0, summary.avg_accuracy_5))
        y = margin_top + plot_h - (acc / 5.0) * plot_h
        x_mean = margin_left + (max(0.0, summary.mean_latency_ms) / x_max) * plot_w
        x_p95 = margin_left + (max(0.0, summary.p95_latency_ms) / x_max) * plot_w

        # Connecting spread line
        lines.append(
            f'  <line x1="{x_mean:.1f}" y1="{y:.1f}" x2="{x_p95:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.5" stroke-dasharray="5,3"/>'
        )
        # Mean solid marker
        lines.append(
            f'  <circle cx="{x_mean:.1f}" cy="{y:.1f}" r="8" fill="{color}" '
            f'stroke="{COLOR_BG}" stroke-width="2"/>'
        )
        # p95 diamond marker
        d = 7.5
        diamond_pts = (
            f"{x_p95:.1f},{y - d:.1f} {x_p95 + d:.1f},{y:.1f} "
            f"{x_p95:.1f},{y + d:.1f} {x_p95 - d:.1f},{y:.1f}"
        )
        lines.append(
            f'  <polygon points="{diamond_pts}" fill="{COLOR_BG}" '
            f'stroke="{color}" stroke-width="2.4"/>'
        )

        # Callout label positioned to avoid vertical overlap across profiles
        label_y_offset = -16 if idx % 2 == 0 else 24
        label_y = max(
            margin_top + 48, min(margin_top + plot_h - 10, y + label_y_offset)
        )
        anchor_x = (x_mean + x_p95) / 2.0
        text_anchor = "middle"
        if anchor_x < margin_left + 130:
            anchor_x = max(margin_left + 10, x_mean)
            text_anchor = "start"
        elif anchor_x > margin_left + plot_w - 130:
            anchor_x = min(margin_left + plot_w - 10, x_p95)
            text_anchor = "end"

        callout = (
            f"{_short_model_label(summary, 28)} — "
            f"mean {_format_latency(summary.mean_latency_ms)}, "
            f"p95 {_format_latency(summary.p95_latency_ms)}, "
            f"acc {acc:.2f}/5"
        )
        lines.append(
            f'  <text x="{anchor_x:.1f}" y="{label_y:.1f}" text-anchor="{text_anchor}" '
            f'class="bar-label" style="fill:{color}">{_xml(callout)}</text>'
        )

    lines.append(
        f'  <text x="36" y="{height - 18}" class="note">'
        "Solid circle = mean latency; diamond = p95 tail latency; horizontal span shows latency variability."
        "</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_cost_offline_svg(report: ComparisonReport) -> str:
    """Render side-by-side Cost per 1k Queries (USD) and Offline Completion Rate (%) panels."""
    title = "Operational Cost per 1k Queries & Offline Completion Rate"
    subtitle = _report_metadata_subtitle(report)
    if not report.summaries:
        return _render_empty_chart(title, subtitle)

    width = 960
    height = 480
    panel_top = 128
    panel_bottom = 104
    plot_h = height - panel_top - panel_bottom

    left_x = 76
    panel_w = 376
    right_x = 532

    lines = _svg_header(
        width,
        height,
        title,
        "Two-panel comparison of estimated API cost per 1,000 queries and overall vs. offline completion rates.",
    )
    lines.append(f'  <text x="36" y="42" class="title">{_xml(title)}</text>')
    lines.append(f'  <text x="36" y="64" class="subtitle">{_xml(subtitle)}</text>')

    # Panel containers
    lines.append(
        f'  <text x="{left_x}" y="102" class="panel-title">Estimated Cost / 1k Queries (USD)</text>'
    )
    lines.append(
        f'  <text x="{right_x}" y="102" class="panel-title">Completion &amp; Offline Viability (%)</text>'
    )

    # Legend for Right Panel
    lines.append(
        f'  <rect x="{right_x + 216}" y="91" width="10" height="10" rx="2" fill="{COLOR_REMOTE}"/>'
    )
    lines.append(
        f'  <text x="{right_x + 230}" y="100" class="tick-label">Overall</text>'
    )
    lines.append(
        f'  <rect x="{right_x + 286}" y="91" width="10" height="10" rx="2" fill="{COLOR_LOCAL}"/>'
    )
    lines.append(
        f'  <text x="{right_x + 300}" y="100" class="tick-label">Offline</text>'
    )

    # --- LEFT PANEL: Cost per 1k Queries ---
    max_cost = max(
        (s.estimated_cost_per_1k_queries_usd for s in report.summaries),
        default=0.0,
    )
    y_cost_max = _nice_upper_bound(max_cost * 1.15 if max_cost > 0 else 1.0, 1.0)
    for i in range(5):
        val = (y_cost_max / 4.0) * i
        y = panel_top + plot_h - (val / y_cost_max) * plot_h
        stroke = COLOR_AXIS if i == 0 else COLOR_GRID
        lines.append(
            f'  <line x1="{left_x}" y1="{y:.1f}" x2="{left_x + panel_w}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{left_x - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick-label">'
            f"{_xml(_format_usd(val))}</text>"
        )

    n = len(report.summaries)
    left_slot = panel_w / n
    cost_bar_w = min(72.0, left_slot * 0.52)

    for idx, summary in enumerate(report.summaries):
        color = _profile_color(summary.profile, idx)
        cx = left_x + (idx + 0.5) * left_slot
        cost = max(0.0, summary.estimated_cost_per_1k_queries_usd)
        bar_h = (cost / y_cost_max) * plot_h if y_cost_max > 0 else 0.0
        by = panel_top + plot_h - bar_h
        bx = cx - cost_bar_w / 2.0

        if cost == 0.0:
            lines.append(
                f'  <rect x="{bx:.1f}" y="{panel_top + plot_h - 4:.1f}" width="{cost_bar_w:.1f}" '
                f'height="4" rx="2" fill="{color}"/>'
            )
            lines.append(
                f'  <text x="{cx:.1f}" y="{panel_top + plot_h - 12:.1f}" text-anchor="middle" '
                f'class="bar-label" style="fill:{COLOR_LOCAL}">$0.00 (Free/Local)</text>'
            )
        else:
            lines.append(
                f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{cost_bar_w:.1f}" '
                f'height="{max(2.0, bar_h):.1f}" rx="4" fill="{color}"/>'
            )
            lines.append(
                f'  <text x="{cx:.1f}" y="{max(panel_top - 4, by - 8):.1f}" text-anchor="middle" '
                f'class="bar-label">{_xml(_format_usd(cost))}</text>'
            )

        lines.append(
            f'  <text x="{cx:.1f}" y="{panel_top + plot_h + 24:.1f}" text-anchor="middle" '
            f'class="axis-label">{_xml(_short_model_label(summary, 20))}</text>'
        )
        lines.append(
            f'  <text x="{cx:.1f}" y="{panel_top + plot_h + 42:.1f}" text-anchor="middle" '
            f'class="note">{summary.mean_tokens_per_second:.1f} tok/s</text>'
        )

    # --- RIGHT PANEL: Completion Rate & Offline Completion Rate ---
    for pct in (0, 25, 50, 75, 100):
        y = panel_top + plot_h - (pct / 100.0) * plot_h
        stroke = COLOR_AXIS if pct == 0 else COLOR_GRID
        lines.append(
            f'  <line x1="{right_x}" y1="{y:.1f}" x2="{right_x + panel_w}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"/>'
        )
        lines.append(
            f'  <text x="{right_x - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick-label">{pct}%</text>'
        )

    right_slot = panel_w / n
    pair_gap = 8.0
    sub_bar_w = min(46.0, (right_slot * 0.68 - pair_gap) / 2.0)

    for idx, summary in enumerate(report.summaries):
        cx = right_x + (idx + 0.5) * right_slot
        overall_pct, offline_pct = _completion_rates(summary)
        group_w = 2 * sub_bar_w + pair_gap
        bx_overall = cx - group_w / 2.0
        bx_offline = bx_overall + sub_bar_w + pair_gap

        # Overall completion bar
        h_overall = (overall_pct / 100.0) * plot_h
        y_overall = panel_top + plot_h - h_overall
        lines.append(
            f'  <rect x="{bx_overall:.1f}" y="{y_overall:.1f}" width="{sub_bar_w:.1f}" '
            f'height="{max(2.0, h_overall):.1f}" rx="4" fill="{COLOR_REMOTE}"/>'
        )
        lines.append(
            f'  <text x="{bx_overall + sub_bar_w / 2:.1f}" y="{max(panel_top - 4, y_overall - 6):.1f}" '
            f'text-anchor="middle" class="bar-label">{overall_pct:.0f}%</text>'
        )

        # Offline completion bar
        h_offline = (offline_pct / 100.0) * plot_h
        y_offline = panel_top + plot_h - h_offline
        if offline_pct == 0.0:
            lines.append(
                f'  <rect x="{bx_offline:.1f}" y="{panel_top + plot_h - 4:.1f}" width="{sub_bar_w:.1f}" '
                f'height="4" rx="2" fill="{COLOR_BORDER}"/>'
            )
            lines.append(
                f'  <text x="{bx_offline + sub_bar_w / 2:.1f}" y="{panel_top + plot_h - 10:.1f}" '
                f'text-anchor="middle" class="tick-label">0%</text>'
            )
        else:
            lines.append(
                f'  <rect x="{bx_offline:.1f}" y="{y_offline:.1f}" width="{sub_bar_w:.1f}" '
                f'height="{max(2.0, h_offline):.1f}" rx="4" fill="{COLOR_LOCAL}"/>'
            )
            lines.append(
                f'  <text x="{bx_offline + sub_bar_w / 2:.1f}" y="{max(panel_top - 4, y_offline - 6):.1f}" '
                f'text-anchor="middle" class="bar-label">{offline_pct:.0f}%</text>'
            )

        ok_cases = max(0, summary.total_cases - summary.error_count)
        offline_badge = "Offline-ready" if summary.offline_capable else "Cloud only"
        lines.append(
            f'  <text x="{cx:.1f}" y="{panel_top + plot_h + 24:.1f}" text-anchor="middle" '
            f'class="axis-label">{_xml(_short_model_label(summary, 20))}</text>'
        )
        lines.append(
            f'  <text x="{cx:.1f}" y="{panel_top + plot_h + 42:.1f}" text-anchor="middle" '
            f'class="note">{ok_cases}/{summary.total_cases} ok · {_xml(offline_badge)}</text>'
        )

    lines.append(
        f'  <text x="36" y="{height - 18}" class="note">'
        "UN SDG 4.a lens: local Ollama models provide $0 marginal API cost and full offline viability for low-connectivity schools."
        "</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_all_charts(
    report: ComparisonReport,
    output_dir: Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Render all comparative SVG charts for a report into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_specs: list[tuple[str, str, str]] = [
        (
            "accuracy_pedagogy",
            f"{prefix}accuracy_pedagogy.svg",
            render_accuracy_pedagogy_svg(report),
        ),
        (
            "latency_vs_accuracy",
            f"{prefix}latency_vs_accuracy.svg",
            render_latency_accuracy_svg(report),
        ),
        (
            "cost_and_offline",
            f"{prefix}cost_and_offline.svg",
            render_cost_offline_svg(report),
        ),
    ]
    written: dict[str, Path] = {}
    for key, filename, svg_content in chart_specs:
        target = output_dir / filename
        target.write_text(svg_content, encoding="utf-8")
        written[key] = target
    return written


def load_comparison_report(report_path: Path) -> ComparisonReport:
    """Load and validate a `ComparisonReport` JSON file from disk."""
    if not report_path.exists():
        raise FileNotFoundError(f"Benchmark report JSON not found: {report_path}")
    return ComparisonReport.model_validate_json(report_path.read_text(encoding="utf-8"))


def generate_charts_from_file(
    report_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Load a benchmark JSON report and render all SVG charts offline."""
    report = load_comparison_report(report_path)
    resolved_dir = output_dir or (report_path.parent / f"{report_path.stem}_charts")
    return render_all_charts(report, resolved_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render offline SVG comparison charts from a Brasão AI benchmark JSON report."
    )
    parser.add_argument(
        "report_json",
        type=Path,
        help="Path to a benchmark ComparisonReport JSON file (e.g., evals/reports/comparison_*.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where SVG charts will be written (defaults to <report_stem>_charts alongside the report).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, str]:
    """CLI entrypoint for `python -m evals.charts <report.json>`."""
    args = _parse_args(argv)
    written = generate_charts_from_file(args.report_json, output_dir=args.output_dir)
    for name, path in written.items():
        print(f"Chart ({name}): {path}")
    return {name: str(path) for name, path in written.items()}


if __name__ == "__main__":
    main()
