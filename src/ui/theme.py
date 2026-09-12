"""
Giao diện: CSS dùng chung và các thành phần hiển thị nhỏ.

Chỉ dùng lớp CSS do chính dự án đặt tên (tiền tố `hc-`) và các biến theme
của Streamlit, không bám vào tên lớp nội bộ của Streamlit — như vậy nâng
cấp Streamlit không làm vỡ giao diện.
"""

import streamlit as st

from ui.labels import risk_color, risk_icon, risk_level_vi


CSS = """
<style>
  /* ----- khung trang ----- */
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

  /* ----- tiêu đề ứng dụng ----- */
  .hc-hero {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: .4rem 1rem;
    padding: 1.1rem 1.3rem; margin-bottom: 1.1rem;
    border: 1px solid var(--hc-line); border-left: 4px solid var(--primary-color);
    border-radius: 8px; background: var(--hc-surface);
  }
  .hc-hero h1 { margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; }
  .hc-hero p { margin: 0; opacity: .72; font-size: .95rem; }

  /* ----- thẻ ----- */
  .hc-card {
    border: 1px solid var(--hc-line); border-radius: 8px;
    background: var(--hc-surface); padding: 1rem 1.15rem; margin-bottom: .9rem;
  }
  .hc-card h4 {
    margin: 0 0 .55rem 0; font-size: .78rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase; opacity: .62;
  }

  /* ----- huy hiệu mức nguy cơ ----- */
  .hc-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .18rem .7rem; border-radius: 999px;
    font-size: .82rem; font-weight: 700; letter-spacing: .02em;
    border: 1.5px solid currentColor;
  }

  /* ----- thẻ kết quả từng bệnh ----- */
  .hc-risk {
    border: 1px solid var(--hc-line); border-radius: 8px;
    background: var(--hc-surface); padding: .95rem 1.05rem; height: 100%;
    border-top: 3px solid var(--hc-accent);
  }
  .hc-risk .name { font-size: .95rem; font-weight: 600; margin-bottom: .35rem; }
  .hc-risk .pct {
    font-size: 2rem; font-weight: 700; line-height: 1.1;
    font-variant-numeric: tabular-nums; letter-spacing: -.02em;
  }
  .hc-risk .bar {
    height: 7px; border-radius: 999px; margin: .55rem 0 .5rem 0;
    background: color-mix(in srgb, var(--hc-accent) 18%, transparent);
    overflow: hidden;
  }
  .hc-risk .bar span { display: block; height: 100%; border-radius: 999px; background: var(--hc-accent); }

  /* ----- dải tóm tắt tổng thể ----- */
  .hc-summary {
    display: flex; flex-wrap: wrap; align-items: center; gap: .6rem 1.6rem;
    padding: .95rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;
    background: color-mix(in srgb, var(--hc-accent) 9%, transparent);
    border: 1px solid color-mix(in srgb, var(--hc-accent) 42%, transparent);
  }
  .hc-summary .lead { font-size: 1.02rem; font-weight: 600; }
  .hc-summary .item { font-size: .9rem; opacity: .82; }
  .hc-summary .item b { font-variant-numeric: tabular-nums; }

  /* ----- trạng thái rỗng ----- */
  .hc-empty {
    border: 1.5px dashed var(--hc-line); border-radius: 10px;
    padding: 2rem 1.2rem; text-align: center; opacity: .8;
  }
  .hc-empty .ico { font-size: 1.9rem; display: block; margin-bottom: .45rem; }
  .hc-empty .t { font-weight: 600; margin-bottom: .2rem; }
  .hc-empty .d { font-size: .9rem; opacity: .75; }

  /* ----- chú thích nhỏ ----- */
  .hc-note { font-size: .86rem; opacity: .72; }

  /* ----- nút chính to hơn ----- */
  .stButton > button { border-radius: 7px; font-weight: 600; }

  /* ----- tab dễ bấm hơn ----- */
  .stTabs [data-baseweb="tab"] { font-weight: 600; padding: .55rem .2rem; }

  /* ----- biến màu theo theme sáng / tối ----- */
  :root { --hc-line: rgba(16,30,39,.14); --hc-surface: rgba(255,255,255,.72); }
  @media (prefers-color-scheme: dark) {
    :root { --hc-line: rgba(230,239,243,.16); --hc-surface: rgba(255,255,255,.035); }
  }
</style>
"""


def inject_css():
    """Nạp CSS dùng chung (gọi một lần ở đầu trang)."""

    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    """Tiêu đề ứng dụng."""

    st.markdown(
        f'<div class="hc-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def risk_badge(level):
    """Huy hiệu mức nguy cơ có màu."""

    color = risk_color(level)

    return (
        f'<span class="hc-badge" style="color:{color}">'
        f'{risk_icon(level)} {risk_level_vi(level)}</span>'
    )


def risk_card(disease_name, probability, level):
    """Thẻ kết quả cho một bệnh: tên, xác suất, thanh mức, huy hiệu."""

    color = risk_color(level)
    percent = max(0.0, min(1.0, float(probability))) * 100

    st.markdown(
        f'<div class="hc-risk" style="--hc-accent:{color}">'
        f'<div class="name">{disease_name}</div>'
        f'<div class="pct" style="color:{color}">{percent:.1f}%</div>'
        f'<div class="bar"><span style="width:{percent:.1f}%"></span></div>'
        f'{risk_badge(level)}'
        f'</div>',
        unsafe_allow_html=True
    )


def summary_bar(overall_level, priority_name, high_count, moderate_count):
    """Dải tóm tắt kết quả tổng thể."""

    color = risk_color(overall_level)

    st.markdown(
        f'<div class="hc-summary" style="--hc-accent:{color}">'
        f'<span class="lead">Nguy cơ tổng thể: {risk_badge(overall_level)}</span>'
        f'<span class="item">Cần ưu tiên: <b>{priority_name}</b></span>'
        f'<span class="item">Bệnh nguy cơ cao: <b>{high_count}</b></span>'
        f'<span class="item">Nguy cơ trung bình: <b>{moderate_count}</b></span>'
        f'</div>',
        unsafe_allow_html=True
    )


def empty_state(icon, title, description):
    """Khối hướng dẫn khi chưa có dữ liệu."""

    st.markdown(
        f'<div class="hc-empty"><span class="ico">{icon}</span>'
        f'<div class="t">{title}</div><div class="d">{description}</div></div>',
        unsafe_allow_html=True
    )


def card_title(text):
    """Nhãn nhóm trường nhập liệu."""

    st.markdown(
        f'<div class="hc-card"><h4>{text}</h4></div>',
        unsafe_allow_html=True
    )


def note(text):
    """Dòng chú thích nhỏ."""

    st.markdown(f'<p class="hc-note">{text}</p>', unsafe_allow_html=True)
