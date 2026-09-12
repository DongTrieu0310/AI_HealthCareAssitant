"""
Nhãn hiển thị tiếng Việt.

Các engine giữ nguyên khoá tiếng Anh (LOW / MODERATE / HIGH và khoá bệnh);
chỉ phần chữ hiện lên màn hình mới được dịch.
"""

RISK_LEVEL_VI = {
    "LOW": "THẤP",
    "MODERATE": "TRUNG BÌNH",
    "HIGH": "CAO",
}

DISEASE_NAME_VI = {
    "cardio": "Bệnh tim mạch",
    "cardiovascular": "Bệnh tim mạch",
    "diabetes": "Đái tháo đường",
    "hypertension": "Tăng huyết áp",
}

# Màu ngữ nghĩa cho từng mức nguy cơ.
RISK_COLOR = {
    "LOW": "#0F7A4A",
    "MODERATE": "#9A6100",
    "HIGH": "#B3261E",
}

RISK_ICON = {
    "LOW": "🟢",
    "MODERATE": "🟡",
    "HIGH": "🔴",
}


def risk_level_vi(level):
    """Hiển thị mức nguy cơ bằng tiếng Việt."""

    return RISK_LEVEL_VI.get(str(level).upper(), level)


def disease_name_vi(key):
    """Hiển thị tên bệnh bằng tiếng Việt."""

    return DISEASE_NAME_VI.get(str(key).lower(), key)


def risk_color(level):
    """Màu tương ứng với mức nguy cơ."""

    return RISK_COLOR.get(str(level).upper(), "#5B7280")


def risk_icon(level):
    """Biểu tượng tương ứng với mức nguy cơ."""

    return RISK_ICON.get(str(level).upper(), "⚪")
