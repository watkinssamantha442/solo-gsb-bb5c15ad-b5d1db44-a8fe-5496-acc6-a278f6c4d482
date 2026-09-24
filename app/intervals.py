"""数值区间预测报告：把揭示原文解析为「下界..上界」区间并做十进制精确覆盖判定。

纯函数模块：不依赖数据库与 HTTP，可独立单测。

区间文本语法（严格匹配，不允许空白、下划线、Infinity/NaN 等宽松形式）：

    interval       := finite_decimal ".." finite_decimal
    finite_decimal := [+-]? ( 数字+ ["." 数字*] | "." 数字+ ) [ (e|E) [+-]? 数字+ ]

解析出的两个数均为**有限十进制数**（decimal.Decimal 精确表示），且要求
下界 <= 上界；不满足任一条件即判为无效。覆盖判定（闭区间）与区间宽度
均用 decimal.Decimal 精确计算，全程不经过浮点。
"""

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID

# 有限十进制数的严格语法：显式排除 Decimal 构造器宽容接受的
# 空白、下划线、Infinity 与 NaN
FINITE_DECIMAL_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

# 区间分隔符：恰好一个 ".."，把原文分成下界与上界两段
SEPARATOR = ".."

# 无效原因
INVALID_FORMAT = "INVALID_FORMAT"  # 不是「有限十进制数..有限十进制数」
LOWER_BOUND_EXCEEDS_UPPER = "LOWER_BOUND_EXCEEDS_UPPER"  # 下界大于上界


def parse_finite_decimal(text: str) -> Decimal | None:
    """把字符串解析为有限十进制数；语法非法或非有限（Infinity/NaN）返回 None。"""
    if not FINITE_DECIMAL_PATTERN.fullmatch(text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        # 语法合法但指数超出 Decimal 可表示范围（如 1e999999999999999999999）
        return None
    if not value.is_finite():
        return None
    return value


@dataclass(frozen=True)
class ParsedInterval:
    """一条成功解析的区间：边界原文逐字保留，数值与宽度为精确十进制。"""

    lower_text: str
    upper_text: str
    lower: Decimal
    upper: Decimal
    width: Decimal  # upper - lower（>= 0，精确十进制）


def parse_interval(text: str) -> tuple[ParsedInterval | None, str | None]:
    """把揭示原文解析为「下界..上界」区间。

    返回 (区间, None) 或 (None, 原因)；原因取 INVALID_FORMAT（不是两个
    有限十进制数）或 LOWER_BOUND_EXCEEDS_UPPER（下界大于上界）。
    """
    parts = text.split(SEPARATOR)
    if len(parts) != 2:
        return None, INVALID_FORMAT
    lower_text, upper_text = parts
    lower = parse_finite_decimal(lower_text)
    upper = parse_finite_decimal(upper_text)
    if lower is None or upper is None:
        return None, INVALID_FORMAT
    if lower > upper:
        return None, LOWER_BOUND_EXCEEDS_UPPER
    return ParsedInterval(lower_text, upper_text, lower, upper, upper - lower), None


@dataclass(frozen=True)
class IntervalEntry:
    """一条合法区间的报告条目（命中或未命中）。"""

    commitment_id: UUID
    committed_at: datetime
    lower_text: str
    upper_text: str
    width: Decimal
    hit: bool


@dataclass(frozen=True)
class InvalidIntervalEntry:
    """一条无法解析为合法区间的揭示。"""

    commitment_id: UUID
    committed_at: datetime
    original_text: str
    reason: str  # INVALID_FORMAT | LOWER_BOUND_EXCEEDS_UPPER


@dataclass(frozen=True)
class IntervalReport:
    """区间报告的确定性结果：hits 已排名，misses / invalid 仅稳定排序。"""

    hits: list[IntervalEntry]  # 按 (宽度, 承诺提交时间, 承诺 ID) 升序，名次 = 下标 + 1
    misses: list[IntervalEntry]  # 按 (承诺提交时间, 承诺 ID) 升序，不排名
    invalid: list[InvalidIntervalEntry]  # 按 (承诺提交时间, 承诺 ID) 升序
    unrevealed_count: int  # 承诺存在但从未揭示的数量


def build_interval_report(
    standard: Decimal,
    revealed: list[tuple[UUID, datetime, str]],
    unrevealed_count: int,
) -> IntervalReport:
    """对全部已揭示原文做区间解析与覆盖判定，返回确定性排序的报告。

    - standard：结算标准结果（有限十进制数，由调用方解析）；
    - revealed：(承诺 ID, 承诺提交时间, 揭示原文) 三元组，输入顺序任意；
    - 覆盖判定为闭区间 lower <= standard <= upper，十进制精确比较；
    - 命中者按 (区间宽度, 承诺提交时间, 承诺 ID) 升序排名（名次 = 下标 + 1）；
    - 未命中者与无效条目分别按 (承诺提交时间, 承诺 ID) 稳定排序，不排名。
    """
    hits: list[IntervalEntry] = []
    misses: list[IntervalEntry] = []
    invalid: list[InvalidIntervalEntry] = []
    for commitment_id, committed_at, original_text in revealed:
        parsed, reason = parse_interval(original_text)
        if parsed is None:
            invalid.append(
                InvalidIntervalEntry(commitment_id, committed_at, original_text, reason)
            )
            continue
        entry = IntervalEntry(
            commitment_id=commitment_id,
            committed_at=committed_at,
            lower_text=parsed.lower_text,
            upper_text=parsed.upper_text,
            width=parsed.width,
            hit=parsed.lower <= standard <= parsed.upper,
        )
        (hits if entry.hit else misses).append(entry)

    # 承诺 ID 取 UUID 128 位整数排序，与承诺集合证明的叶序一致
    hits.sort(key=lambda e: (e.width, e.committed_at, e.commitment_id.int))
    misses.sort(key=lambda e: (e.committed_at, e.commitment_id.int))
    invalid.sort(key=lambda e: (e.committed_at, e.commitment_id.int))
    return IntervalReport(
        hits=hits, misses=misses, invalid=invalid, unrevealed_count=unrevealed_count
    )
