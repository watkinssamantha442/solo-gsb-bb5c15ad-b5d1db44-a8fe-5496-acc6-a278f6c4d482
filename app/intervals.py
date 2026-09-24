"""数值区间预测报告：把揭示原文解析为「下界..上界」区间并做十进制精确覆盖判定。

纯函数模块：不依赖数据库与 HTTP，可独立单测。

区间文本语法（严格匹配，不允许空白、下划线、Infinity/NaN 等宽松形式）：

    interval       := finite_decimal ".." finite_decimal
    finite_decimal := [+-]? ( 数字+ ["." 数字*] | "." 数字+ ) [ (e|E) [+-]? 数字+ ]

解析出的两个数均为**有限十进制数**（decimal.Decimal 精确表示），且要求
下界 <= 上界；不满足任一条件即判为无效。覆盖判定（闭区间）用
decimal.Decimal 的**精确比较**（比较不经过上下文舍入），区间宽度则由
整数系数运算得出**无舍入的精确差值**——不依赖 Decimal 上下文精度：
默认 28 位精度会把超过 28 位有效数字的合法宽度截成同一值，既丢末位又
可能把较宽区间排到较窄者之前。全程不经过浮点。

精确宽度按**无舍入减法的自然表示**保存（对齐到更细指数的整数系数
差值，保留操作数精度）；其系数有效位数超过 ``MAX_WIDTH_DIGITS``
（4096）时抛出 :class:`IntervalWidthTooLargeError`，由接口层转为
明确的 422 错误，而不是静默舍入或在渲染时触发 500。
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

# 精确宽度系数允许的最大有效位数：超过即拒绝，避免静默舍入与超长响应
MAX_WIDTH_DIGITS = 4096

# 无效原因
INVALID_FORMAT = "INVALID_FORMAT"  # 不是「有限十进制数..有限十进制数」
LOWER_BOUND_EXCEEDS_UPPER = "LOWER_BOUND_EXCEEDS_UPPER"  # 下界大于上界


def parse_finite_decimal(text: str) -> Decimal | None:
    """把字符串解析为有限十进制数；语法非法或非有限（Infinity/NaN）返回 None。

    Decimal 由字符串/元组构造时**不经过上下文精度舍入**，任意位数的有限
    十进制端点都按原文精确保存；仅在指数超出 Decimal 可表示范围时失败。
    """
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


def _signed_coefficient(value: Decimal) -> int:
    """取 Decimal 的带符号整数系数（significand），如 -12.300 -> -12300。"""
    sign, digits, _ = value.as_tuple()
    coefficient = int("".join(str(d) for d in digits))
    return -coefficient if sign else coefficient


def _decimal_from_int(coefficient: int, exponent: int, sign: int = 0) -> Decimal:
    """由带符号整数系数与指数经元组构造 Decimal（不经过上下文精度舍入）。

    系数为 0 时保留调用方给定的符号位（精确减法的自然结果可能是 -0）。
    """
    if coefficient == 0:
        return Decimal((sign, (0,), exponent))
    return Decimal(
        (0 if coefficient > 0 else 1, tuple(int(d) for d in str(abs(coefficient))), exponent)
    )


class IntervalWidthTooLargeError(ValueError):
    """精确区间宽度的系数有效位数超过 MAX_WIDTH_DIGITS。"""

    def __init__(
        self,
        lower_text: str,
        upper_text: str,
        digit_count: int,
        commitment_id: UUID | None = None,
    ):
        self.lower_text = lower_text
        self.upper_text = upper_text
        # 经结构预判快速拒绝时为位数下界，实际精确计算后为准确位数
        self.digit_count = digit_count
        self.commitment_id = commitment_id
        super().__init__(
            f"区间 {lower_text}..{upper_text} 的精确宽度至少有 {digit_count} 位有效数字，"
            f"超过上限 {MAX_WIDTH_DIGITS} 位"
        )


def _exact_difference_within_limit(lower: Decimal, upper: Decimal) -> Decimal:
    """无舍入计算 upper - lower，并在结果系数位数超限时快速拒绝。

    对齐到更细（更小）的指数后做整数系数减法，结果保留与无限精度
    Decimal 算术相同的自然表示（含操作数尾随 0），且**不执行**会被
    上下文精度截断的 Decimal 运算。对指数差巨大的输入先按对齐后位数
    做结构预判：若粗操作数对齐后比细操作数多至少 2 位，则差值不可能
    借位抵消到更短，可在不构造 10**shift 巨型中间整数的前提下拒绝
    （避免 ``-1..1e999999999`` 这类短输入拖垮进程、冒泡成 500）。
    """
    lc = _signed_coefficient(lower)
    uc = _signed_coefficient(upper)
    le = lower.as_tuple().exponent
    ue = upper.as_tuple().exponent

    if le == ue:
        difference = uc - lc
        exponent = le
        # 零结果符号遵循减法通行约定（与 Decimal 同指数算术一致）：非零
        # 相等数相减恒为正 0（-1 - -1 = 0）；两个操作数均为 0 时仅当
        # 被减数为 -0、减数为 +0 才得 -0（-0 - 0 = -0），其余均为 +0
        zero_sign = 0
        if difference == 0 and uc == 0 and lc == 0:
            zero_sign = upper.as_tuple().sign & (1 - lower.as_tuple().sign)
        return _decimal_from_int(difference, exponent, sign=zero_sign)

    if le < ue:
        # 下界更细：上界系数放大到下界指数，difference = uc*10**shift - lc
        shift = ue - le
        exponent = le
        n_coarse_digits = len(upper.as_tuple().digits) + shift
        fine_c = lc
    else:
        # 上界更细：下界系数放大到上界指数，difference = uc - lc*10**shift
        shift = le - ue
        exponent = ue
        n_coarse_digits = len(lower.as_tuple().digits) + shift
        fine_c = uc

    # 粗操作数对齐后比细操作数多至少 2 位时，加/减均无法借位抵消到
    # aligned_digits - 2 位以下：结果位数 >= aligned_digits - 1。
    # 据此在构造 10**shift（可能是 10**999999999）之前直接拒绝。
    fine_digits = 0 if fine_c == 0 else len(str(abs(fine_c)))
    if fine_digits <= n_coarse_digits - 2 and n_coarse_digits - 1 > MAX_WIDTH_DIGITS:
        raise IntervalWidthTooLargeError(str(lower), str(upper), n_coarse_digits - 1)

    # 预判未命中时才构造放大系数；此时中间整数规模不超过输入长度
    if le < ue:
        difference = uc * 10**shift - lc
    else:
        difference = uc - lc * 10**shift
    # 零结果符号遵循减法通行约定：数值相等的非零数相减恒为正 0
    # （-0.2 - -0.20 = 0.00）；两个操作数本身均为 0 时仅当被减数为
    # -0、减数为 +0 才得 -0，其余为 +0（-0 与 +0 数值相等，不影响
    # 宽度输出、排名与覆盖判定）
    zero_sign = 0
    if difference == 0 and uc == 0 and lc == 0:
        zero_sign = upper.as_tuple().sign & (1 - lower.as_tuple().sign)
    result = _decimal_from_int(difference, exponent, sign=zero_sign)
    digit_count = len(result.as_tuple().digits)
    if digit_count > MAX_WIDTH_DIGITS:
        raise IntervalWidthTooLargeError(str(lower), str(upper), digit_count)
    return result


def exact_decimal_difference(upper: Decimal, lower: Decimal) -> Decimal:
    """计算 ``upper - lower``（upper >= lower）的**无舍入**精确十进制差值。

    Decimal 算术（``upper - lower``）会把结果舍入到当前上下文精度（默认
    28 位有效数字），例如::

        Decimal("0.123456789012345678901234567891") - Decimal("0")
        # 默认精度下得到 0.1234567890123456789012345679 —— 末位被舍掉，
        # 与 ...7890 的宽度塌缩为同一值

    这里对齐指数后做**整数系数减法**（经元组构造），不经任何上下文
    舍入，结果与无限精度 Decimal 算术的自然表示一致（如
    ``2.00 - 2.0 = 0.00``、``0..10`` 的宽度为 ``10``）。有效系数位数
    超过 :data:`MAX_WIDTH_DIGITS` 时抛出
    :class:`IntervalWidthTooLargeError`。
    """
    return _exact_difference_within_limit(lower, upper)


@dataclass(frozen=True)
class ParsedInterval:
    """一条成功解析的区间：边界原文逐字保留，宽度为精确（无舍入）十进制。"""

    lower_text: str
    upper_text: str
    lower: Decimal
    upper: Decimal
    width: Decimal  # upper - lower（>= 0，精确十进制，不经过上下文精度舍入）


def parse_interval(text: str) -> tuple[ParsedInterval | None, str | None]:
    """把揭示原文解析为「下界..上界」区间。

    返回 (区间, None) 或 (None, 原因)；原因取 INVALID_FORMAT（不是两个
    有限十进制数）或 LOWER_BOUND_EXCEEDS_UPPER（下界大于上界）。
    精确宽度超过 MAX_WIDTH_DIGITS 位时抛出 IntervalWidthTooLargeError。
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
    width = exact_decimal_difference(upper, lower)
    return ParsedInterval(lower_text, upper_text, lower, upper, width), None


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
    - 覆盖判定为闭区间 lower <= standard <= upper，十进制精确比较
      （Decimal 比较不经上下文舍入，任意位数端点均逐位比较）；
    - 区间宽度为无舍入精确值：合法端点如
      ``0..0.123456789012345678901234567891`` 与
      ``0..0.123456789012345678901234567890`` 得到相差末位的两个宽度，
      不会在默认 28 位精度下塌缩为同一值；
    - 命中者按 (区间真实宽度, 承诺提交时间, 承诺 ID) 升序排名
      （名次 = 下标 + 1，较宽区间不会因舍入而排到较窄者之前）；
    - 未命中者与无效条目分别按 (承诺提交时间, 承诺 ID) 稳定排序，不排名；
    - 任一区间的精确宽度有效位数超过 MAX_WIDTH_DIGITS 时抛出
      IntervalWidthTooLargeError（异常带承诺 ID，由调用方转为 422，
      绝不静默舍入）。
    """
    hits: list[IntervalEntry] = []
    misses: list[IntervalEntry] = []
    invalid: list[InvalidIntervalEntry] = []
    for commitment_id, committed_at, original_text in revealed:
        try:
            parsed, reason = parse_interval(original_text)
        except IntervalWidthTooLargeError as exc:
            # 携带承诺上下文向上抛出，供接口层生成明确的 422
            exc.commitment_id = commitment_id
            raise
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

    # 承诺 ID 取 UUID 128 位整数排序，与承诺集合证明的叶序一致；
    # Decimal 比较逐位精确，不受上下文精度影响
    hits.sort(key=lambda e: (e.width, e.committed_at, e.commitment_id.int))
    misses.sort(key=lambda e: (e.committed_at, e.commitment_id.int))
    invalid.sort(key=lambda e: (e.committed_at, e.commitment_id.int))
    return IntervalReport(
        hits=hits, misses=misses, invalid=invalid, unrevealed_count=unrevealed_count
    )
