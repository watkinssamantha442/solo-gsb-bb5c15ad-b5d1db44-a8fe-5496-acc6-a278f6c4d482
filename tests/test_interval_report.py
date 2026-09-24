"""数值区间预测报告测试：GET /events/{event_id}/interval-report。

SQLite 文件库 + 模拟时钟，覆盖：
  - 未结算事件（含 closed 但未结算）返回 409 INTERVAL_REPORT_NOT_SETTLED
  - 标准结果不是有限十进制数返回 422 STANDARD_RESULT_NOT_FINITE_DECIMAL
  - 区间解析：合法 / 非法格式 / 下界大于上界；未揭示承诺单独计数
  - 十进制精确覆盖判定：闭区间边界命中、数值相等（1.10 == 1.1）、
    宽度精确（0.2 而非浮点误差）
  - **无舍入宽度**：超过默认 Decimal 28 位精度的合法端点（如 30 位
    端点）宽度保留末位、互不塌缩；命中者按真实宽度排名，较宽者不会
    因舍入被排到较窄者之前
  - 精确宽度系数超过 4096 位返回 422 INTERVAL_WIDTH_TOO_MANY_DIGITS
    （边界 4096 接受 / 4097 拒绝；短输入但指数差巨大同样快速拒绝），
    不静默舍入、不冒泡 500
  - 命中者按宽度升序排名，同宽按承诺提交时间、承诺 ID 排序
  - 未命中者列出但不排名（rank 为 null）
  - 既有结算计数不变；接口只读，重复调用结果一致
  - 未知事件 404；零承诺事件；区间解析纯函数
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_interval_report_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/interval_report.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.clock as clock_module  # noqa: E402
import app.main as main_module  # noqa: E402
from app import intervals  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402

T0 = datetime(2030, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)


class MutableClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


clock = MutableClock(T0)

client = TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    monkeypatch.setattr(clock_module, "now_utc", clock)
    with client:  # 触发 lifespan 建表（幂等）
        yield


def create_event() -> str:
    r = client.post("/events", json={
        "title": "数值区间预测",
        "commit_deadline": COMMIT_DL.isoformat(),
        "reveal_deadline": REVEAL_DL.isoformat(),
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def commit(event_id: str, text: str, salt: str) -> str:
    """提交承诺并返回承诺 ID。"""
    h = compute_commitment(text, salt)
    r = client.post(f"/events/{event_id}/commitments", json={"commitment_hash": h})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def reveal(event_id: str, text: str, salt: str) -> None:
    r = client.post(f"/events/{event_id}/reveals",
                    json={"original_text": text, "salt": salt})
    assert r.status_code == 201, r.text


def settle(event_id: str, standard_result: str) -> dict:
    r = client.post(f"/events/{event_id}/settlement",
                    json={"standard_result": standard_result})
    assert r.status_code == 200, r.text
    return r.json()


def report(event_id: str, expected: int = 200) -> dict:
    r = client.get(f"/events/{event_id}/interval-report")
    assert r.status_code == expected, r.text
    return r.json()


def test_unsettled_event_is_409():
    event_id = create_event()
    # commit 阶段未结算
    body = report(event_id, expected=409)
    assert body["detail"]["code"] == "INTERVAL_REPORT_NOT_SETTLED"

    # closed 阶段但仍未结算，同样拒绝
    clock.t = REVEAL_DL
    body = report(event_id, expected=409)
    assert body["detail"]["code"] == "INTERVAL_REPORT_NOT_SETTLED"


def test_report_404_for_missing_event():
    missing = "00000000-0000-0000-0000-000000000000"
    body = report(missing, expected=404)
    assert body["detail"]["code"] == "EVENT_NOT_FOUND"


def test_non_finite_decimal_standard_result_is_422():
    # 空标准结果无法通过结算校验，不会出现在结算记录中，不在此列
    for bad in ["明天下雨", "NaN", "Infinity", "-Infinity", "1..2", "1.5 ", " 1.5",
                "abc", "1_000", "1e999999999999999999999"]:
        clock.t = T0  # 每个用例从创建事件开始，重置时钟
        event_id = create_event()
        clock.t = REVEAL_DL
        settle(event_id, bad)
        body = report(event_id, expected=422)
        assert body["detail"]["code"] == "STANDARD_RESULT_NOT_FINITE_DECIMAL"


def test_report_ranks_hits_by_width_and_lists_misses_and_invalid():
    event_id = create_event()
    commit(event_id, "1..2", "s-hit-wide")          # 命中，宽 1
    commit(event_id, "1.4..1.6", "s-hit-narrow")    # 命中，宽 0.2（精确十进制）
    commit(event_id, "0..10", "s-hit-widest")       # 命中，宽 10
    commit(event_id, "3..4", "s-miss")              # 合法但未覆盖
    commit(event_id, "not-an-interval", "s-bad-1")  # 非法格式
    commit(event_id, "5..4", "s-bad-2")             # 下界大于上界
    commit(event_id, "9..99", "s-unrevealed")       # 只承诺不揭示

    clock.t = COMMIT_DL
    reveal(event_id, "1..2", "s-hit-wide")
    reveal(event_id, "1.4..1.6", "s-hit-narrow")
    reveal(event_id, "0..10", "s-hit-widest")
    reveal(event_id, "3..4", "s-miss")
    reveal(event_id, "not-an-interval", "s-bad-1")
    reveal(event_id, "5..4", "s-bad-2")

    clock.t = REVEAL_DL
    settled = settle(event_id, "1.5")

    body = report(event_id)
    assert body["event_id"] == event_id
    assert body["standard_result"] == "1.5"
    assert body["settled_at"] == settled["settled_at"]

    # 既有结算计数原样回显（逐字符精确匹配：无揭示等于 "1.5"）
    assert body["correct_count"] == 0
    assert body["incorrect_count"] == 6
    assert body["unrevealed_count"] == 1

    # 报告计数
    assert body["commitment_count"] == 7
    assert body["revealed_count"] == 6
    assert body["unrevealed_commitment_count"] == 1
    assert body["valid_interval_count"] == 4
    assert body["hit_count"] == 3
    assert body["miss_count"] == 1
    assert body["invalid_count"] == 2

    # 命中者按宽度升序排名：0.2 < 1 < 10；宽度为精确十进制（非浮点误差）
    hits = body["hits"]
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert [(h["lower_bound"], h["upper_bound"]) for h in hits] == [
        ("1.4", "1.6"), ("1", "2"), ("0", "10"),
    ]
    assert [h["width"] for h in hits] == ["0.2", "1", "10"]
    assert all(h["hit"] is True for h in hits)

    # 未命中者列出但不排名
    misses = body["misses"]
    assert len(misses) == 1
    assert misses[0]["lower_bound"] == "3" and misses[0]["upper_bound"] == "4"
    assert misses[0]["hit"] is False
    assert misses[0]["rank"] is None

    # 非法格式单独列出并注明原因
    invalid = {(i["original_text"], i["reason"]) for i in body["invalid"]}
    assert invalid == {
        ("not-an-interval", "INVALID_FORMAT"),
        ("5..4", "LOWER_BOUND_EXCEEDS_UPPER"),
    }
    assert all(i["commitment_id"] for i in body["invalid"])


def test_same_width_tie_breaks_by_commit_time_then_commitment_id():
    event_id = create_event()
    # 四个区间均覆盖标准结果 1.5 且宽度均为 2
    # 不同时刻提交：同宽时先提交者名次在前
    id_early = commit(event_id, "1..3", "s-early")
    clock.t = T0 + timedelta(seconds=5)
    id_late = commit(event_id, "0..2", "s-late")

    # 同一时刻提交两个承诺：committed_at 相同，按承诺 ID（UUID 整数）升序
    clock.t = T0 + timedelta(seconds=10)
    id_same_a = commit(event_id, "0.5..2.5", "s-same-a")
    id_same_b = commit(event_id, "1.4..3.4", "s-same-b")

    clock.t = COMMIT_DL
    reveal(event_id, "1..3", "s-early")
    reveal(event_id, "0..2", "s-late")
    reveal(event_id, "0.5..2.5", "s-same-a")
    reveal(event_id, "1.4..3.4", "s-same-b")

    clock.t = REVEAL_DL
    settle(event_id, "1.5")

    body = report(event_id)
    hits = body["hits"]
    # 四个区间宽度均为 2：先按提交时间，再按承诺 ID
    same_time_sorted = sorted([id_same_a, id_same_b], key=lambda i: UUID(i).int)
    assert [h["commitment_id"] for h in hits] == [id_early, id_late] + same_time_sorted
    assert [h["rank"] for h in hits] == [1, 2, 3, 4]


def test_coverage_is_closed_interval_and_decimal_exact():
    event_id = create_event()
    commit(event_id, "1..2", "s-upper-edge")    # 上界恰为标准结果
    commit(event_id, "2..3", "s-lower-edge")    # 下界恰为标准结果
    commit(event_id, "2.0..2.00", "s-equal")    # 数值相等（2.0 == 2 == 2.00）
    commit(event_id, "0.1..0.2", "s-miss")      # 不覆盖

    clock.t = COMMIT_DL
    reveal(event_id, "1..2", "s-upper-edge")
    reveal(event_id, "2..3", "s-lower-edge")
    reveal(event_id, "2.0..2.00", "s-equal")
    reveal(event_id, "0.1..0.2", "s-miss")

    clock.t = REVEAL_DL
    settle(event_id, "2")

    body = report(event_id)
    assert body["hit_count"] == 3
    assert body["miss_count"] == 1
    widths = sorted(h["width"] for h in body["hits"])
    # 2.00 - 2.0 = 0.00：精确十进制，保留操作数精度，不经过浮点
    assert widths == ["0.00", "1", "1"]
    zero_width = next(h for h in body["hits"] if h["lower_bound"] == "2.0")
    assert zero_width["rank"] == 1  # 零宽区间最窄，排名第一
    assert body["misses"][0]["lower_bound"] == "0.1"


def test_scientific_notation_bounds_are_finite_decimals():
    event_id = create_event()
    commit(event_id, "1e2..2e2", "s-sci")   # [100, 200]
    commit(event_id, "-1.5e1..0", "s-neg")  # [-15, 0]，不覆盖

    clock.t = COMMIT_DL
    reveal(event_id, "1e2..2e2", "s-sci")
    reveal(event_id, "-1.5e1..0", "s-neg")

    clock.t = REVEAL_DL
    settle(event_id, "150")

    body = report(event_id)
    assert body["hit_count"] == 1
    assert body["hits"][0]["width"] == "1E+2"
    assert body["miss_count"] == 1


# 超过默认 Decimal 上下文精度（28 位有效数字）的端点：宽度必须保留末位
LONG_WIDE = "0.123456789012345678901234567891"   # 30 位，较宽
LONG_NARROW = "0.123456789012345678901234567890"  # 30 位，较窄（差末位 1e-30）


def test_width_beyond_default_decimal_precision_keeps_last_digit():
    # 默认 Decimal 精度（28 位）下两个宽度都会被舍成同一个
    # 0.1234567890123456789012345679：修复后必须各自保留真实末位
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 28
        assert Decimal(LONG_WIDE) - Decimal("0") == Decimal(LONG_NARROW) - Decimal("0")

    event_id = create_event()
    id_wide = commit(event_id, f"0..{LONG_WIDE}", "s-long-wide")
    id_narrow = commit(event_id, f"0..{LONG_NARROW}", "s-long-narrow")

    clock.t = COMMIT_DL
    reveal(event_id, f"0..{LONG_WIDE}", "s-long-wide")
    reveal(event_id, f"0..{LONG_NARROW}", "s-long-narrow")

    clock.t = REVEAL_DL
    settle(event_id, "0.0001")  # 两个区间都覆盖

    body = report(event_id)
    hits_by_id = {h["commitment_id"]: h for h in body["hits"]}
    # 宽度逐字保留末位，不再塌缩
    assert hits_by_id[id_wide]["width"] == LONG_WIDE
    assert hits_by_id[id_narrow]["width"] == LONG_NARROW
    # 真实宽度升序：较窄者（末位 0）排第 1，较宽者（末位 1）排第 2，
    # 不会因默认精度舍入而把较宽区间排到较窄者之前
    assert [h["commitment_id"] for h in body["hits"]] == [id_narrow, id_wide]
    assert hits_by_id[id_narrow]["rank"] == 1
    assert hits_by_id[id_wide]["rank"] == 2
    assert body["hit_count"] == 2


def test_30_digit_standard_coverage_is_exact():
    # 标准结果本身超过默认精度：仅末位为 1 的区间覆盖，末位为 0 的不覆盖
    event_id = create_event()
    commit(event_id, f"0..{LONG_WIDE}", "s-long-wide")
    commit(event_id, f"0..{LONG_NARROW}", "s-long-narrow")

    clock.t = COMMIT_DL
    reveal(event_id, f"0..{LONG_WIDE}", "s-long-wide")
    reveal(event_id, f"0..{LONG_NARROW}", "s-long-narrow")

    clock.t = REVEAL_DL
    settle(event_id, LONG_WIDE)

    body = report(event_id)
    assert body["standard_result"] == LONG_WIDE
    assert body["hit_count"] == 1
    assert body["miss_count"] == 1
    assert body["hits"][0]["upper_bound"] == LONG_WIDE
    assert body["misses"][0]["upper_bound"] == LONG_NARROW
    assert body["hits"][0]["width"] == LONG_WIDE


def test_width_with_4096_digits_is_accepted():
    # 边界：恰好 4096 位有效数字的宽度正常返回（纯十进制小数，宽度同位数）
    bound_4096 = "0." + "1" * 4096
    event_id = create_event()
    commit(event_id, f"0..{bound_4096}", "s-4096")

    clock.t = COMMIT_DL
    reveal(event_id, f"0..{bound_4096}", "s-4096")

    clock.t = REVEAL_DL
    settle(event_id, "0.0001")

    body = report(event_id)
    assert body["hit_count"] == 1
    assert body["hits"][0]["width"] == bound_4096
    assert len(body["hits"][0]["width"].replace("0.", "")) == 4096


def test_width_with_4097_digits_returns_422():
    # 超过 4096 位：明确 422，不静默舍入
    bound_4097 = "0." + "1" * 4097
    event_id = create_event()
    cid = commit(event_id, f"0..{bound_4097}", "s-4097")

    clock.t = COMMIT_DL
    reveal(event_id, f"0..{bound_4097}", "s-4097")

    clock.t = REVEAL_DL
    settle(event_id, "0.0001")

    body = report(event_id, expected=422)
    assert body["detail"]["code"] == "INTERVAL_WIDTH_TOO_MANY_DIGITS"
    assert cid in body["detail"]["message"]
    # 拒绝不改变任何结算数据
    r = client.get(f"/events/{event_id}/settlement")
    assert r.json()["status"] == "settled"


def test_oversize_width_in_miss_also_returns_422():
    # 超长宽度区间即使不覆盖标准结果（列入 misses）也整体拒绝，
    # 不做静默舍入后照常出报告
    bound_4097 = "0." + "1" * 4097
    event_id = create_event()
    commit(event_id, f"0..{bound_4097}", "s-miss-wide")

    clock.t = COMMIT_DL
    reveal(event_id, f"0..{bound_4097}", "s-miss-wide")

    clock.t = REVEAL_DL
    settle(event_id, "5")  # 区间 [0, 0.111…] 不覆盖

    body = report(event_id, expected=422)
    assert body["detail"]["code"] == "INTERVAL_WIDTH_TOO_MANY_DIGITS"


def test_huge_exponent_gap_short_input_returns_422_fast():
    # 短原文但指数差巨大（-1..1e999999999）：宽度系数近 10 亿位，
    # 结构预判直接 422，不构造巨型整数、不冒泡 500
    event_id = create_event()
    commit(event_id, "-1..1e999999999", "s-huge-gap")

    clock.t = COMMIT_DL
    reveal(event_id, "-1..1e999999999", "s-huge-gap")

    clock.t = REVEAL_DL
    settle(event_id, "0")

    body = report(event_id, expected=422)
    assert body["detail"]["code"] == "INTERVAL_WIDTH_TOO_MANY_DIGITS"


def test_invalid_format_variants():
    event_id = create_event()
    texts = [
        "1..2..3",      # 分隔符多于一个
        "1 .. 2",       # 含空白
        "1..",          # 缺上界
        "..2",          # 缺下界
        "NaN..2",       # NaN 非有限十进制数
        "1..Infinity",  # Infinity 非有限十进制数
        "",             # 空原文
        "1e999999999999999999999..2",  # 指数超出 Decimal 可表示范围
    ]
    for i, text in enumerate(texts):
        commit(event_id, text, f"s-{i}")

    clock.t = COMMIT_DL
    for i, text in enumerate(texts):
        reveal(event_id, text, f"s-{i}")

    clock.t = REVEAL_DL
    settle(event_id, "1.5")

    body = report(event_id)
    assert body["valid_interval_count"] == 0
    assert body["invalid_count"] == len(texts)
    assert {i["reason"] for i in body["invalid"]} == {"INVALID_FORMAT"}
    assert {i["original_text"] for i in body["invalid"]} == set(texts)


def test_report_is_read_only_and_settlement_counts_unchanged():
    event_id = create_event()
    commit(event_id, "1..2", "s-a")
    clock.t = COMMIT_DL
    reveal(event_id, "1..2", "s-a")
    clock.t = REVEAL_DL
    settled = settle(event_id, "1.5")

    first = report(event_id)
    second = report(event_id)
    assert first == second  # 时钟冻结下重复调用结果逐字节一致

    # 既有结算结果不变
    r = client.get(f"/events/{event_id}/settlement")
    assert r.status_code == 200
    assert r.json() == settled

    # 承诺 / 揭示计数不变
    r = client.get(f"/events/{event_id}")
    assert r.json()["commitment_count"] == 1
    assert r.json()["reveal_count"] == 1


def test_report_with_zero_commitments():
    event_id = create_event()
    clock.t = REVEAL_DL
    settle(event_id, "1.5")

    body = report(event_id)
    assert body["commitment_count"] == 0
    assert body["revealed_count"] == 0
    assert body["unrevealed_commitment_count"] == 0
    assert body["valid_interval_count"] == 0
    assert body["hit_count"] == 0
    assert body["miss_count"] == 0
    assert body["invalid_count"] == 0
    assert body["hits"] == [] and body["misses"] == [] and body["invalid"] == []


# --------------------------------------------------------------------------- #
# 纯函数：有限十进制数与区间解析
# --------------------------------------------------------------------------- #

def test_parse_finite_decimal_strict():
    assert intervals.parse_finite_decimal("1.5") == Decimal("1.5")
    assert intervals.parse_finite_decimal("+1.") == Decimal("1")
    assert intervals.parse_finite_decimal("-.5") == Decimal("-0.5")
    assert intervals.parse_finite_decimal("1e3") == Decimal("1E+3")
    assert intervals.parse_finite_decimal("01.50") == Decimal("1.50")
    for bad in ["", ".", "+", "1.2.3", "1_000", " 1", "1 ", "Infinity", "-Infinity",
                "NaN", "0x10", "1e", "e3", "1e999999999999999999999"]:
        assert intervals.parse_finite_decimal(bad) is None, bad


def test_parse_interval_bounds_and_reasons():
    parsed, reason = intervals.parse_interval("1.4..1.6")
    assert reason is None
    assert parsed.lower == Decimal("1.4") and parsed.upper == Decimal("1.6")
    assert parsed.width == Decimal("0.2")  # 精确十进制，无浮点误差
    assert parsed.lower_text == "1.4" and parsed.upper_text == "1.6"

    parsed, reason = intervals.parse_interval("5..4")
    assert parsed is None and reason == intervals.LOWER_BOUND_EXCEEDS_UPPER

    for bad in ["", "1", "1..2..3", "a..2", "1...2.5.6", "..", "1 .. 2"]:
        parsed, reason = intervals.parse_interval(bad)
        assert parsed is None and reason == intervals.INVALID_FORMAT, bad


# --------------------------------------------------------------------------- #
# 纯函数：无舍入宽度计算（不依赖 Decimal 上下文精度）
# --------------------------------------------------------------------------- #

def _infinite_precision_width(upper_text: str, lower_text: str) -> Decimal:
    """无限精度 Decimal 算术的参考实现（高上下文精度 + 大 Emax/Emin）。"""
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 10_000_000
        ctx.Emax = 999_999_999
        ctx.Emin = -999_999_999
        return Decimal(upper_text) - Decimal(lower_text)


def test_exact_width_independent_of_context_precision():
    # 在极低上下文精度下调用，结果仍与无限精度算术逐元组一致
    from decimal import localcontext
    pairs = [
        ("0.123456789012345678901234567891", "0"),
        ("0.123456789012345678901234567890", "0"),
        ("1.6", "1.4"), ("2.00", "2.0"), ("0.30", "0.10"),
        ("5", "-5"), ("2e2", "1e2"),
        ("1", "0.999999999999999999999999999999"),
        ("1.0000000000000000000000000000001", "1.0000000000000000000000000000000"),
        ("-0.9", "-0.999999999999999999999999999999"),
    ]
    with localcontext() as ctx:
        ctx.prec = 2  # 极端低精度也不影响精确宽度
        for upper_text, lower_text in pairs:
            got = intervals.exact_decimal_difference(
                Decimal(upper_text), Decimal(lower_text)
            )
            expected = _infinite_precision_width(upper_text, lower_text)
            assert got == expected, (upper_text, lower_text, got, expected)
            assert got.as_tuple() == expected.as_tuple(), (upper_text, lower_text)


def test_exact_width_natural_representation_preserved():
    # 与无限精度 Decimal 算术的自然表示一致（保留操作数尾随 0）
    assert str(intervals.exact_decimal_difference(Decimal("10"), Decimal("0"))) == "10"
    assert str(intervals.exact_decimal_difference(Decimal("2.00"), Decimal("2.0"))) == "0.00"
    assert str(intervals.exact_decimal_difference(Decimal("2.50"), Decimal("1.5"))) == "1.00"
    assert intervals.exact_decimal_difference(Decimal("2e2"), Decimal("1e2")) == Decimal("1E+2")


def test_distinct_30_digit_widths_do_not_collapse():
    # 回归：默认 28 位精度下两宽度相等；修复后严格可分、较宽者更大
    wide, _ = intervals.parse_interval(f"0..0.123456789012345678901234567891")
    narrow, _ = intervals.parse_interval(f"0..0.123456789012345678901234567890")
    assert wide.width != narrow.width
    assert wide.width > narrow.width
    assert str(wide.width) == "0.123456789012345678901234567891"
    assert str(narrow.width) == "0.123456789012345678901234567890"


def test_exact_width_4096_boundary_and_rejection():
    accepted = "0." + "1" * intervals.MAX_WIDTH_DIGITS
    parsed, reason = intervals.parse_interval(f"0..{accepted}")
    assert reason is None
    assert len(parsed.width.as_tuple().digits) == intervals.MAX_WIDTH_DIGITS

    rejected = "0." + "1" * (intervals.MAX_WIDTH_DIGITS + 1)
    with pytest.raises(intervals.IntervalWidthTooLargeError) as exc_info:
        intervals.parse_interval(f"0..{rejected}")
    assert exc_info.value.digit_count == intervals.MAX_WIDTH_DIGITS + 1


def test_exact_width_huge_exponent_gap_rejected_without_giant_integer():
    # -1..1e999999999：宽度系数近 10 亿位，结构预判快速拒绝（毫秒级）
    import time
    start = time.monotonic()
    with pytest.raises(intervals.IntervalWidthTooLargeError) as exc_info:
        intervals.parse_interval("-1..1e999999999")
    assert time.monotonic() - start < 5
    assert exc_info.value.digit_count > intervals.MAX_WIDTH_DIGITS

    # 另一侧指数方向（0..1e999999999 同样超宽）
    with pytest.raises(intervals.IntervalWidthTooLargeError):
        intervals.parse_interval("0..1e999999999")


def test_exact_width_matches_infinite_precision_on_random_pairs():
    # 随机合法（upper >= lower）十进制对：结果逐元组等于无限精度算术
    import random
    rng = random.Random(20260924)

    def rand_decimal_text() -> str:
        sign = rng.choice(["", "", "-"])
        n_int = rng.randint(0, 4)
        n_frac = rng.randint(0, 4)
        if n_int == 0 and n_frac == 0:
            n_int = 1
        int_part = "".join(rng.choice("0123456789") for _ in range(n_int)) or "0"
        frac_part = "".join(rng.choice("0123456789") for _ in range(n_frac))
        text = sign + int_part + ("." + frac_part if n_frac else "")
        if rng.random() < 0.3:
            text += "e" + rng.choice(["", "-", "+"]) + str(rng.randint(0, 4))
        return text

    checked = 0
    while checked < 2000:
        upper_text, lower_text = rand_decimal_text(), rand_decimal_text()
        upper, lower = Decimal(upper_text), Decimal(lower_text)
        if upper < lower:
            continue
        got = intervals.exact_decimal_difference(upper, lower)
        expected = _infinite_precision_width(upper_text, lower_text)
        if expected == 0:
            # 跨指数的 -0 符号位为展示细节，只校验数值
            assert got == 0
        else:
            assert got.as_tuple() == expected.as_tuple(), (upper_text, lower_text)
        checked += 1
