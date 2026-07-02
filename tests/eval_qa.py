"""
QA evaluation script for heta memory system.

Usage:
    python tests/eval_qa.py                    # run all questions
    python tests/eval_qa.py --out results.json # also save raw JSON
    python tests/eval_qa.py -q 1 3 5           # run specific question numbers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# ── allow running from repo root without installing ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heta.config.io import load_config
from heta.mem.recall import recall

# ── QA definitions ───────────────────────────────────────────────────────────

QA_CASES: list[dict] = [
    # ── L2 基础事实（冲突消解）──────────────────────────────────────────────
    {
        "id": 1,
        "category": "L2-冲突消解",
        "question": "陈浩现在在哪家公司工作？",
        "expected": "星图数据（极光科技已被覆盖）",
        "keywords": ["星图数据"],
        "anti_keywords": ["极光科技"],
    },
    {
        "id": 2,
        "category": "L2-冲突消解",
        "question": "陈浩现在住在哪里？",
        "expected": "望京（经历过：朝阳区→海淀区→望京）",
        "keywords": ["望京"],
        "anti_keywords": ["朝阳", "海淀"],
    },
    {
        "id": 3,
        "category": "L2-冲突消解",
        "question": "陈浩现在的薪资是多少？",
        "expected": "28k（薪资链：18k→22k→25k→28k）",
        "keywords": ["28"],
        "anti_keywords": ["18k", "22k", "25k"],
    },
    {
        "id": 4,
        "category": "L2-冲突消解",
        "question": "陈浩现在的通勤时间是多少？",
        "expected": "步行10分钟（经历过：1小时→20分钟→步行10分钟）",
        "keywords": ["10分钟", "步行"],
        "anti_keywords": ["1小时", "20分钟"],
    },
    {
        "id": 5,
        "category": "L2-累加事实",
        "question": "陈浩喜欢什么运动？",
        "expected": "爬山、羽毛球、偶尔篮球、游泳（兴趣爱好应累加保留）",
        "keywords": ["爬山", "羽毛球"],
        "anti_keywords": [],
    },
    {
        "id": 6,
        "category": "L2-累加事实",
        "question": "陈浩在学什么新技术？",
        "expected": "Rust（用于高性能数据处理）；需要加强Go经验",
        "keywords": ["Rust"],
        "anti_keywords": [],
    },
    {
        "id": 7,
        "category": "L2-基础属性",
        "question": "陈浩的绩效评级是什么？",
        "expected": "A绩效（上个季度，在极光科技）",
        "keywords": ["A"],
        "anti_keywords": [],
    },
    # ── L1 情景事件 ────────────────────────────────────────────────────────
    {
        "id": 8,
        "category": "L1-情景事件",
        "question": "陈浩参加了什么技术会议？会议的主题是什么？",
        "expected": "公司技术分享会，主题是大模型在工程中的落地，约50人参加",
        "keywords": ["大模型", "技术分享", "50"],
        "anti_keywords": [],
    },
    {
        "id": 9,
        "category": "L1-情景事件",
        "question": "陈浩和李薇开会讨论了什么？",
        "expected": "用户画像模块改版需求评审，双方有争议，定三周后上线",
        "keywords": ["用户画像", "李薇"],
        "anti_keywords": [],
    },
    {
        "id": 10,
        "category": "L1-情景事件",
        "question": "陈浩在极光科技最后一天发生了什么？",
        "expected": "完成了所有交接，和团队一起吃了散伙饭，感觉不舍",
        "keywords": ["散伙饭", "交接"],
        "anti_keywords": [],
    },
    {
        "id": 11,
        "category": "L1-情景事件",
        "question": "陈浩最近去哪里旅游了？和谁一起去的？大概花了多少钱？",
        "expected": "青岛，和王强、赵敏、刘洋，三天，每人约1500元，住市南区民宿",
        "keywords": ["青岛", "王强", "1500"],
        "anti_keywords": [],
    },
    {
        "id": 12,
        "category": "L1-事件序列",
        "question": "陈浩妈妈的健康情况如何？",
        "expected": "最初血压高，去协和医院检查，服降压药；后来复查血压稳定，已停药",
        "keywords": ["血压", "协和"],
        "anti_keywords": [],
    },
    # ── 时间推理 ───────────────────────────────────────────────────────────
    {
        "id": 13,
        "category": "时间推理",
        "question": "用户画像模块最终是什么时候上线的？经历了哪些波折？",
        "expected": "比原计划晚了近两个月；需求评审定三周后→推迟到下下个月（前端资源不足）→最终上线，首日UV2万",
        "keywords": ["推迟", "上线"],
        "anti_keywords": [],
    },
    {
        "id": 14,
        "category": "时间推理",
        "question": "陈浩什么时候离开极光科技的？",
        "expected": "拿到offer后下周一入职星图数据；在极光科技的最后一天完成交接",
        "keywords": ["极光科技", "最后"],
        "anti_keywords": [],
    },
    {
        "id": 15,
        "category": "时间推理",
        "question": "陈浩妈妈的降压药大概吃了多久？",
        "expected": "大约一个月（医生开了一个月的药，复查后停药）",
        "keywords": ["一个月", "停药"],
        "anti_keywords": [],
    },
    # ── 复合推理 ───────────────────────────────────────────────────────────
    {
        "id": 16,
        "category": "复合推理",
        "question": "陈浩的职业发展轨迹是什么？",
        "expected": "极光科技后端工程师（18k，绩效A）→裁员担忧→加入星图数据（25k→28k，Go/Rust技术栈）",
        "keywords": ["极光科技", "星图数据"],
        "anti_keywords": [],
    },
    {
        "id": 17,
        "category": "复合推理",
        "question": "陈浩换工作的原因是什么？",
        "expected": "公司宣布裁员10%有担忧；拿到星图数据更高薪资的offer（25k）",
        "keywords": ["裁员", "星图数据"],
        "anti_keywords": [],
    },
    {
        "id": 18,
        "category": "复合推理",
        "question": "陈浩目前的生活状态怎么样？",
        "expected": "住望京，在星图数据工作（28k），学Rust和Go，游泳，妈妈健康稳定",
        "keywords": ["望京", "星图数据", "28"],
        "anti_keywords": [],
    },
    {
        "id": 19,
        "category": "复合推理",
        "question": "用户画像模块这个项目经历了哪些波折？",
        "expected": "需求评审有争议→定三周后上线→推迟到下下个月（前端资源）→上线首日UV2万",
        "keywords": ["推迟", "前端"],
        "anti_keywords": [],
    },
    # ── 边界 / 负样本 ──────────────────────────────────────────────────────
    {
        "id": 20,
        "category": "边界-无记忆",
        "question": "陈浩有没有去过上海？",
        "expected": "没有相关记忆",
        "keywords": [],
        "anti_keywords": ["上海"],
        "expect_no_memory": True,
    },
    {
        "id": 21,
        "category": "边界-无记忆",
        "question": "陈浩结婚了吗？",
        "expected": "没有相关记忆",
        "keywords": [],
        "anti_keywords": [],
        "expect_no_memory": True,
    },
    {
        "id": 22,
        "category": "边界-基础属性",
        "question": "陈浩今年多少岁？",
        "expected": "28岁",
        "keywords": ["28"],
        "anti_keywords": [],
    },
    {
        "id": 23,
        "category": "边界-历史状态",
        "question": "陈浩之前说要涨薪到22k，这个涨薪最终兑现了吗？",
        "expected": "没有明确记录兑现；后来换工作了，该涨薪应已被覆盖",
        "keywords": ["22k", "换工作"],
        "anti_keywords": [],
    },
    {
        "id": 24,
        "category": "边界-细节检索",
        "question": "陈浩带妈妈去哪家医院看的病？",
        "expected": "协和医院",
        "keywords": ["协和"],
        "anti_keywords": [],
    },
    {
        "id": 25,
        "category": "边界-细节检索",
        "question": "陈浩在极光科技认识了哪些人？",
        "expected": "产品经理李薇；技术分享会上认识了做推理优化的同事（无具体名字）",
        "keywords": ["李薇"],
        "anti_keywords": [],
    },
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QAResult:
    id: int
    category: str
    question: str
    expected: str
    actual_answer: str
    layer_ranking: list[str]
    keyword_hit: bool
    anti_keyword_hit: bool
    auto_pass: bool   # keyword-based heuristic
    elapsed_s: float
    error: str = ""


# ── scoring ───────────────────────────────────────────────────────────────────

def _check_keywords(answer: str, keywords: list[str], anti_keywords: list[str]) -> tuple[bool, bool]:
    a = answer.lower()
    hit = all(kw.lower() in a for kw in keywords) if keywords else True
    anti_hit = any(kw.lower() in a for kw in anti_keywords) if anti_keywords else False
    return hit, anti_hit


def _auto_pass(case: dict, answer: str) -> bool:
    """Heuristic pass: all keywords present AND no anti-keywords."""
    if case.get("expect_no_memory"):
        no_mem_phrases = ["no relevant memory", "没有相关记忆", "没有记录", "未找到", "无相关"]
        return any(p in answer.lower() for p in no_mem_phrases)
    hit, anti = _check_keywords(answer, case["keywords"], case["anti_keywords"])
    return hit and not anti


# ── main ──────────────────────────────────────────────────────────────────────

def run_eval(question_ids: list[int] | None = None) -> list[QAResult]:
    config = load_config()
    if config is None:
        print("[ERROR] Heta is not initialised. Run `heta init` first.", file=sys.stderr)
        sys.exit(1)

    cases = QA_CASES if not question_ids else [c for c in QA_CASES if c["id"] in question_ids]

    results: list[QAResult] = []
    for case in cases:
        print(f"  Q{case['id']:02d} [{case['category']}] {case['question']}", end=" ... ", flush=True)
        t0 = time.time()
        error = ""
        answer = ""
        ranking: list[str] = []
        try:
            result = recall(case["question"], config)
            answer = result.answer
            ranking = result.ranking
        except Exception as exc:
            error = str(exc)
            answer = ""
        elapsed = round(time.time() - t0, 2)

        hit, anti = _check_keywords(answer, case["keywords"], case["anti_keywords"])
        passed = _auto_pass(case, answer)

        status = "PASS" if passed else "FAIL"
        print(f"{status}  ({elapsed}s)")

        results.append(QAResult(
            id=case["id"],
            category=case["category"],
            question=case["question"],
            expected=case["expected"],
            actual_answer=answer,
            layer_ranking=ranking,
            keyword_hit=hit,
            anti_keyword_hit=anti,
            auto_pass=passed,
            elapsed_s=elapsed,
            error=error,
        ))
    return results


def print_report(results: list[QAResult]) -> None:
    passed = sum(1 for r in results if r.auto_pass)
    total = len(results)
    print()
    print("=" * 70)
    print(f" RESULT: {passed}/{total} passed  ({100*passed//total}%)")
    print("=" * 70)

    # group by category
    by_cat: dict[str, list[QAResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat, group in by_cat.items():
        cat_pass = sum(1 for r in group if r.auto_pass)
        print(f"\n── {cat}  ({cat_pass}/{len(group)}) ──")
        for r in group:
            icon = "✓" if r.auto_pass else "✗"
            print(f"  {icon} Q{r.id:02d}: {r.question}")
            if not r.auto_pass:
                print(f"       期望: {r.expected}")
                print(f"       实际: {r.actual_answer[:200]}")
                if r.error:
                    print(f"       错误: {r.error}")

    avg_t = sum(r.elapsed_s for r in results) / len(results) if results else 0
    print(f"\n平均响应时间: {avg_t:.1f}s")


def save_results(results: list[QAResult], path: str) -> None:
    data = [asdict(r) for r in results]
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存至 {path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate heta memory QA")
    parser.add_argument("-q", "--questions", nargs="*", type=int, metavar="N",
                        help="Only run these question IDs (e.g. -q 1 3 5)")
    parser.add_argument("--out", metavar="FILE",
                        help="Save raw JSON results to this file")
    args = parser.parse_args()

    print(f"Running {len(QA_CASES) if not args.questions else len(args.questions)} QA cases...\n")
    results = run_eval(args.questions)
    print_report(results)
    if args.out:
        save_results(results, args.out)
