"""Hạ tầng đa agent: Message, Blackboard (bộ nhớ chung), MessageBus, Agent."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Message:
    sender: str
    receiver: str
    performative: str          # REQUEST | INFORM | CRITIQUE ...
    action: str                # tên hành động -> agent gọi hàm on_<action>
    content: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class Blackboard:
    """Bộ nhớ chung giữa các agent (dữ liệu lớn nằm ở đây, message chỉ mang tham chiếu)."""

    def __init__(self):
        self.data = {}         # ds -> dict(X_train, X_test, y_train, y_test, numeric, categorical)
        self.profiles = {}     # ds -> hồ sơ dữ liệu do DataAgent lập
        self.rounds = {}       # ds -> [kết quả từng vòng huấn luyện]
        self.verdicts = {}     # ds -> [nhận xét của Agent 2 theo vòng]
        self.models = {}       # (ds, round) -> pipeline đã huấn luyện
        self.final = {}        # ds -> kết quả cuối được chọn
        self.ablation = {}     # ds -> kết quả kiểm tra loại bỏ đặc trưng
        self.comparison = {}   # kết quả so sánh dataset
        self.report_files = {}


class MessageBus:
    def __init__(self):
        self.agents = {}
        self.queue = deque()
        self.trace = []

    def register(self, agent):
        self.agents[agent.name] = agent
        agent.bus = self

    def send(self, msg: Message):
        self.queue.append(msg)
        self.trace.append(msg)

    def run(self, max_steps: int = 10_000):
        steps = 0
        while self.queue and steps < max_steps:
            msg = self.queue.popleft()
            replies = self.agents[msg.receiver].handle(msg) or []
            for r in replies:
                self.send(r)
            steps += 1

    def dump(self, path):
        def brief(v):
            if isinstance(v, dict):
                return {k: brief(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [brief(x) for x in v[:12]]
            if isinstance(v, float):
                return round(v, 4)
            if isinstance(v, str):
                return v[:300]
            return v if isinstance(v, (int, bool)) or v is None else str(v)[:120]

        with open(path, "w", encoding="utf-8") as f:
            for i, m in enumerate(self.trace, 1):
                d = asdict(m)
                d["seq"] = i
                d["content"] = brief(d["content"])
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


class Agent:
    name = "Agent"

    def __init__(self, bb: Blackboard, cfg):
        self.bb, self.cfg, self.bus = bb, cfg, None

    def say(self, text: str):
        print(f"[{self.name}] {text}", flush=True)

    def msg(self, to: str, performative: str, action: str, **content) -> Message:
        return Message(self.name, to, performative, action, content)

    def handle(self, m: Message):
        fn = getattr(self, f"on_{m.action}", None)
        if fn is None:
            self.say(f"không hiểu hành động '{m.action}' từ {m.sender}")
            return []
        return fn(m)
