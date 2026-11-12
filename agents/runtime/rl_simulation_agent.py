"""Optional Q-learning policy operating exclusively on synthetic states."""
from __future__ import annotations

import random
import time

from .contracts import AgentResult, error_result


class RLSimulationAgent:
    name = "RLSimulationAgent"
    version = "rl-synthetic-v1.0"
    actions = ("education_basics", "lifestyle_module", "simulated_followup_30d", "simulated_followup_90d")
    states = tuple((risk, engagement) for risk in range(3) for engagement in range(3))

    def __init__(self, seed: int = 42):
        self.q = {state: [0.0] * len(self.actions) for state in self.states}
        self._train(seed)

    @staticmethod
    def _reward(state, action):
        risk, engagement = state
        if action == 0:
            return 1.2 if engagement == 0 else 0.5
        if action == 1:
            return 1.1 if risk >= 1 else 0.6
        if action == 2:
            return 1.4 if risk == 2 else 0.4
        return 1.0 if risk == 0 else 0.3

    def _train(self, seed):
        rng = random.Random(seed)
        alpha, gamma, epsilon = 0.18, 0.85, 0.2
        for _ in range(1200):
            state = rng.choice(self.states)
            action = rng.randrange(len(self.actions)) if rng.random() < epsilon else max(range(len(self.actions)), key=self.q[state].__getitem__)
            reward = self._reward(state, action)
            next_state = (state[0], min(2, state[1] + (1 if action in (0, 1) else 0)))
            target = reward + gamma * max(self.q[next_state])
            self.q[state][action] += alpha * (target - self.q[state][action])

    def execute(self, request_id: str, state: object) -> AgentResult:
        started = time.perf_counter()
        if not isinstance(state, dict):
            return error_result(request_id, self.name, self.version, "INVALID_SIMULATION_STATE",
                                "simulation_state phải là JSON object.", status="invalid")
        risk_map = {"low": 0, "moderate": 1, "high": 2}
        engagement_map = {"low": 0, "moderate": 1, "high": 2}
        risk = risk_map.get(str(state.get("risk_band", "moderate")).lower())
        engagement = engagement_map.get(str(state.get("engagement", "moderate")).lower())
        if risk is None or engagement is None:
            return error_result(request_id, self.name, self.version, "INVALID_SIMULATION_STATE",
                                "risk_band và engagement chỉ nhận low/moderate/high.", status="invalid")
        index = max(range(len(self.actions)), key=self.q[(risk, engagement)].__getitem__)
        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload={"simulation_only": True, "state": {"risk_band": risk, "engagement": engagement},
                     "action": self.actions[index], "q_values": [round(v, 4) for v in self.q[(risk, engagement)]],
                     "prohibited_uses": ["prescribe_medication", "change_treatment", "control_real_patient_care"]},
            evidence=[{"type": "synthetic_policy", "id": self.version,
                       "rationale": "Q-learning trên môi trường thưởng tổng hợp; không học từ bệnh nhân thật."}],
            version=self.version,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
