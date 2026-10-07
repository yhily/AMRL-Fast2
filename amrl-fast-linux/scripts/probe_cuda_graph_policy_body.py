#!/usr/bin/env python3
"""Screen CUDA-graph capture of the deterministic B=1 PPO policy body.

Only the fixed-shape feature-to-(action-logits,value) neural body is captured.
Categorical distribution validation, action sampling, rollout buffer handling,
PPO loss, backward, and optimiser all remain in the standard SB3 path.  This
avoids suppressing distribution checks while reducing repeated small-kernel
launches in the actual rollout hotspot.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import PPOBaseAgent, set_global_seed
from amrl_fast.base_agent_backend import _metric
from amrl_fast.envs import make_env


class GraphPolicyBody:
    def __init__(self, policy, *, feature_dim: int) -> None:
        self.policy = policy
        self.static_features = torch.empty((1, feature_dim), device="cuda")
        self.graph = torch.cuda.CUDAGraph()
        # Warm up allocation paths outside capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(8):
                latent_pi, latent_vf = policy.mlp_extractor(self.static_features)
                policy.action_net(latent_pi); policy.value_net(latent_vf)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad(), torch.cuda.graph(self.graph):
            latent_pi, latent_vf = policy.mlp_extractor(self.static_features)
            self.static_logits = policy.action_net(latent_pi)
            self.static_values = policy.value_net(latent_vf)

    @torch.no_grad()
    def __call__(self, features):
        self.static_features.copy_(features)
        self.graph.replay()
        return self.static_logits, self.static_values


def _sync(): torch.cuda.synchronize()


def _time(fn, *, warmup=200, repeats=4000):
    with torch.no_grad():
        for _ in range(warmup): fn()
        _sync(); started=time.perf_counter()
        for _ in range(repeats): fn()
        _sync()
    return (time.perf_counter()-started)*1e6/repeats


def main():
    set_global_seed(0)
    env, resolved = make_env("CartPole-v1", seed=0)
    try:
        agent=PPOBaseAgent(env, device=torch.device("cuda"), seed=0, rollout_steps=5000)
        policy=agent.model.policy
        input_dim=policy.mlp_extractor.policy_net[0].in_features
        x=torch.randn(1,input_dim,device="cuda",requires_grad=True)
        graph=GraphPolicyBody(policy,feature_dim=input_dim)
        with torch.no_grad():
            pi,vf=policy.mlp_extractor(x.detach()); reference_logits=policy.action_net(pi); reference_values=policy.value_net(vf)
            candidate_logits,candidate_values=graph(x.detach())
            standard_us=_time(lambda: (lambda pi_vf: (policy.action_net(pi_vf[0]), policy.value_net(pi_vf[1])))(policy.mlp_extractor(x.detach())))
            graph_us=_time(lambda: graph(x.detach()))
        # The captured path is inference-only. PPO backward stays in the
        # unmodified evaluate_actions path; report its source-gradient check.
        pi,vf=policy.mlp_extractor(x); loss=(policy.action_net(pi).square().mean()+policy.value_net(vf).square().mean()); loss.backward()
        grad=torch.cat([p.grad.detach().reshape(-1) for p in policy.mlp_extractor.parameters()]); policy.mlp_extractor.zero_grad(set_to_none=True)
        payload={
            "development_only":True,"environment":"CartPole-v1","resolved_environment_id":resolved,
            "candidate":"cuda-graph(policy MLP + action/value heads; distribution outside graph)",
            "input_shape":[1,input_dim],"standard_body_forward_us":standard_us,"graph_body_forward_us":graph_us,
            "speedup":standard_us/graph_us,
            "logits":_metric(reference_logits,candidate_logits),"values":_metric(reference_values,candidate_values),
            "ppo_backward_path":"standard SB3 evaluate_actions; no gradient-bearing operation is substituted",
            "source_gradient_finite":bool(torch.isfinite(grad).all().item()),
        }
        payload["admitted"]=bool(payload["speedup"]>=1.02 and all(payload[k]["correlation"]>=.999 and payload[k]["relative_l2"]<=5e-3 and payload[k]["max_absolute"]<=.1 for k in ("logits","values")) and payload["source_gradient_finite"])
        print(json.dumps(payload,indent=2,sort_keys=True))
        out=ROOT/'artifacts'/'base_agent_hotspots'/'cuda_graph_policy_body_probe.json'; out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    finally: env.close()


if __name__=='__main__': main()
