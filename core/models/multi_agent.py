import torch
import torch.optim as optim
from core.envs.multigraph_env import MultiGraphEnv
from core.models.multi_policy import PlannerPolicy, TruckPolicy

class MultiAgentREINFORCE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.planner = PlannerPolicy(
            node_dim=3,
            embed_dim=cfg.embed_dim,
            num_trucks=cfg.num_trucks
        ).to(self.device)

        self.trucks = torch.nn.ModuleList([
            TruckPolicy(node_dim=3, embed_dim=cfg.embed_dim).to(self.device)
            for _ in range(cfg.num_trucks)
        ])

        self.optimizer = optim.Adam(
            list(self.planner.parameters()) +
            list(self.trucks.parameters()),
            lr=cfg.lr
        )

        self.planner_log_probs = []
        self.truck_log_probs = []
        self.rewards = []

    def act(self, state, eval_mode=False):
        node_features = state["node_features"].to(self.device)
        edge_index = state["edge_index"].to(self.device)
        edge_attr = state["edge_attr"].to(self.device)
        truck_pos = state["truck_pos"].to(self.device)

        T = self.cfg.num_trucks

        # Planner
        planner_probs = self.planner(node_features, edge_index, edge_attr)

        planner_actions = []
        planner_log_probs_step = []

        for t in range(T):
            dist = torch.distributions.Categorical(planner_probs[t])
            if eval_mode:
                action = torch.argmax(planner_probs[t])
            else:
                action = dist.sample()
                planner_log_probs_step.append(dist.log_prob(action))
            planner_actions.append(action)

        # Trucks
        truck_actions = []
        truck_log_probs_step = []

        for t in range(T):
            probs = self.trucks[t](
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                truck_pos=truck_pos[t],
                planner_suggestion=planner_actions[t]
            )

            dist = torch.distributions.Categorical(probs)
            if eval_mode:
                action = torch.argmax(probs)
            else:
                action = dist.sample()
                truck_log_probs_step.append(dist.log_prob(action))

            truck_actions.append(action.item())

        if not eval_mode:
            self.planner_log_probs.append(torch.stack(planner_log_probs_step))
            self.truck_log_probs.append(torch.stack(truck_log_probs_step))

        return truck_actions

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = 0
        returns = []

        for r in reversed(self.rewards):
            R = r + R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []

        for t_step, R_t in enumerate(returns):
            lp_planner = self.planner_log_probs[t_step].sum()
            lp_trucks = self.truck_log_probs[t_step].sum()
            total_lp = lp_planner + lp_trucks
            policy_loss.append(-total_lp * R_t)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        self.planner_log_probs = []
        self.truck_log_probs = []
        self.rewards = []

        return loss.item()

    def evaluate_mode(self, cfg):
        env = MultiGraphEnv(cfg)
        state, _ = env.reset()
        terminated = False
        total_reward = 0

        routes = [[state["truck_pos"][i].item()] for i in range(cfg.num_trucks)]

        self.planner.eval()
        for t in self.trucks:
            t.eval()

        with torch.no_grad():
            while not terminated:
                actions = self.act(state, eval_mode=True)
                state, reward, terminated, truncated, _ = env.step(actions)
                total_reward += reward

                for i in range(cfg.num_trucks):
                    routes[i].append(state["truck_pos"][i].item())

        return total_reward, routes