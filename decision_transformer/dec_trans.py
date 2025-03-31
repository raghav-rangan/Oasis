import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import transformers
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import numpy as np
import sys

class DecisionTransformerData(Dataset):
    def __init__(self, states, actions, rewards_to_go, execution_times, timesteps, seq_len=144):
        self.states = states
        self.actions = actions
        self.rewards_to_go = rewards_to_go
        self.execution_times = execution_times
        self.timesteps = timesteps
        self.seq_len = seq_len

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        reward_to_go = torch.tensor(self.rewards_to_go[idx], dtype=torch.float32)
        execution_time = torch.tensor(self.execution_times[idx], dtype=torch.float32)

        combined_action = torch.zeros(action.shape[0], 2)
        combined_action[:, 0] = action.squeeze(-1)  # Submit decision
        combined_action[:, 1] = execution_time.squeeze(-1)

        rewards = torch.clone(reward_to_go)
        rewards[:143, :] = 0
        timesteps = torch.tensor(self.timesteps[idx], dtype=torch.int32)
        attn_mask = torch.ones(state.shape[0], dtype=torch.float32)
        attn_mask[-1] = 0

        if len(state) < self.seq_len:
            pad_size = self.seq_len - len(state)
            state = torch.cat([state, torch.zeros(pad_size)])
            reward_to_go = torch.cat([reward_to_go, torch.zeros(pad_size)])
            rewards = torch.cat([rewards, torch.zeros(pad_size)])
            combined_action = torch.cat([combined_action, torch.zeros(pad_size, 2)])
            timesteps = torch.cat([timesteps, torch.zeros(pad_size)])

        return {
            "states": state,
            "actions": combined_action,
            "rewards_to_go": reward_to_go,
            "rewards": rewards,
            "timesteps": timesteps,
            "attn_mask": attn_mask
        }

def main():
    assert len(sys.argv) == 3
    training_data_path = sys.argv[1]
    checkpoint_folder = sys.argv[2]
    print("training data path:", training_data_path)
    print("checkpoint save path:", checkpoint_folder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    states, actions, rewards_to_go, timesteps = None, None, None, None
    with open(training_data_path + "state_sequence.pickle", "rb") as f:
        states = pickle.load(f).float()
        print("num_samples:", len(states))
        print("imported states.shape:", states.shape)
        print("imported states.dtype:", states.dtype)
    with open(training_data_path + "action_sequence.pickle", "rb") as f:
        actions = pickle.load(f).float()
        print("imported actions.shape:", actions.shape)
        print("imported actions.dtype:", actions.dtype)
    with open(training_data_path + "reward_sequence.pickle", "rb") as f:
        rewards_to_go = pickle.load(f).float()
        print("imported rewards_to_go.shape:", rewards_to_go.shape)
        print("imported rewards_to_go.dtype:", rewards_to_go.dtype)
    with open(training_data_path + "execution_time_sequence.pickle", "rb") as f:
        execution_times = pickle.load(f).float()
        print("imported execution_times.shape:", execution_times.shape)
        print("imported execution_times.dtype:", execution_times.dtype)
    with open(training_data_path + "timestep_sequence.pickle", "rb") as f:
        timesteps = pickle.load(f).long()
        print("imported timesteps.shape:", timesteps.shape)
        print("imported timesteps.dtype:", timesteps.dtype)


    execution_max = execution_times.max().item()
    execution_times = execution_times / execution_max
    states = states.to(device)
    actions = actions.to(device)
    rewards_to_go = rewards_to_go.to(device)
    execution_times = execution_times.to(device)
    timesteps = timesteps.to(device)

    dt_config = DecisionTransformerConfig(
        state_dim=42,
        act_dim=2,
        hidden_size=32,
        n_inner=32,
        n_layer=2,
        n_head=1,
        max_ep_len=144,
        n_positions=432,
        activation_function="relu",
    )
    print(dt_config)
    model = DecisionTransformerModel(dt_config)
    print(sum(torch.numel(p) for p in model.parameters()))
    model.to(device)
    all_data = DecisionTransformerData(states, actions, rewards_to_go, execution_times, timesteps)
    num_train_data = int(len(all_data) * 0.8)
    num_val_data = len(all_data) - num_train_data
    train_data, val_data = random_split(all_data, [num_train_data, num_val_data])

    train_data_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=256, shuffle=True)

    # rand_loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    num_epochs = 100
    warmup = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_exec_loss = 0
        total_submit_loss = 0
        # total_rand_loss = 0
        for batch in train_data_loader:
            optimizer.zero_grad()

            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rewards_to_go = batch["rewards_to_go"].to(device)
            timesteps = batch["timesteps"].squeeze(-1).to(device)
            rewards = batch["rewards"].to(device)
            attn_mask = batch["attn_mask"].to(device)


            # print("states.shape:", states.shape)
            # print("actions.shape:", actions.shape)
            # print("rewards.shape:", rewards.shape)
            # print("rewards_to_go.shape:", rewards_to_go.shape)
            # print("timesteps.shape:", timesteps.shape)
            # print("attn_mask.shape:", attn_mask.shape)
            state_pred, action_pred, return_pred = model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rewards_to_go,
                timesteps=timesteps,
                attention_mask=attn_mask,
                return_dict=False
            )
            submit_loss = nn.MSELoss()(action_pred[:, :, 0], actions[:, :, 0])
    
            # 2. Execution time loss - use MSE on second dimension
            exec_time_loss = nn.SmoothL1Loss()(action_pred[:, :, 1], actions[:, :, 1])
            loss = submit_loss + exec_time_loss if epoch > warmup else submit_loss
            # random_guesses = actions
            # random_guesses[:,-1,:] = 0.0
            # rand_loss = rand_loss_fn(random_guesses.view(-1), actions.view(-1))
            # total_rand_loss += rand_loss.item()
            loss.backward()
            optimizer.step()
            
            total_submit_loss += submit_loss.item()
            total_exec_loss += exec_time_loss.item()
            total_loss += loss.item()

        # now compute validation loss
        model.eval()
        total_val_loss = 0
        total_val_same = 0
        total_val_items = 0
        total_exec_mae = 0
        with torch.no_grad():
            for batch in val_data_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rewards_to_go = batch["rewards_to_go"].to(device)
                timesteps = batch["timesteps"].squeeze(-1).to(device)
                rewards = batch["rewards"].to(device)
                attn_mask = batch["attn_mask"].to(device)

                state_pred, action_pred, return_pred = model(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    returns_to_go=rewards_to_go,
                    timesteps=timesteps,
                    attention_mask=attn_mask,
                    return_dict=False
                )

                # round each action_pred[:, -1, :] to a decision
                val_actions_pred = action_pred[:, -1, 0]
                val_actions_pred = (val_actions_pred > 0.0).float().mul(2).add(-1)
                val_actions = actions[:, -1, 0]

                total_val_same += int((val_actions_pred == val_actions).sum().item())
                total_val_items += torch.numel(val_actions_pred)

                val_exec_pred = action_pred[:, -1, 1]
                val_exec_target = actions[:, -1, 1]
                
                # Calculate mean absolute error for execution time
                exec_mae = torch.abs(val_exec_pred - val_exec_target).sum().item() * execution_max
                total_exec_mae += exec_mae
                
                # Calculate validation losses
                val_submit_loss = nn.MSELoss()(action_pred[:, :, 0], actions[:, :, 0])
                val_exec_loss = nn.SmoothL1Loss()(action_pred[:, :, 1], actions[:, :, 1])
                val_loss = val_submit_loss + val_exec_loss if epoch > warmup else val_submit_loss
                
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch+1}, "
                f"Loss: {total_loss:.4f}, "
                f"Submit Loss: {total_submit_loss:.4f}, "
                f"Exec Loss: {total_exec_loss:.4f}, "
                f"Val Loss: {total_val_loss:.4f}, "
                f"Decision Accuracy: {100*total_val_same/total_val_items:.2f}%, "
                f"Exec Time MAE: {total_exec_mae/total_val_items:.2f}")
        if epoch % 10 == 9:
            save_path = f"{checkpoint_folder}checkpoint_{epoch+1}.cpt"
            model.save_pretrained(save_path)
            print("Saved checkpoint")
            # torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": total_loss}, f"")
            metadata_path = f"{checkpoint_folder}metadata_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "norm_factor": execution_max,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": {
                    "total": total_loss,
                    "submit": total_submit_loss,
                    "exec": total_exec_loss
                },
                "val_metrics": {
                    "loss": total_val_loss,
                    "decision_accuracy": 100*total_val_same/total_val_items,
                    "exec_mae": total_exec_mae/total_val_items
                }
            }, metadata_path)

if __name__ == "__main__":
    main()
