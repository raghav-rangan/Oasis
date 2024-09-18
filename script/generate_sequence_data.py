import pickle
import torch
import sys

def main():
    raw_data = None
    with open(sys.argv[1], "rb") as rdf:
        raw_data = pickle.load(rdf)
    features, rewards = raw_data
    rewards = rewards.unsqueeze(1)
    print("raw_features.shape:", features.shape)
    print("raw_rewards.shape:", rewards.shape)

    # make the sequences of 144 tokens (state_vector[i], action=0, reward_to_go=reward)
    # the input tensors are N x 144 x 42, N x 1
    # we want to make:
    #   1) N x 144 x 42 (basically just the input)
    #   2) N x 144 x  1 (the action space, all 0 except for last entry)
    #   3) N x 144 x  1 (the rewards to go, probably all equal to the last reward from the input)
    # we are also going to generate additional samples that do not exist in the data
    # the offline data gen has some probe parameter, call it P, and it generates sequences starting from a certain point for each of the 684 samples
    # it tests submitting at t, t+(interval/P), t+(2interval/P), ...
    # so we can derive samples for NOT submitting at t, NOT submitting at t+(interval/P), ...
    # we can take the max of the reward of the future samples because that is a feasible path for an expert that submits at the time that actually generated the reward

    # we need to know the number of probing points, in our case it was 7, so if we are not the last sequence in the probing list we can make a new sample using our matrix and the max reward over the remaining probing points

    N = features.shape[0]
    num_probing_points = 13
    num_nosubmit = N - (N // num_probing_points)
    nosubmit_sample_counter = 0
    nosubmit_state = torch.zeros(num_nosubmit, 144, 42, dtype=torch.float32)
    nosubmit_actions = torch.full((num_nosubmit, 144, 1), -1, dtype=torch.int32)
    nosubmit_rewards = torch.zeros(num_nosubmit, 144, 1, dtype=torch.float32)

    for i in range(features.shape[0]):
        if i % num_probing_points != (num_probing_points - 1):
            # we can make a pseudo-sample
            # take the max over the remaining samples in this sequence
            best_not_submit = rewards[i+1]
            for delta_i in range(1, num_probing_points - (i % num_probing_points)):
                best_not_submit = max(best_not_submit, rewards[i + delta_i])
            nosubmit_state[nosubmit_sample_counter] = features[i]
            nosubmit_rewards[nosubmit_sample_counter, :, :] = best_not_submit
            nosubmit_sample_counter += 1

    print("nosubmit actions sum:", nosubmit_actions.sum())
    print("nosubmit num nonzero rewards:", nosubmit_rewards.count_nonzero())

    # print(nosubmit_actions[0])
    # print(nosubmit_actions[0].sum())
    print("nosubmit_state.shape:", nosubmit_state.shape)
    print("nosubmit_actions.shape:", nosubmit_actions.shape)
    print("nosubmit_rewards.shape:", nosubmit_rewards.shape)
    print()

    # make the state sequence
    state_seq = features
    print("state.shape", state_seq.shape)
    state_seq = torch.cat((state_seq, nosubmit_state))
    print("combined state.shape", state_seq.shape)
    print()

    # make the action sequences
    single_act_seq = torch.full((features.shape[1], 1), -1)
    single_act_seq[-1] = 1
    act_seq = single_act_seq.unsqueeze(0).repeat(features.shape[0], 1, 1)
    # print(act_seq[0])
    print(act_seq[0].sum())
    print("actions.shape", act_seq.shape)
    act_seq = torch.cat((act_seq, nosubmit_actions))
    print("combined actions.shape", act_seq.shape)

    print("num submit final_actions:", act_seq[:, -1, :].sum())
    print()

    # make the rewards sequences
    rew_seq = torch.ones(1, 144, 1) * rewards
    print("rewards.shape", rew_seq.shape)
    rew_seq = torch.cat((rew_seq, nosubmit_rewards))
    print("combined rewards.shape", rew_seq.shape)
    print()

    times = torch.arange(144)
    time_seq = times.view(1, 144, 1).expand(state_seq.shape[0], -1, -1)
    print("timeseq.shape:", time_seq.shape)

    # save the tensors
    with open("state_sequence.pickle", "wb") as f:
        pickle.dump(state_seq, f)
    with open("action_sequence.pickle", "wb") as f:
        pickle.dump(act_seq, f)
    with open("reward_sequence.pickle", "wb") as f:
        pickle.dump(rew_seq, f)
    with open("timestep_sequence.pickle", "wb") as f:
        pickle.dump(time_seq, f)


if __name__ == "__main__":
    main()
