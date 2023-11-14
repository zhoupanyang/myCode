import copy

import numpy as np

from MEC_DDQN.env import ENV

from MEC_DDQN.DDQN import Double_DQN

import matplotlib.pyplot as plt

learning_start_step = 200
learning_fre = 5
batch_size = 64
gamma = 0.9
lr = 0.01
max_grad_norm = 0.5
tao = 0.01
memory_size = 2000
EPOCH = 350
STEP = 200


list_all = []
def train(rate,decay,greed,batch,e,incre,memory,ue=3, mec=7+1, k=11*3):
    u = ue
    m = mec
    k = k
    env = ENV(u, m, k)
    dqn = Double_DQN(env,learning_rate=rate,reward_decay=decay,e_greedy=greed,memory_size=memory,batch_size=batch,epoch=e,e_greedy_increment=incre)

    obs_shape_n = [env.n_features for i in range(env.UEs)]
    action_shape_n = [env.n_actions for i in range(env.UEs)]


    obs_size = []
    action_size = []
    game_step = 0
    episode_rew_dqn= [0.0]
    episode_time_dqn, episode_time_local, episode_time_ran, episode_time_mec = [0.0], [0.0], [0.0], [0.0]
    epoch_average_rew_dqn = []
    epoch_average_time_dqn, epoch_average_time_local, epoch_average_time_mec, epoch_average_time_ran = [], [], [], []



    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a



    for epoch in range(EPOCH):
        print("epoch:",epoch)
        obs = env.reset()
        for time_1 in range(STEP):
            action_dqn = dqn.choose_action(obs)

            o = copy.deepcopy(obs)
            obs_old = copy.deepcopy(obs)
            obs_dqn, rew_dqn, time_dqn , local_times, mec_times, ran_times= env.step(o, action_dqn, is_prob=False, is_compared=True)


            dqn.store_memory(obs_old, action_dqn, rew_dqn, obs_dqn)

            episode_rew_dqn[-1] += np.sum(rew_dqn)
            episode_time_dqn[-1] += np.sum(time_dqn)
            episode_time_local[-1] += np.sum(local_times)
            episode_time_mec[-1] += np.sum(mec_times)
            episode_time_ran[-1] += np.sum(ran_times)


            if game_step > 1000 and game_step % 100 == 0:
                dqn.learn()


            game_step += 1


        epoch_average_rew_dqn.append(- episode_rew_dqn[-1] / (env.UEs * STEP))

        epoch_average_time_dqn.append(episode_time_dqn[-1] / (env.UEs * STEP))
        print(epoch_average_time_dqn)

        epoch_average_time_local.append(episode_time_local[-1] / (env.UEs * STEP))
        epoch_average_time_mec.append(episode_time_mec[-1] / (env.UEs * STEP))
        epoch_average_time_ran.append(episode_time_ran[-1] / (env.UEs * STEP))



        episode_rew_dqn.append(0)

        episode_time_dqn.append(0)
        episode_time_local.append(0)
        episode_time_mec.append(0)
        episode_time_ran.append(0)

    return epoch_average_time_dqn



if __name__ == '__main__':

    rate = 0.053
    decay = 0.98
    greed = 0.98
    batch = 5
    e = 100
    incre = 0.001
    memory = 500

    while True:
        dqn_time = train(rate,decay,greed,batch,e,incre,memory)
        episodes_list = list(range(len(dqn_time)))
        plt.plot(episodes_list, dqn_time)
        plt.xlabel('Episodes')
        plt.ylabel('time')
        plt.title("dqn")
        plt.show()



