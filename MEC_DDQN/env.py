import copy
import math

import numpy as np

class ENV():
    def __init__(self, UEs, MECs, k):
        self.UEs = UEs
        self.MECs = MECs
        self.k = k

        q = np.full((k, 1), 0.)
        p = np.linspace(0, 1, k).reshape((k, 1))

        for i in range(MECs - 1):
            a = np.full((k, 1), float(i + 1))
            b = np.linspace(0, 1, k).reshape((k, 1))
            q = np.append(q, a, axis=0)
            p = np.append(p, b, axis=0)

        self.actions = np.hstack((q, p))
        self.n_actions = len(self.actions)
        self.n_features = 3 + MECs * 4
        self.discount = 0.01


        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz
        self.nor = 10**(-3)

        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB
        self.GB = 1024 * self.MB


        self.UE_f = np.random.randint(1.5 * self.GHz * self.nor, 2 * self.GHz * self.nor)
        self.MEC_f = np.random.randint(6 * self.GHz * self.nor, 8 * self.GHz * self.nor)

        self.r = 40 * math.log2(1 + (16 * 10)) * self.mHz * self.nor
        self.R = 200
        self.L = 10


        self.et, self.lt = 1, 1
        self.local_core_max, self.local_core_min = 1.3 * self.UE_f, 0.7 * self.UE_f
        self.server_core_max, self.server_core_min = 1.3 * self.MEC_f, 0.7 * self.MEC_f
        self.uplink_max, self.uplink_min = 1.3 * self.r, 0.7 * self.r
        self.downlink_max, self.downlink_min = 1.3 * self.r, 0.7 * self.r
        self.uplink_max_cloud, self.uplink_min_cloud = 1.5 * self.r, 0.9 * self.r
        self.downlink_max_cloud,self.downlink_min_cloud = 1.5 * self.r,0.9*self.r
        self.cloud_max,self.cloud_min = 3.2 * self.MEC_f,2.3 * self.MEC_f
        self.e = 1


    def reset(self):
        obs = []
        servers_cap = []
        new_cap = True
        for i in range(self.UEs):
            uplink, downlink = [], []
            leave_time = []

            task_size = np.random.randint(10 * self.MB, 12 * self.MB)

            task_cpu_cycle = np.random.randint(20, 22)
            local_comp = np.random.randint(0.9 * self.UE_f, 1.1 * self.UE_f)

            vi = np.random.randint(80, 120)
            xi = np.random.randint(-1000, 1000)

            di_list = [-1, 1]
            di_index = np.random.randint(0, 2)
            di = di_list[di_index]
            yi = 0
            if di > 0:
                yi = np.random.uniform(-self.L + 0.1, 0)
            else:
                yi = np.random.uniform(0.1, self.L)

            vehicle = np.array([xi,yi,di,vi])
            for j in range(self.MECs):
                if j == 0:
                    up = np.random.randint(1.1 * self.r, 1.3 * self.r)
                    down = np.random.randint(1.1 * self.r, 1.3 * self.r)
                    xj,yj = 0,0
                    if new_cap:
                        cap = np.random.randint(2.5 * self.MEC_f, 3 * self.MEC_f)
                        servers_cap.append(cap)
                else:
                    up = np.random.randint(0.9 * self.r, 1.1 * self.r)
                    down = np.random.randint(0.9 * self.r, 1.1 * self.r)
                    yj_list = [-self.L / 2, self.L / 2]
                    yj_index = np.random.randint(0, 2)
                    yj = yj_list[yj_index]
                    while True:
                        xj = np.random.randint(xi - self.R, xi + self.R)
                        if math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)) < self.R:
                            break
                    if new_cap:
                        cap = np.random.randint(0.9 * self.MEC_f, 1.1 * self.MEC_f)
                        servers_cap.append(cap)


                uplink.append(up)
                downlink.append(down)
                leave = (di*(xj - xi) + math.sqrt(math.pow(self.R,2) - math.pow((yi-yj),2))) / vi * 1000
                if j == 0:
                    leave = 100000
                leave_time.append(leave)
            observation = np.array([task_size, task_cpu_cycle, local_comp])
            observation = np.hstack((observation, servers_cap, uplink, downlink,leave_time))
            obs.append(observation)
            new_cap = False
        return obs

    def choose_action(self, prob):

        action_choice = np.linspace(0, 1, self.k)
        actions = []
        for i in range(self.UEs):
            a = np.random.choice(a=(self.MECs * self.k), p=prob[i])
            target_server = int(a / self.k)
            percen = action_choice[a % self.k]
            action = [target_server, percen]
            actions.append(action)
        return actions

    def step(self, observation, actions_prob, is_prob=True, is_compared=True):
        if is_prob:
            actions = self.choose_action(actions_prob)
        else: actions = actions_prob
        new_cap = False
        obs_ = []
        rew = []
        dqn_times, local_times, ran_times, mec_times = [], [], [], []

        for i in range(self.UEs):
            if i == self.UEs - 1: new_cap = True

            task_size, task_cpu_cycle, local_comp, servers_cap, uplink, downlink,leave_time = \
                observation[i][0], observation[i][1], observation[i][2], observation[i][3:3+self.MECs], observation[i][3+self.MECs:3+self.MECs*2], observation[i][3+self.MECs*2:3+self.MECs*3],observation[i][3+self.MECs*3:3+self.MECs*4]


            action = actions[i]
            target_server, percen = int(action[0]), action[1]


            tr_time = (percen * task_size) / uplink[target_server] + self.discount * ( percen * task_size) / downlink[target_server]
            comp_local_time = int(task_size) * int(task_cpu_cycle) * (1 - percen) / (local_comp)
            comp_mec_time = (percen * task_size * task_cpu_cycle) / servers_cap[target_server]

            time_cost = max(comp_local_time,comp_mec_time) + tr_time


            if target_server == 0:
                wait_time = np.random.randint(35,100)
                time_cost += wait_time

            reward = -time_cost





            if time_cost > leave_time[target_server]:
                reward -= 10000



            local_only_time = task_size * task_cpu_cycle/(local_comp) * self.et


            mec_only_tr_time = task_size / uplink[target_server] + self.discount * task_size / downlink[target_server]
            mec_only_comp_time = task_size * task_cpu_cycle / servers_cap[target_server]

            mec_only_time_cost = (mec_only_tr_time + mec_only_comp_time) * self.et
            mec_only_wait_time = np.random.randint(150, 160)
            mec_only_time_cost += mec_only_wait_time

            if target_server == 0:
                wait_time = np.random.randint(35,100)
                mec_only_time_cost += wait_time


            percen_ran = np.random.uniform()
            mec_ran = np.random.randint(self.MECs)

            random_tr_time = (percen_ran * task_size) / uplink[mec_ran] + (self.discount * percen_ran * task_size) / downlink[mec_ran]
            random_comp_local_time = (1 - percen_ran) * task_size * task_cpu_cycle / local_comp
            random_comp_mec_time = percen_ran * task_size * task_cpu_cycle / servers_cap[mec_ran]

            random_comp_time = max(random_comp_local_time, random_comp_mec_time)
            random_time_cost = (random_comp_time + random_tr_time) * self.et
            random_wait_time = np.random.randint(105, 115)
            random_time_cost += random_wait_time
            if target_server == 0:
                wait_time = np.random.randint(35,100)
                random_time_cost += wait_time



            x = np.random.uniform()
            y = 0.5
            if (x > y):
                local_comp = min(local_comp + np.random.randint(0, 0.2 * self.UE_f), self.local_core_max)
                for j in range(self.MECs):
                    if j == 0:
                        cap = min(servers_cap[j] + np.random.randint(0, 0.3 * self.UE_f),self.cloud_max)
                        if new_cap:
                            for x in range(self.UEs):
                                observation[x][2 + j] = cap
                        downlink[j] = min(downlink[j] + np.random.randint(0, 0.2 * self.r), self.downlink_max_cloud)
                        uplink[j] = min(uplink[j] + np.random.randint(0, 0.2 * self.r), self.uplink_max_cloud)
                    else:
                        cap = min(servers_cap[j] + np.random.randint(0, 0.3 * self.UE_f), self.server_core_max)

                        if new_cap:
                            for x in range(self.UEs):
                                observation[x][2 + j] = cap
                        downlink[j] = min(downlink[j] + np.random.randint(0, 0.2 * self.r), self.downlink_max)
                        uplink[j] = min(uplink[j] + np.random.randint(0, 0.2 * self.r), self.uplink_max)

            else:
                local_comp = max(local_comp + np.random.randint(-0.2 * self.UE_f, 0), self.local_core_min)
                for j in range(self.MECs):
                    if j == 0:
                        cap = max(servers_cap[j] - np.random.randint(0, 0.3 * self.UE_f),self.cloud_min)
                        if new_cap:
                            for x in range(self.UEs):
                                observation[x][2 + j] = cap
                        downlink[j] = max(downlink[j] - np.random.randint(0, 0.2 * self.r), self.downlink_min_cloud)
                        uplink[j] = max(uplink[j] - np.random.randint(0, 0.2 * self.r), self.uplink_min_cloud)

                    else:
                        if new_cap:
                            cap = max(servers_cap[j] + np.random.randint(0, 0.3 * self.UE_f), self.server_core_max)
                            for x in range(self.UEs):
                                observation[x][2 + j] = cap
                        downlink[j] = max(downlink[j] - np.random.randint(0, 0.2 * self.r), self.downlink_min)
                        uplink[j] = max(uplink[j] - np.random.randint(0, 0.2 * self.r), self.uplink_min)

            task_size = np.random.randint(10 * self.MB, 12 * self.MB)
            task_cpu_cycle = np.random.randint(20,22)

            vi = np.random.randint(80, 120)
            xi = np.random.randint(-1000, 1000)

            di_list = [-1, 1]
            di_index = np.random.randint(0, 2)
            di = di_list[di_index]
            yi = 0
            if di > 0:
                yi = np.random.uniform(-self.L + 0.1, 0)
            else:
                yi = np.random.uniform(0.1, self.L)


            for ii in range(self.MECs):
                if ii == 0:
                    xj,yj = 0,0
                else:
                    if new_cap:
                        yj_list = [-self.L / 2, self.L / 2]
                        yj_index = np.random.randint(0, 2)
                        yj = yj_list[yj_index]
                        while True:
                            xj = np.random.randint(xi - self.R, xi + self.R)
                            if math.sqrt(math.pow(xi - xj, 2) + math.pow(yi - yj, 2)) < self.R:
                                break
                leave = (di * (xj - xi) + math.sqrt(math.pow(self.R, 2) - math.pow((yi - yj), 2))) / vi * 1000
                if ii == 0:
                    leave = 100000
                leave_time[ii] = leave

            observation_ = np.array([task_size, task_cpu_cycle, local_comp])
            observation_ = np.hstack((observation_, servers_cap, uplink, downlink,leave_time))
            obs_.append(observation_)

            rew.append(reward)

            dqn_times.append(time_cost)
            local_times.append(local_only_time)
            mec_times.append(mec_only_time_cost)
            ran_times.append(random_time_cost)



        if is_compared:
            return obs_, rew, dqn_times, local_times, mec_times, ran_times
        else:
            return obs_, rew, dqn_times