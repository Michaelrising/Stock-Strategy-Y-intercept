import numpy as np
from PPO import *
from EnvStock import StockStrategy
import torch
import time
# from params import configs
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse
# device = torch.device(configs.device)



################################## set device ##################################

def set_device(cuda=None):
    print("============================================================================================")

    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available() and cuda is not None:
        device = torch.device('cuda:' + str(cuda))
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")
    return device


################################### Training ###################################

def train(summary_dir, pars, configs):
    ################## set device ##################
    device = 'cuda:1' #set_device() if configs.cuda_cpu == "cpu" else set_device(configs.cuda)

    ####### initialize environment hyperparameters ######

    num_env = configs.num_envs
    max_updates = configs.max_updates
    eval_interval = 50
    max_ep_len = 64  # max timesteps in one episode

    print_freq = 2  # print avg reward in the interval (in num updating steps)
    log_freq = 2  # log avg reward in the interval (in num updating steps)
    explore_eps = 0.8

    ####################################################
    ################ PPO hyperparameters ################
    batch_size = 14
    decay_step_size = 1000
    decay_ratio = 0.8
    grad_clamp = 0.2
    update_timestep = 2  # update policy every n epoches
    K_epochs = 2  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0001*3  # learning rate for actor network
    lr_critic = 0.00005*3  # learning rate for critic network

    ########################### Env Parameters ##########################
    tickers, model_path, money, r, pred_T = pars
    envs = [StockStrategy(tickers=tickers, model_path=model_path, money=money, r=r, pred_T=pred_T) for _ in range(configs.num_envs)]

    test_env = StockStrategy(tickers=tickers, model_path=model_path, money=money, r=r, pred_T=pred_T) # gym.make(configs.env_id, patient=patient).unwrapped

    # state space dimension
    state_dim = len(tickers)

    # action space dimension
    action_dim = envs[0].action_space.n
    env_name = 'StockStrategy'
    print("training environment name : " + env_name)
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    t = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = "../Log/summary/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # summary_dir = log_dir + '/' + str(t) + "-num_env-" + str(num_env)
    writer = SummaryWriter(log_dir=summary_dir)

    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")
    print("num of envs : " + str(num_env))
    print("max training updating times : ", max_updates)
    print("max timesteps per episode : ", max_ep_len)

    # print("model saving frequency : " + str(save_model_freq) + " episodes")
    print("log frequency : " + str(log_freq) + " episodes")
    print("printing average reward over episodes in last : " + str(print_freq) + " episodes")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")
    print("The initial explore rate : " + str(explore_eps) + " and initial exploit rate is : 1- " + str(explore_eps))

    print("PPO update frequency : " + str(update_timestep) + " episodes")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")
    if configs.decayflag:
        print("decaying optimizer with step size : ", decay_step_size, " decay ratio : ", decay_ratio)
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        num_env,
        device,
        decay_step_size,
        decay_ratio)

    # ppo_agent.load(
    #     "PPO_pretrained/analysis/patient025_20220121-1603-m1-0.5-AI-0.8_PPO_gym_cancerCancerControl-v0_0_0.pth")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    ep_rewards = [[] for _ in range(num_env)]
    ep_dones = [[] for _ in range(num_env)]
    num_episods = [0 for _ in range(num_env)]
    flag_step = [0 for _ in range(num_env)]
    # training loop
    reward_record = -1000000
    for i_update in range(max_updates):
        for i, env in enumerate(envs):
            i_step = 1
            num_episods[i] = 0
            ep_rewards[i] = []
            ep_dones[i] = []
            flag_step[i] = 0
            while i_step < batch_size:
                num_episods[i] += 1
                eps = max(- max(i_update - 10000, 0) * (explore_eps - 0.5) / 10000 + explore_eps, 0.5)
                determine = np.random.choice(2, p=[1 - eps, eps])  # explore epsilon
                fea= env.reset()
                while True:
                    # select action with policy, with torch.no_grad()

                    state_tensor, action, action_logprob = ppo_agent.select_action(fea) \
                        if determine else ppo_agent.greedy_select_action(fea)  # state_tensor is the tensor of current state
                    ppo_agent.buffers[i].states.append(state_tensor)
                    ppo_agent.buffers[i].actions.append(action)
                    ppo_agent.buffers[i].logprobs.append(action_logprob)
                    # action = actlist[i_step-1]
                    # print(i_step-1)
                    fea, reward, done = env.step(action)
                    # print("Reward: {} \t Time: {}".format(reward, time))

                    # saving reward and is_terminals
                    ppo_agent.buffers[i].rewards.append(reward)
                    ppo_agent.buffers[i].is_terminals.append(done)

                    # print(action)
                    ep_rewards[i].append(reward)
                    ep_dones[i].append(done)
                    i_step += 1
                    # break; if the episode is over
                    if done:
                        flag_step[i] = i_step
                        break

        mean_rewards_all_env = sum([sum(ep_rewards[i][: flag_step[i]])/num_episods[i] for i in range(num_env)]) / num_env

        # update PPO agent
        if i_update % update_timestep == 0:
            loss = ppo_agent.update(decayflag=configs.decayflag, grad_clamp=grad_clamp)

            # log in logging file
            # if i_update % log_freq == 0:
            print("steps:{} \t\t rewards:{}".format(i_update, np.round(mean_rewards_all_env, 3)))
            writer.add_scalar('VLoss', loss, i_update)
        writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)

        # if i_update % eval_interval == 0:
        #     # rewards, survivalMonth, actions, states, colors = evaluate(test_env, ppo_agent.policy_old, eval_times)
        #     g_rewards, rewardSeq, ActionSeq, ActSeq, ModeSeq, TimeSeq = greedy_evaluate(test_env, ppo_agent.policy_old)
        #     # writer.add_scalar("Reward/evaluate", rewards, i_update)
        #     writer.add_scalar("Reward/greedy_evaluate", g_rewards, i_update)
        #     torch.save(ppo_agent.policy_old.state_dict(),summary_dir + '/PPO-ProgramEnv-converge-model-{}.pth'.format(configs.filepath[2:-3]))
        #     print(" Test Total reward:{} \n Test rewards List:{} \n Test Actions:{} \n Test Acts:{} \n Test Modes: {} \n Test Times:{}".format(np.round(g_rewards, 3),rewardSeq, ActionSeq, ActSeq,ModeSeq, TimeSeq))
        # #
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    crt_time = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join("log", 'summary', str(crt_time))
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    total1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default= 2)
    parser.add_argument('--max_updates', type=int, default=10**5)
    parser.add_argument('--decayflag', type=str, default=True)
    parser.add_argument('--tickers', type=list, default=['1332 JT', '1333 JT', '1605 JT'])
    parser.add_argument('--model_path', type=str, default='../Log/20221118-0641/models')
    parser.add_argument('--money', type=int, default=10**5)
    parser.add_argument('--pred_T', type=int, default=7)
    parser.add_argument('--interest_rate', type=float, default=0.01)


    configs=parser.parse_args()

    pars = (configs.tickers, configs.model_path, configs.money, configs.pred_T, configs.interest_rate)
    train(summary_dir, pars, configs)








