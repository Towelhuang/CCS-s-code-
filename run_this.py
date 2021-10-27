from factory_env import RHFSS 
from DQN_method import DeepQNetwork
import numpy as np
import teest as mp
import csv
def run_env(episode_num):
	step = 0    # 用来控制什么时候学习
	for episode in range(episode_num):
		# initial observation 初始化環境
		observation = env.reset()
#       print('初始狀態%s')%(observation)
		while True:


			# RL choose action based on observation 根據觀測值選擇行為
			action = RL.choose_action(observation,observation)

			# RL take action and get next observation and reward 環境根據行為給出下一個state,reward, 是否終止
			observation_, reward, done = env.step(observation, action)

			#DQN儲存記憶
			RL.store_transition(observation, action, reward, observation_)
			#控制學習開始時間和頻率(先累積一些記憶再開始學習)
			if (step > 300) and (step % 25 == 0):
				RL.learn()

			# swap observation 將下一個state_變為下次循環的state
			observation = observation_

			# break while loop when end of this episode 如果終止，就跳出循環
			if done:
				#print("排完終止%s"%(done))
				break
			step += 1 #總步數

	# end of game
	print('game over')
	print(mp.makespan(observation)) 
	print(observation)
	return observation_

#当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
#当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
#防止在被其他文件导入时显示多余的程序主体部分。
if __name__ == "__main__": 
	# maze game
	env = RHFSS()
	RL = DeepQNetwork(env.n_actions,  env.n_features,
						learning_rate=0.005,
						reward_decay=0.9,
						e_greedy=0.95,
						replace_target_iter=3,  #每200步替換一次target_net的參數
						memory_size=20,         #記憶體上限
						# output_graph=True       #是否輸出tensorboard文件
						)
	#run_env()
	#跑100次輸出結果訂單狀態和c_max
	result = []
	for i in np.arange(5,300,10):
		makespan = mp.makespan(run_env(i))
		result.append(makespan)
	with open('small_output_DQN_0.005_learning rate.csv', newline='', mode = 'a', encoding='utf-8') as f:
		wri = csv.writer(f, delimiter = ',')
		wri.writerow('learning rate 0.05')
		wri.writerow(result)
#    env.after(100, run_maze)
#    env.mainloop()
	RL.plot_cost()      #查看神經網路的誤差曲線
    
    