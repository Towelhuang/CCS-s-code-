# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:07:02 2020

@author: mb002b
"""
import teest as mp
import numpy as np
import copy
class RHFSS( object):
	def __init__(self):
		super(RHFSS, self).__init__()
		#self.action_space = ['a', 'b', 'c']
		self.n_actions = mp.order_type #可以採取的action = 訂單種類
		self.n_features = mp.n #神經網路輸入的特徵=總共要排入的訂單數


	def reset(self):  #我要寫原本的狀態
		return(np.zeros(mp.n)-1)  #讀取訂單數產生對應-1-1-1
	
	
	def step(self, observation, action):
		s = observation 
		s_ = s.copy()
		#print("=============================")

		#判斷現在要加哪一個訂單
		for n in range(len(s)):
			if s_[n] == -1:
				break
		#print("正要採取行動的訂單是第幾筆%s"%(n))
		
		#判斷採取的動作並更新s
		s_[n] = action+1

		#print("加入訂單後的狀態%s%s"%(action,s_))

		#計算目前排入的各個訂單數
		#print("加入前%s，加入後%s"%(mp.makespan(s), mp.makespan(s_)))
		if s[0] == -1:
			reward = 0
		elif s[len(s)-1] == -1 and s[len(s)-2] != -1:
			reward = 0
		elif mp.makespan(s_) == mp.makespan(s):
			reward = 100
		else:
			reward = (mp.makespan(s) - mp.makespan(s_))
		#print(mp.makespan(s), mp.makespan(s_), reward)
		#計算reward
		#print("回饋值%s"%(reward))
		
		
		
		#判斷是否結束這回合
		if n == len(s_)-1:
			done = True
		else:
			done = False
		return s_, reward, done
