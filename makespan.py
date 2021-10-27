# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:32:32 2019

@author: mb002b
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#/************************************************************************/   
#/*        參數設定(與問題相關)                                          */   
#/************************************************************************/  

#// 藉由設定n, N, W 來讀取問題檔案

data = pd.read_csv(r"C:\Users\MB002B\Desktop\109張詠勝\程式\data\n20_p6_w24_525_small.csv")

U = np.array(data['工作站順序']) #U[i] 階段i在哪個工作站
n = 8			#工單數量??????????????????????????????????????????????????????????????????????????????????????????
N = U.shape[0]			#製程階段數量
W = np.max(U)+1     		# 工作站數量
order_type = 6  #訂單種類數 可以採取的action?????????????????????????????????????????????????????????????????????????都一樣
order_type_num = [2,2,1,2,1,0]#?????????????????????????????????????????????????????????????????????????
#R = int(1 * W) 	# 被選出的stocker的個數
B_Max = 1			#緩存區容量 1
STK_Max = 1 		#自動倉儲容量 1
t = 10				#相鄰兩工作站之間傳送時間 5
max_M = 2;	#一個工作站中最大平行機台數目 = max{M[.]}
openedSTKID = np.array([0,2])
#openedSTKID = np.array([1,3,5,7,9])
#openedSTKID = np.array([1,2,3,4,5,6,7,8,9,11,13,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57])
#openedSTKID = np.array([1,3,5,7,9,11,13,15,17,19,21,23])               #手動輸入要開啟哪個工作站的STK?????????????????????????????????????????????????????????????????????????





def find_nearest_above(my_array, target): #獲得numpy陣列中最接近的值的索引
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return my_array[masked_diff.argmin()]



#產生要跑的資料 假設輸入進的資料長[1,2,3,1,1,1]  n=6(工單數會變) 
product1  = np.array(data['工單1'])
product2  = np.array(data['工單2'])
product3  = np.array(data['工單3'])
product4  = np.array(data['工單4'])
product5  = np.array(data['工單5'])
product6  = np.array(data['工單6'])

noproduct = np.zeros(N)
#state = np.random.randint(1,4, size = n, dtype = int)



#/************************************************************************/   
#/*        適應值函式                                                    */   
#/************************************************************************/ 
def makespan(list):
	state = list
	#計算n實際排入的訂單數到底多少 例如:[1,1,2,3,0,0,0,0,0]只有4筆
	for n_ in range(len(state)): #沒有排任何工單
		if state[n_] == -1:
			break
	#n_是排入的工單個數
	def producep(list,n_):  #????????????????????????????????????????????????????????????????????????????????????????????????可以再改進
		schedule = np.zeros([n_,N])
		for i in range(n_):
			if (list[i] == 1):
				schedule[i] = product1
			elif(list[i] == 2):
				schedule[i] = product2
			elif(list[i] == 3):
				schedule[i] = product3
			elif(list[i] == 4):
				schedule[i] = product4
			elif(list[i] == 5):
				schedule[i] = product5
			elif(list[i] == 6):
				schedule[i] = product6
			else:
				schedule[i] = noproduct
		schedule = np.transpose(schedule) #因為Numpy的行列順序不同
		
		return schedule
	p = producep(state,n_)
	

	
	#// 下方參數從問題檔案中讀取
	
	#p = np.zeros([N,n], dtype = int)	#p[i][j] = 在第i個階段第j個工單的加工時間 (索引值[][]跟檔案中的矩陣相反) excel中N=column加工階段  n=row
	#U = np.zeros(N , dtype = int) 	#U[i] --> 階段i在U[i]工作站加工  //資料下面的一個橫條 長度=製成階段 0123401234012340
	M = np.ones(W, dtype = int)*max_M		#M[i] --> 第i個工作站裡有M[i]個平行機台數 //資料最下面一條 6666固定值每個都依樣 6長度等於幾個加工站 
	T = np.zeros([W,W],  dtype = int)	#T[i][j] --> 工作站i到工作站j的傳送時間  主程式會計算T
	useTimes_sm = np.zeros([W,max_M])    #[s][m] = 機台m 在工作站s 使用次數 (大於1次表 目前工單 前已有工單在加工)
	C_sm = np.zeros([W,max_M])           #C[s][m] = 工作站s 在 機台 m 加工後的完工時間 (用於找此工單前一個工單的完工時間)
	past_Csm = np.zeros([W,max_M])
	#計算T[][]
	for i in range(W):
		j=i+1
		for j in range (W):
			tmp_diff = j - i
			if ( W - j + i < j - i ):
				tmp_diff = W - j + i
			T[i][j] = t * tmp_diff *(-1)
			T[j][i] = t * tmp_diff *(-1)
	#// 變數
	
	#int alpha[W];			#alpha[i] = 工作站i中最早可以開始使用的平行機台的時間
	s = np.zeros([N,n_], dtype = int)			#s[i][j] = 階段i第j個工單的開始加工時間
	c = np.zeros([N,n_], dtype = int)			#c[i][j] = 階段i第j個工單的完工時間
	machine = np.zeros([N,n_], dtype = int)		#m[i][j] = 階段i第j個工單的加工機台編號 (宜萱沒考慮到的)
	tau =  np.zeros([W,max_M], dtype = int)#tau[i][j] = 第i個工作站的第j個機台的available time
	#beta = np.zeros(W, dtype = int)		#beta[i] = 機台i的buffer可以用的最早時間
	#delta = np.zeros(R, dtype = int)		#delta[i] = 第 i 個 stocker 最早可提供的時間
	#程式內

	
	
	pi = np.array(range(n_), dtype = int)
	
	#STK開啟數量有關的
	openedSTK = np.zeros(W, dtype = int) #openedSTK[i] == 1 表示工作站i有STK
	STK_Vol = np.ones(W, dtype = int)*STK_Max #有開的stocker容量只能裝到1個
	#根據openedSTKID來產生openedSTK
	openedSTK[openedSTKID] = 1 #指示哪些工作站有stocker
	STK_Vol[openedSTKID] = 0 #stocker清空內部暫存
	#跟STOKER和warehouse有關的時間資料
	#STK_Vol = np.array([STK_Max,0,STK_Max,0]) #目前STK已有工單數，沒有STK的視為已滿
	InSTK = np.array([[0],[0],[0]])
	OutSTK = np.array([[0],[0],[0]])
	#InSTK = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0,0]])
	#OutSTK = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0,0]])
	#InSTK = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0,0],[0,0]], dtype = object)
	#OutSTK = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0,0],[0,0]], dtype = object)
	#InSTK = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0,0]], dtype = object)#??????????????????????????????????????????
	#OutSTK = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0,0]], dtype = object)#??????????????????????????????????????????
	InWH = [0]
	OutWH = [0]
	
	WHtime = 60 #進出倉庫時間統一固定
	
	
	
	#/開始計算makespan************************************************************************/   
	
	#計算 stage 0 的開始時間與完工時間           
	for h in range(n_):        #tau[i][j] = 第i個工作站的第j個機台的available time
		i=0
		min_tau = tau[U[0]][0]     # // 假設tau[U[0]][.]最小值為tau[U[0]][0]  //U[0]第一個加工階段的工作站
		min_tau_index = 0          #// 存下此最小值的索引值為0
		for j in range(M[U[0]]):   #// 考慮所有tau[U[0[][.]以求得min_tau和min_tau_index  //// M[i] --> 第i個工作站裡有M[i]個平行機台
			if (tau[U[0]][j] < min_tau):   #//第一個工作站的第j個機台的可用時間小於最小可用時間
				min_tau = tau[U[0]][j]   #//目前最快的可用的時間就是第一階段的第j個機台的可用時間
				min_tau_index = j			 #//目前最快可用的就是第j個機台
	
	
		#// 設定加工此工單的機台編號 (宜萱沒考慮到) // m[i][j] = 階段i第j個工單的加工機台編號 int pi[n];	// 工單排程的順序
		machine[0][pi[h]] = min_tau_index
		#//為 工作站 機台m 第一個開始加工
		if(useTimes_sm[U[i]][machine[0][pi[h]]] == 0 ):     #//// [s][m] = 機台m 在工作站s 使用次數 0表使沒有工單在加工
			s[0][pi[h]] = 0							#// alpha[U[0]] = min_tau
			c[0][pi[h]] = s[0][pi[h]] + p[0][pi[h]]
			tau[U[0]][min_tau_index] = c[0][pi[h]]		#// (宜萱沒考慮到)
			past_Csm[U[0]][machine[0][pi[h]]] = C_sm[U[0]][machine[0][pi[h]]]
			C_sm[U[0]][machine[0][pi[h]]] = c[0][pi[h]]
			useTimes_sm[U[i]][machine[0][pi[h]]] +=1
		else:        #//如果// [s][m] = 機台m 在工作站s 使用次數不等於0表示在加工
			s[0][pi[h]] = min_tau;					#// alpha[U[0]] = min_tau
			c[0][pi[h]] = s[0][pi[h]] + p[0][pi[h]]
			tau[U[0]][min_tau_index] = c[0][pi[h]]	#// (宜萱沒考慮到)
			past_Csm[U[0]][machine[0][pi[h]]] = C_sm[U[0]][machine[0][pi[h]]]
			C_sm[U[0]][machine[0][pi[h]]] = c[0][pi[h]]
			useTimes_sm[U[i]][machine[0][pi[h]]] += 1  #//工作站U[i]的機台machine[0][pi[h]] 使用次數++
	
	
		#// 計算 stages 1, 2, ..., N-1 的開始時間與完工時間   int pi[n];	// 工單排程的順序
	for i in range(1,N):
		#根據前一階段的完工時間C得到現階段的pi工單順序
		index = c[i-1].argsort()
		pi = index #新這階段的工單順序
		#計算開始時間與完工時間
		for h in range(n_):
			min_tau = tau[U[i]][0]                  #// 假設tau[U[i]][.]最小值為tau[U[i]][0]
			min_tau_index = 0                       #// 存下此最小值的索引值為0
			for j in range(max_M):                #// 考慮所有tau[U[i]][.]以求得min_tau和min_tau_index
				if (tau[U[i]][j] < min_tau):        #//看哪個機台的最快可使用
					min_tau = tau[U[i]][j]
					min_tau_index = j
			machine[i][pi[h]] = min_tau_index
		#//判斷是否需暫存 
		#//不須暫存
			if ( c[i-1][pi[h]] >= min_tau ):  #//完工時間大於下個機台的可用時間
				s[i][pi[h]] = c[i-1][pi[h]]  #//把第I工作站的第pi[h]工單的開始加工時間 = 第i-1工作站的第pi[h]工單的開始加工時間
				past_Csm[U[i]][machine[i][pi[h]]] = C_sm[U[i]][machine[i][pi[h]]]
				c[i][pi[h]] = s[i][pi[h]] + p[i][pi[h]] #//工單j在階段i的完工時間= 開始+加工時間
				tau[U[i]][min_tau_index] = c[i][pi[h]] #// (宜萱沒考慮到)
				C_sm[U[i]][machine[i][pi[h]]] = c[i][pi[h]] #// C[s][m] = 工作站s 在 機台 m 加工後的完工時間 (用於找此工單前一個工單的完工時間)
				useTimes_sm[U[i]][machine[i][pi[h]]]+=1 #//// useTimes_sm[s][m] = 機台m 在工作站s 使用次數 (大於1次表 目前工單 前已有工單在加工)
		#需要暫存
			elif (c[i-1][pi[h]]>= past_Csm[U[i]][machine[i][pi[h]]]):  #進入Buffer
				s[i][pi[h]] = min_tau          #//若需要站存 把第I工作站的第pi[h]工單的開始加工時間當作目前最小可用時間
				past_Csm[U[i]][machine[i][pi[h]]] = C_sm[U[i]][machine[i][pi[h]]]
				c[i][pi[h]] = s[i][pi[h]] + p[i][pi[h]] #//工單j在階段i的完工時間= 開始+加工時間
				tau[U[i]][min_tau_index] = c[i][pi[h]] #// (宜萱沒考慮到)
				C_sm[U[i]][machine[i][pi[h]]] = c[i][pi[h]] #// C[s][m] = 工作站s 在 機台 m 加工後的完工時間 (用於找此工單前一個工單的完工時間)
				useTimes_sm[U[i]][machine[i][pi[h]]]+=1 #//// useTimes_sm[s][m] = 機台m 在工作站s 使用次數 (大於1次表 目前工單 前已有工單在加工)
			else: #進入STK
				#==========================  #這我看不董sunny
				#計算目前STK_Vol中每個STK的數量
				for m in openedSTKID: 
					AtSTK = c[i-1][pi[h]] + T[U[i-1]][m]  #因為到不同STK的時間不同
					Vol_tmp = 0
					for o in range(len(InSTK[m])):
						if (OutSTK[m][o] > AtSTK | AtSTK > InSTK[m][o]):
							Vol_tmp += 1
					STK_Vol[m] = Vol_tmp
				#print(STK_Vol)
				#===============================================
				#根據STK_Vol判斷是否全部滿載 若滿載 Overload = True
				Overload = True
				for Vol in  STK_Vol:
					if(Vol <2):
						Overload = False
				#print("滿載%s"%(Overload))
				#===============================================
				#沒有滿載
				if(Overload == False):                             #這我看不董sunny
					#看進入哪一個STK #U[i]是工作站 #openedSTK是哪個工作站有開
					STK_tmp= []
					for k in openedSTKID:
						if (STK_Vol[k] <2):
							STK_tmp.append(k)
					#找去最近且有位置的STK
					STK_avi =np.array(STK_tmp)
					STK_avi = np.append(STK_avi, STK_avi+W)
					STK = STK_avi[np.argmin(abs(STK_avi-U[i]))] #STK=這個工單要去的STK  #這可以算最短距離的stocker
					if (STK > W-1):
						STK = STK-W
					#print('J%s-%s-STK%s-時間%s'%(pi[h],i,STK,T[U[i-1]][STK]*2))
					InSTK[STK]=np.append(c[i-1][pi[h]]+T[U[i-1]][STK])
					BackST = c[i-1][pi[h]] + T[U[i-1]][STK]+T[U[i]][STK] #BackST是計算假設馬上被傳回去到站的時間
					if(BackST < past_Csm[U[i]][machine[i][pi[h]]]):  #會停留在STK不能馬上被傳送回工作站
						s[i][pi[h]] = C_sm[U[i]][machine[i][pi[h]]]
						past_Csm[U[i]][machine[i][pi[h]]] = C_sm[U[i]][machine[i][pi[h]]]
						OutSTK[STK].append(s[i][pi[h]]-T[U[i]][STK]) #計算出去stocker的時間就是工單的開始時間
						#print(BackST,s[i][pi[h]])
						#print("停留STK")
					else: #判斷進入STK立刻馬上傳回工作站時是進入機台還是BUFFER
						if (BackST < C_sm[U[i]][machine[i][pi[h]]]): #進Buffer
							s[i][pi[h]] = C_sm[U[i]][machine[i][pi[h]]]
							past_Csm[U[i]][machine[i][pi[h]]] = min_tau
							OutSTK[STK].append(c[i-1][pi[h]] + T[U[i-1]][STK])
							#print(BackST,s[i][pi[h]])
							#print("進buffer")
						else:
							s[i][pi[h]] = BackST
							past_Csm[U[i]][machine[i][pi[h]]] = min_tau
							OutSTK[STK].append(c[i-1][pi[h]] + T[U[i-1]][STK])
							#print(BackST,s[i][pi[h]])
							#print("進機台")
				else:#STK滿載判斷去倉庫====================5/11
					#print("滿載!!!!!!!!!!!!")
					InWH.append(c[i-1][pi[h]] + WHtime)
					BackST = c[i-1][pi[h]] + 2*WHtime
					if(BackST < past_Csm[U[i]][machine[i][pi[h]]]):  #會停留在STK不能馬上被傳送回工作站
						s[i][pi[h]] = C_sm[U[i]][machine[i][pi[h]]]
						past_Csm[U[i]][machine[i][pi[h]]] = C_sm[U[i]][machine[i][pi[h]]]
						OutWH.append(s[i][pi[h]]-WHtime)
						#print(BackST,s[i][pi[h]])
						#print("停留Warehouse")
					else: #判斷進入STK立刻馬上傳回工作站時是進入機台還是BUFFER
						if (BackST < C_sm[U[i]][machine[i][pi[h]]]): #進Buffer
							s[i][pi[h]] = C_sm[U[i]][machine[i][pi[h]]]
							past_Csm[U[i]][machine[i][pi[h]]] = min_tau
							OutWH.append(s[i][pi[h]]-WHtime)
							#print(BackST,s[i][pi[h]])
							#print("傳到WH後進buffer")
						else:
							s[i][pi[h]] = BackST
							past_Csm[U[i]][machine[i][pi[h]]] = min_tau
							OutWH.append(s[i][pi[h]]-WHtime)
							#print(BackST,s[i][pi[h]])
							#print("傳到WH後進機台")
				
				c[i][pi[h]] = s[i][pi[h]] + p[i][pi[h]] #//工單j在階段i的完工時間= 開始+加工時間
				tau[U[i]][min_tau_index] = c[i][pi[h]] #// (宜萱沒考慮到)
				C_sm[U[i]][machine[i][pi[h]]] = c[i][pi[h]] #// C[s][m] = 工作站s 在 機台 m 加工後的完工時間 (用於找此工單前一個工單的完工時間)
				useTimes_sm[U[i]][machine[i][pi[h]]]+=1 #//// useTimes_sm[s][m] = 機台m 在工作站s 使用次數 (大於1次表 目前工單 前已有工單在加工)
	
	
	if n_ == 0:
		c_max = 0
	else:
		c_max = c[N-1][0]
		for i in range(n_):
			if (c[N-1, i] > c_max):
				c_max = c[N-1][i] 
	
	
	
	# =============================================================================
	# Ctrl+1:注释/撤销注释
	# Ctrl+4/5:块注释/撤销块注释
	# Ctrl+L:跳转到行号
	# F5:运行
	# F11:全屏
	# =============================================================================
	#%%
	s.dtype = 'int'
	c.dtype = 'int'
	
	
	ylabel = W*max_M+1
	
	plt.figure(figsize=(40, 8))
	plt.xlabel("time")
	plt.ylabel("station &machine")
	x = np.linspace(0,c_max+10, 100)
	plt.xlim((0,c_max+10)) #x軸長度
	#產生y軸座標
	ylabelname= [0,]
	
	for i in range(W): #工作站數		for j in range(max_M): #平行機台數
			ylabelname.append('s%sm%s'%(i+1,j+1))
	plt.yticks(np.arange(25),ylabelname)
	#工單數10個
	colorlist = ['aquamarine','beige','chartreuse','coral','gold','pink','orange','lavender','lightblue','lightgreen','aquamarine','beige','chartreuse','coral','gold','pink','orange','lavender','lightblue','lightgreen','aquamarine','beige','chartreuse','coral','gold','pink','orange','lavender','lightblue','lightgreen','aquamarine','beige','chartreuse','coral','gold','pink','orange','lavender','lightblue','lightgreen','aquamarine','beige','chartreuse','coral','gold','pink','orange','lavender','lightblue','lightgreen','aquamarine','beige','chartreuse','coral','gold','pink','orange','lavender','lightblue','lightgreen']
	for h in range(N):
		for i in range(n_):  #s橫的一列一列看
			plt.barh(1+max_M*U[h]+machine[h][i], p[h][i], left = s[h][i], color = colorlist[i]) #畫長條吐
			plt.text(s[h][i], 1+max_M*U[h]+machine[h][i], '%s'%(s[h][i]), color = "black") #寫開始時間S
			plt.text(s[h][i]+p[h][i]/2-4, 1+max_M*U[h]+machine[h][i], 'J%s-%s'%(i,h ), color = "red",verticalalignment="top",family = "fantasy",style = "italic") #紀錄J工單-階段
			plt.text((c[h][i]-8), 1+max_M*U[h]+machine[h][i], '%s'%(c[h][i]), color = "black") #完工時間
	
	
	fig = plt.gcf()
	#fig.savefig('C:\\Users\\mb002b\\Desktop\\meeting\\2.png')
	return c_max
#https://blog.csdn.net/TeFuirnever/article/details/88947248



