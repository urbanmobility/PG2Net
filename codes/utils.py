import pickle
from collections import defaultdict
import numpy as np

data = pickle.load(open('foursquare_test.pk', 'rb'), encoding='iso-8859-1')
vid_list = data['vid_list']
uid_list = data['uid_list']
data_neural = data['data_neural']
poi_coordinate = data['vid_lookup']

#计算时间相似度
def caculate_time_sim(data_neural):
    time_checkin_set = defaultdict(set)
    for uid in data_neural: #每个用户
        uid_sessions = data_neural[uid]
        train_id = data_neural[uid]['train'] #训练id
        for c, sid in enumerate(train_id):
        #for sid in uid_sessions['sessions']: #每个用户中的每个时段
            session_current = uid_sessions['sessions'][sid]
            for checkin in session_current: #每个时段中的每个序列
                timid = checkin[1] #时间
                locid = checkin[0] #POI位置
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid) #获得每个时序中的出现过的位置
    sim_matrix = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            set_i = time_checkin_set[i]  #这个时间内出现过的位置
            set_j = time_checkin_set[j]  #这个时间内出现过的位置
            jaccard_ij = len(set_i & set_j)/len(set_i | set_j)  #获得任意两个时间的相似性
            sim_matrix[i][j] = jaccard_ij
    return sim_matrix  #是一个对称矩阵

#计算距离矩阵
def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance
def caculate_poi_distance(poi_coors):
    #print("distance matrix")
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    for i in range(len(poi_coors)):
        for j in range(i,len(poi_coors)):
            poi_current = i + 1
            poi_target = j + 1
            poi_current_coor = poi_coors[poi_current]
            poi_target_coor = poi_coors[poi_target]
            distance_between = geodistance(poi_current_coor[0], poi_current_coor[1], poi_target_coor[0], poi_target_coor[1])
#             if distance_between < 1:
#                 distance_between = 1
            sim_matrix[poi_current][poi_target] = distance_between
            sim_matrix[poi_target][poi_current] = distance_between
    pickle.dump(sim_matrix, open('distance_nyc.pkl', 'wb'))
    return sim_matrix

#计算t和category的关系
def caculate_time_cid(data_neural):
    sim_matrix = np.zeros((48,242))
    for uid in data_neural: #每个用户
        uid_sessions = data_neural[uid]
        for sid in uid_sessions['sessions']: #每个用户中的每个时段
            session_current = uid_sessions['sessions'][sid]
            for checkin in session_current: #每个时段中的每个序列
                timid = checkin[1] #时间
                locid = checkin[2] #POI类别
                sim_matrix[timid,locid] += 1
    return sim_matrix

#构造location和location category 有向有权图
def construc_graph(data_neural):
    time_checkin_set = {}
    for uid in data_neural: #每个用户
        uid_sessions = data_neural[uid]
        train_id = data_neural[uid]['train'] #训练id
        for c, sid in enumerate(train_id):
            session_current = uid_sessions['sessions'][sid]
            size = len(session_current)
            for i in range(1,size):
                locid_pre = session_current[i][2]
                locid_aft = session_current[i-1][2]
                msg = "{}  {}".format(locid_aft,locid_pre)
                if msg not in time_checkin_set:
                    time_checkin_set[msg] = 1
                else:
                    time_checkin_set[msg] += 1
    with open("cid.txt","a") as file:
        for data in time_checkin_set.items():
            msg = data[0] +"  "+ str(data[1])
            file.write(msg + "\n")
    file.close()

