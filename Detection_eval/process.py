import csv
import math
import os

def combine_fore_back(foreground,background,output_csv):
    back_x = []
    back_y = []
    with open(background, 'r') as file2:
        reader2 = csv.reader(file2)
        # line_list = list(reader)[line_number]
        background_lines = list(reader2)
        for i in range(len(background_lines)-1):
            back_x.append(background_lines[i+1][-2])
            back_y.append(background_lines[i+1][-1])

   
   
    with open(output_csv, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["id","prompt","object_1","d_1","object_2","d_2","obj_net_left","obj_net_up"])
        
        with open(foreground, 'r') as file1:
            reader1 = csv.reader(file1)
            # line_list = list(reader)[line_number]
            foreground_lines = list(reader1)
            
            for j in range(len(foreground_lines)-1):
                # print(foreground_lines[j+1][-2])
                if foreground_lines[j+1][-2]!='' and foreground_lines[j+1][-1]!='':
                    fore_x = foreground_lines[j+1][-2]
                    fore_y = foreground_lines[j+1][-1]
                    num = math.ceil((j+1)/2)-1
                    # print(num,"  ",back_x[num],"  ",fore_x)
                    obj_net_left = float(back_x[num])-float(fore_x)
                    obj_net_up = float(back_y[num]) - float(fore_y)
                    row = foreground_lines[j+1][:6]+[obj_net_left,obj_net_up]
                    writer.writerow(row)
                    output_file.flush()
        
                elif foreground_lines[j+1][-2]=='' and foreground_lines[j+1][-1]=='':
                    row = foreground_lines[j+1][:6]+["",""]
                    writer.writerow(row)
                    output_file.flush()
                else:
                    print("NO WAY")

def object_score(obj1_net_left,left_thresh,obj1_net_up,up_thresh,d_1):
    W = 856
    H = 480
    good_direction = False
    score_tmp = 0
    obj1_net_left = float(obj1_net_left)
    obj1_net_up = float(obj1_net_up)
    if d_1 == "left":
        if obj1_net_left>left_thresh:
            good_direction = True
            score_tmp = abs(obj1_net_left)/W
    elif d_1 == "right":
        if obj1_net_left<-left_thresh:
            good_direction = True
            score_tmp = abs(obj1_net_left)/W
    elif d_1 == "up":
        if obj1_net_up>up_thresh:
            good_direction = True
            score_tmp = abs(obj1_net_up)/H
    elif d_1 == "down":
        if obj1_net_up<-up_thresh:
            good_direction = True
            score_tmp = abs(obj1_net_up)/H
    else:
        print("NO WAY")
    return good_direction,score_tmp
    
def cal_score(output_csv,score_csv): 
    #mid point y:240, x:428  height = 480, width = 856
    left_thresh = 0
    up_thresh = 0
    id = []
    score = []
   
    with open(score_csv, 'w') as score_file:
        writer = csv.writer(score_file)
        writer.writerow(["id","object_1","d_1","object_2","d_2","Score"])
        
        with open(output_csv, 'r') as file1:
            reader1 = csv.reader(file1)
            # line_list = list(reader)[line_number]
            lines = list(reader1)
            vid_num = (len(lines)-1)//2
            for i in range(vid_num):
                id = lines[i*2+1][0]
                d_1 = lines[i*2+1][3]
                d_2 = lines[i*2+2][5]
                obj1 = lines[i*2+1][2]
                obj2 = lines[i*2+2][4]
                obj1_net_left = lines[i*2+1][6]
                obj1_net_up = lines[i*2+1][7]
                obj2_net_left = lines[i*2+2][6]
                obj2_net_up = lines[i*2+2][7]
                good_direction = False
                score_tmp = 0
                
                if d_1!="" and d_2=="": #only 1 object
                    if obj1_net_left != "": #1 object detected
                        good_direction,score_tmp = object_score(obj1_net_left,left_thresh,obj1_net_up,up_thresh,d_1)
                    elif obj1_net_left == "":  #1 object not detected
                        good_direction = False
                        score_tmp=-1
                    else:
                        print("NO WAY 2")
                        
                elif d_1!="" and d_2!="": #2 objects  
                    if obj1_net_left != "": #1st object detected
                        good_direction_1,score_tmp_1 = object_score(obj1_net_left,left_thresh,obj1_net_up,up_thresh,d_1)
                    elif obj1_net_left == "": #1st object not detected
                        good_direction_1 = False
                        score_tmp_1=-1
                    else:
                        print("NO WAY 4")
                    
                    if obj2_net_left != "": #2nd object detected
                        good_direction_2,score_tmp_2 = object_score(obj2_net_left,left_thresh,obj2_net_up,up_thresh,d_2)
                    elif obj2_net_left == "": #2nd object not detected
                        good_direction_2 = False
                        score_tmp_2=-1
                    else:
                        print("NO WAY 4")
                    
                    if good_direction_1==True and good_direction_2==True:
                        score_tmp = (score_tmp_1 + score_tmp_2)*0.5
                    elif good_direction_1==True and good_direction_2==False:
                        score_tmp = 0.5*score_tmp_1
                    elif good_direction_1==False and good_direction_2==True:
                        score_tmp = 0.5*score_tmp_2
                    elif good_direction_1==False and good_direction_2==False:
                        if score_tmp_1<0 and score_tmp_2<0: #2 object 都没detect
                            score_tmp = -1
                        else:
                            score_tmp = 0
                    else:
                        print("NO WAY 5")
                    
                        
                else:
                    print("NO WAY 3")     
            
                score.append(score_tmp)
                # writer.writerow(["id","object_1","d_1","object_2","d_2","Score"])
                writer.writerow([id,obj1,d_1,obj2,d_2,score_tmp])
                
def model_score(csv_path):
    
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        # line_list = list(reader)[line_number]
        lines = list(reader)
        score = 0   #neither detected: -1, detected: motion score 0~1, total scale: -1 ~ 1
        cnt = 0
        score_pos = 0
        cnt_pos = 0
        for line in lines[1:]:
        
            score_tmp = float(line[-1])
            
            if score_tmp<0:
                score_tmp = 0
            elif score_tmp>=0:
                score_pos += score_tmp
                score_tmp = (score_tmp*0.8)+0.2
                cnt_pos+=1
                
            score+=score_tmp
            
            cnt+=1
        
        score = score/cnt
        score_pos = score_pos/cnt_pos
        print("score: ",score)
      

    
                
if __name__ == "__main__":
    csv_path = "../csv_dir/motion_binding"
    
    foreground = os.path.join(csv_path,"foreground.csv")
    background = os.path.join(csv_path,"background.csv")
    output_csv = os.path.join(csv_path,"back_fore.csv")
    combine_fore_back(foreground,background,output_csv)

    score_csv = os.path.join(csv_path,"score.csv")
    cal_score(output_csv,score_csv)
    
    model_score(score_csv)
    