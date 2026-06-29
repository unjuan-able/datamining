import onnxruntime as ort
import numpy as np
import pandas as pd
import torch
import os
import time
from sklearn.preprocessing import StandardScaler
from normalize_feature_infer import normalize_data,process_folder
def get_image_list(image_path):
    valid_suffix=['.csv']
    image_list=[]
    image_dir=[]
    if os.path.isfile(image_path):
        if os.path.splittext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir=os.path.dirname(image_path)
            with open(image_path,'r') as f:
                for line in f:
                    line=line.strip()
                    if len(line.strip())>1:
                        line=line.split()[0]
                    image_list.append(os.path.join(image_dir,line))
    elif os.path.isdir(image_path):
        image_dir=image_path
        for root,dirs,files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.startswith('.'):
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root,f))
    else:
        print("not found")
    return image_list,image_dir
model_path="00000000"
ort_session=ort.InferenceSession(model_path,providers=['CPUExecutionProvider'])
folder_path="000000000"
output_txt="0000000000"
THRES=0.5
scaler=StandardScaler()
import joblib
scaler=joblib.load('')
csv_list1,csv_dir1=get_image_list(folder_path)
correct_count=0
metric={"tp":0,"fn":0,"fp":0,"tn":0}
start=time.time()
with open(output_txt,'w',encoding='utf-8') as f:
    for ind, file_path in enumerate(csv_list1):
        try:
            data=normalize_data(file_path)
            data=process_folder(data)
            final_pred=0
            X=pd.DataFrame(data).values
            X=scaler.transform(X).astype(np.float32)
            if np.isnan(X).any():
                raise ValueError("数据包含NaN")
            onnx_output=ort_session.run(None,{"input":X})[0]
            outputs =torch.from_numpy(onnx_output)
            y_pred_prob=torch.softmax(outputs,dim=1)
            score=y_pred_prob[0][1]
            final_pred=int(score>THRES)
            if '/20260319_abnormal_data/' in file_path:
                label=1
            else:
                label=0
            if label==0:
                if final_pred !=0:
                    metric["fp"]+=1
                else:
                    metric["tn"]+=1
            else:
                if final_pred !=0:
                    metric["tp"]+=1
                else:
                    metric["fn"]+=1
            file_count+=1
            f.write(f"{file_path},{score},{final_pred},{label}\n")
        except Exception as e:
            print(file_path,e)
            countinue
print(time.time()-start)
print((time.time()-start)/file_count)
print(metric)
score={"acc:0","precision:0","recall:0"}
score["acc"]=(metric["tp"]+metric["tn"])/(metric["tp"]+metric["tn"]+metric["fp"]+metric["fn"])
score["recall"]=(metric["tp"]/(metric["tp"]+metric["fn"]))
print(score)
