
import math
import pandas
with open("data/myoutput.csv", 'r') as csvfile:
    df = pandas.read_csv(csvfile)
 # The names of all the columns in the data.

#,cls,timestep,id,x,y,heading,speed,acceleration,label,Unnamed: 8,Unnamed: 9
# distance_columns = ['x', 'y']


print(df.columns.values)






TEXT_EDGE_INDEX = "edge_index"
TEXT_EDGE_WEIGHT = "edge_weight"
TEXT_X = "x"
TEXT_Y = "y"
TEXT_ID = "id"




ADDITIONAL_WEIGHT_FEATURES = [TEXT_X,TEXT_Y,"heading","speed","acceleration"]

field_props = {}
for field in ADDITIONAL_WEIGHT_FEATURES:
    x_pros = df[field].describe().transpose()
    field_props[field]={
        "max":x_pros["max"],
        "min":x_pros["min"],
        "null":df[field].isnull().values.any(),
        "nan":df[field].isna().values.any()
    } 
field_props





import numpy as np 
import json
from tqdm import tqdm
import warnings
ADD_FEATURES_AS_SELF_WEIGHT=True
NEGATIVE_WEIGHT = False

warnings.simplefilter(action='ignore', category=FutureWarning)
unique_node_names = np.unique(df['id']).tolist()
index_to_node = {x: unique_node_names[x] for x in range(0, len(unique_node_names))}
node_to_index = {unique_node_names[x]:x  for x in range(0, len(unique_node_names))}
for num_edges in tqdm([5]): #0,1,2,3,4,5,50,100,150,200,10000
    to_export = {
        TEXT_EDGE_INDEX : { },
        TEXT_EDGE_WEIGHT : { },
        "time_periods": 1000,
        TEXT_Y: [],
        "features": [], # x,y,heading,speed,acceleration
        "node_labels" : []
    }
    to_export["node_labels"] = unique_node_names
    cnt = 0
    for step, d_step in df.groupby(['timestep']):
        step = str(int(step))
        if int(step)> 998:
            arr = np.zeros(len(unique_node_names)).tolist()
            arr_features = np.zeros((len(unique_node_names),5)).tolist()
            to_export[TEXT_EDGE_INDEX][step] = []
            to_export[TEXT_EDGE_WEIGHT][step] = []
            to_export[TEXT_Y].append(arr)
            to_export["features"].append(arr_features)
            for cls, d_cls in d_step.groupby(['cls']):
                added = {}
                cur_edge = 0
                for x, row_x in d_cls.iterrows():
                    lbl = row_x["label"]
                    from_node_index = node_to_index[row_x[TEXT_ID]]
                    arr[from_node_index]=lbl
                    if lbl>0:
                        pass
                
                    additional_fields = {}
                    current_field_index = 0
                    for field in ADDITIONAL_WEIGHT_FEATURES:
                        x_val = row_x[field]
                        min_value = field_props[field]["min"]

                        if not x_val or np.isnan(x_val):
                            x_val=0.000000000001 # TODO: should we use min value or zero? x=>? , acceleration zero ok?
                        else:
                            x_val = float(x_val)

                        if min_value<0:
                            x_val = x_val + abs(field_props[field]["min"])
                        
                        arr_features[from_node_index][current_field_index]=x_val
                        additional_fields[field]=x_val
                        current_field_index+=1
    
                    if ADD_FEATURES_AS_SELF_WEIGHT:
                        for field in ADDITIONAL_WEIGHT_FEATURES:
                            to_export[TEXT_EDGE_INDEX][step].append([from_node_index,from_node_index ])
                            to_export[TEXT_EDGE_WEIGHT][step].append(additional_fields[field])
                for x, row_x in d_cls.iterrows():

                    for y, row_y in d_cls.iterrows():
                        _key = str(x) + "_" + str(y)
                        _key_rev = str(y) + "_" + str(x)
                        if x!=y and not added.get(_key) and not added.get(_key_rev):
                            # from_node_id = node_to_index[row_x[TEXT_ID]]
                            to_node_id = node_to_index[row_y[TEXT_ID]]
                            to_export[TEXT_EDGE_INDEX][step].append([from_node_index,to_node_id ])
                            to_export[TEXT_EDGE_INDEX][step].append([to_node_id, from_node_index])
                            a = np.array((x, y))
                            b = np.array((row_y[TEXT_X], row_y[TEXT_Y]))
                            # https://stackoverflow.com/a/1401828/1290868
                            dist = np.linalg.norm(a-b) 
                            dist = -dist if NEGATIVE_WEIGHT else dist 
                            dist = lbl if lbl ==0 else (-dist/lbl if NEGATIVE_WEIGHT else dist*lbl )
                            to_export[TEXT_EDGE_WEIGHT][step].append(dist)
                            to_export[TEXT_EDGE_WEIGHT][step].append(dist)
                            added[_key]=True
                            added[_key_rev]=True
                            cur_edge +=1
                            if cur_edge > num_edges:
                                break
                    if cur_edge > num_edges:
                        break
    with open(f"data/myoutput_{num_edges}_edges{'_negative' if NEGATIVE_WEIGHT else '_positive'}{'_with_features_as_self_edge' if ADD_FEATURES_AS_SELF_WEIGHT else ''}.json", "w") as outfile:
        json.dump(to_export,outfile)