import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
from phyre_rolllout_collector import load_phyre_rollouts, collect_solving_observations, collect_solving_dataset
from matplotlib import cm
import matplotlib.colors as colors
import cv2
import phyre
import os
import pickle
import random
import json
import gzip
from PIL import ImageDraw, Image, ImageFont
import io
import logging as L


def make_dual_dataset(path, size=(32,32), save=True):
    if os.path.exists(path+".pickle"):
        with open(path+'.pickle', 'rb') as fhandle:
            X, Y = pickle.load(fhandle)
    else:
        X = load_phyre_rollouts(path)
        X, Y = prepare_data(X, size)
        X = T.tensor(X).float()
        Y = T.tensor(Y).float()
        if save:
            with open(path+'.pickle', 'wb') as fhandle:
                pickle.dump((X,Y), fhandle)
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X,Y), 32, shuffle=True)
    return dataloader

def make_mono_dataset_old(path, size=(32,32), save=True, tasks=[], shuffle=True):
    if os.path.exists(path+".pickle") and os.path.exists(path+"_index.pickle"):
       X = T.load(path+'.pickle')
       index = T.load(path+'_index.pickle')
       print(f"Loaded dataset from {path} with shape:", X.shape)
    else:
        if tasks:
            collect_solving_observations(path, tasks, n_per_task=1, stride=5, size=size)
        data_generator = load_phyre_rollout_data(path)
        data, index = format_raw_rollout_data(data_generator, size=size)
        X = T.tensor(data).float()
        if save:
            T.save(X, path+'.pickle')
            T.save(index, path+'_index.pickle')
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), 32, shuffle=shuffle)
    return dataloader, index

def make_mono_dataset(path, size=(32,32), tasks=[], batch_size = 32, solving=True, n_per_task=1, shuffle=True, proposal_dict=None, dijkstra=False, pertempl=False):
    if os.path.exists(path+"/data.pickle") and os.path.exists(path+"/index.pickle"):
        try:
            with gzip.open(path+'/data.pickle', 'rb') as fp:
                X, Y = pickle.load(fp)
                X = T.tensor(X).float()
                Y = T.tensor(Y).float()
        except OSError as e:
            print("WARNING still unzipped data file at", path)
            with open(path+'/data.pickle', 'rb') as fp:
                X, Y = pickle.load(fp)
                X = T.tensor(X).float()
                Y = T.tensor(Y).float()
        with open(path+'/index.pickle', 'rb') as fp:
            index = pickle.load(fp)
        
        # TRAIN TEST SPLIT
        print(f"Loaded dataset from {path} with shape:", X.shape)
    else:
        if proposal_dict is None:
            train_ids, dev_ids, test_ids = phyre.get_fold("ball_within_template", 0)
            all_tasks = train_ids + dev_ids + test_ids
        else:
            all_tasks = tasks
        collect_solving_dataset(path, all_tasks, n_per_task=n_per_task, stride=5, size=size, solving=solving, proposal_dict=proposal_dict, dijkstra=dijkstra, pertempl=pertempl)
        with gzip.open(path+'/data.pickle', 'rb') as fp:
            X, Y = pickle.load(fp)
        with open(path+'/index.pickle', 'rb') as fp:
            index = pickle.load(fp)
        X = T.tensor(X).float()
        Y = T.tensor(Y).float()
        print(f"Loaded dataset from {path} with shape:", X.shape)
    
    # MAKE CORRECT SELECTION
    selection = [i for (i,task) in enumerate(index) if task in tasks]
    #print(len(index), len(tasks), len(selection))
    X = X[selection]
    Y = Y[selection]
    index = [index[s] for s in selection]
    L.info(f"Loaded dataset from {path} with shape: {X.shape}")

    assert len(X)==len(Y)==len(index), "All should be of equal length"
        
    X = X/255 # correct for uint8 encoding
    I = T.arange(len(X), dtype=int)
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X, Y, I), batch_size, shuffle=shuffle)
    return dataloader, index

def shrink_data(path):
    for folder in os.listdir(path):
        if folder.__contains__("64xy"):
            print("loading:", folder)
            try:
                with open(path+'/'+folder+'/data.pickle', 'rb') as fp:
                    data = pickle.load(fp)
                    data = (np.array(data)*255).astype(np.uint8)
                with gzip.GzipFile(path+'/'+folder+'/data.pickle', 'wb') as fp:
                    pickle.dump(data, fp)
            except Exception as e:
                print(f"error loading {folder}:\n{e}")
            finally:
                print(folder, "finished")

def compare_viz(paths, column):
    tlist = []
    beginning = "isy2020/phyre/result/flownet/inspect/VS-bottleneck/default/ball_within_template_fold_0/poch_0_0_diff.png"
    for path in paths:
        if type(path==tuple):
            name = "isy2020/phyre/result/flownet/inspect/"+path[0]+path[1]+"/ball_within_template_fold_0/eval-viz-tensor.pt"
        else:
            name = "isy2020/phyre/result/flownet/inspect/"+path+"/default/ball_within_template_fold_0/eval-viz-tensor.pt"
        tensor = T.load(name)
        tlist.append(tensor[:,column])
    T.stack(tlist, dim=1)
    rows = [str(i/4) for i in range(100)]
    vis_batch(tlist, "./", "viz_compare.png", rows = rows)
    

def vis_batch(batch, path, pic_id, text = [], rows=[], descr=[], save=True, font_size=11):
    #print(batch.shape)

    if len(batch.shape) == 4:
        padded = F.pad(batch, (1,1,1,1), value=0.5)
    elif len(batch.shape) == 5:
        padded = F.pad(batch, (0,0,1,1,1,1), value=0.5)
    else:
        print("Unknown shape:", batch.shape)

    #print(padded.shape)
    reshaped = T.cat([T.cat([channels for channels in sample], dim=1) for sample in padded], dim=0)
    #print(reshaped.shape)
    if np.max(reshaped.numpy())>1.0:
        reshaped = reshaped/256
    if path:
        os.makedirs(path, exist_ok=True)
    if text or rows or descr:
        if rows:
            row_width = int(1.5*font_size//2*max([len(item) for item in rows]))
        else:
            row_width = 0

        if descr:
            descr_wid = int(1.5*font_size//2*max([len(item) for item in descr]))
        else:
            descr_wid = 0
        if text:
            text_height= int(1.5*font_size*max([len(item.split('\n')) for item in text]))
        else:
            text_height=0

        if len(reshaped.shape) == 2:
            reshaped = F.pad(reshaped, (row_width,descr_wid,text_height,0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy()*255), mode="L")
        elif len(reshaped.shape) == 3:
            reshaped = F.pad(reshaped, (0,0,row_width,descr_wid,text_height,0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy()*255))
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)
        draw = ImageDraw.Draw(img)
        
        for i, words in enumerate(text):
            x, y = 3+row_width+i*(reshaped.shape[1]-row_width-descr_wid)//len(text), 0
            draw.text((x, y), words, fill=(0) if len(reshaped.shape)==2 else (0,0,0), font=font)

        for j, words in enumerate(rows):
            x,y = 3, 10+text_height+j*(reshaped.shape[0]-text_height)//len(rows)
            draw.text((x, y), words, fill=(0) if len(reshaped.shape)==2 else (0,0,0), font=font)
        
        for j, words in enumerate(descr):
            x,y =  5+reshaped.shape[1]-descr_wid, text_height+j*(reshaped.shape[0]-text_height)//len(descr)
            #print(x,y)
            draw.text((x, y), words, fill=(0) if len(reshaped.shape)==2 else (0,0,0), font=font)

        if save:
            img.save(f'{path}/'+pic_id+'.png')
        else:
            return img
    else:
        if save:
            plt.imsave(f'{path}/'+pic_id+'.png', reshaped.numpy(), dpi=1000)
        else:
            return reshaped

def gifify(batch, path, pic_id, text = [], constant=None):
    #print(batch.shape)
    if np.max(batch.numpy())>1.0:
        batch = batch/256

    if len(batch.shape) == 4:
        padded = F.pad(batch, (1,1,1,1), value=0.5)
    elif len(batch.shape) == 5:
        padded = F.pad(batch, (0,0,1,1,1,1), value=0.5)
    else:
        print("Unknown shape:", batch.shape)

    os.makedirs(path, exist_ok=True)

    frames = []
    for f_id in range(padded.shape[1]):
        frame = padded[:,f_id]
        frame = T.cat([sample for sample in frame], dim=1)
        if text:
            text_height= 30
            if len(frame.shape) == 2:
                #frame = F.pad(frame, (0,0,text_height,0), value=0.0)
                img = Image.fromarray(np.uint8(frame.numpy()*255), mode="L")
            elif len(frame.shape) == 3:
                #frame = F.pad(frame, (0,0,0,0,text_height,0), value=0.0)
                img = Image.fromarray(np.uint8(frame.numpy()*255))
            font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 9)
            draw = ImageDraw.Draw(img)
            for i, words in enumerate(text):
                x, y = i*frame.shape[1]//len(text), 0
                draw.text((x, y), words, fill=(0) if len(frame.shape)==2 else (0,0,0), font=font)
        else:
            if len(frame.shape) == 2:
                img = Image.fromarray(np.uint8(frame.numpy()*255), mode="L")
            elif len(frame.shape) == 3:
                img = Image.fromarray(np.uint8(frame.numpy()*255))
               
        if constant is not None:
            dst = Image.new('RGB', (img.width, img.height + constant.height), (255, 255, 255))
            dst.paste(constant, (0, 0))
            dst.paste(img, (0, constant.height))

            img = dst
            
        frames.append(img)
        
    frames[0].save(f'{path}/'+pic_id+'.gif', save_all=True, append_images=frames[1:], optimize=True, duration=300, loop=0)

def make_visuals():
    sim = phyre.initialize_simulator(["00018:013", "00020:007", "00018:035"], 'ball')
    res = sim.simulate_action(0, sim.sample(0), stride=40)
    while not res.status.is_solved():
        res = sim.simulate_action(0, sim.sample(0), stride=40)
    #init.save("result/visuals/init1.png")
    
    obs = phyre.observations_to_uint8_rgb(sim.initial_scenes[0])
    obs = np.pad(obs, ((5,5),(5,5),(0,0)))
    init1 = Image.fromarray(obs)
    init = init1.copy()
    for frame in res.images:
        obs = phyre.observations_to_uint8_rgb(frame)
        obs = np.pad(obs, ((5,5),(5,5),(0,0)))
        frame = np.pad(frame, ((5,5),(5,5)))

        objects = Image.fromarray(np.flip((frame!=0), axis=0).astype(np.uint8)*100)
        pic = Image.fromarray(obs)
        #pic.putalpha(0.5)
        init.paste(pic, (0,0), objects)
    blended1 = init

    res = sim.simulate_action(1, sim.sample(1), stride=20)
    while not res.status.is_solved():
        res = sim.simulate_action(1, sim.sample(1), stride=20)
    #init.save("result/visuals/init1.png")

    obs = phyre.observations_to_uint8_rgb(sim.initial_scenes[1])
    obs = np.pad(obs, ((5,5),(5,5),(0,0)))
    init2 = Image.fromarray(obs)    
    init = init2.copy()
    for frame in res.images:
        obs = phyre.observations_to_uint8_rgb(frame)
        obs = np.pad(obs, ((5,5),(5,5),(0,0)))
        frame = np.pad(frame, ((5,5),(5,5)))

        objects = Image.fromarray(np.flip((frame!=0), axis=0).astype(np.uint8)*100)
        pic = Image.fromarray(obs)
        #pic.putalpha(0.5)
        init.paste(pic, (0,0), objects)
    blended2 = init

    base = 256+10
    back = Image.new("RGB",(4*base+15, base))
    back.paste(init1, (0,0))
    back.paste(blended1, (base+5,0))
    back.paste(init2, (2*base+10,0))
    back.paste(blended2, (3*base+15,0))
    os.makedirs("result/visuals", exist_ok=True)
    back.save("result/visuals/phyre.png")


    """
    obs = phyre.observations_to_uint8_rgb(sim.initial_scenes)
    print(obs.shape)
    padded = np.flip(np.pad(obs, ((0,0),(5,5),(5,5),(0,0))), axis=1)
    print(padded.shape)
    init = Image.fromarray(np.concatenate(padded, axis=1))
    """
    #init.save("result/visuals/blended1.png")
    #objects.save("result/visuals/red.png")"""

def prepare_data(data, size):
    targetchannel = 1
    X, Y = [], []
    print("Preparing dataset...")
    #x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for variations in data:
        with_base = len(variations) > 1
        for (j, rollout) in enumerate(variations):
            if not isinstance(rollout, np.ndarray):
                break
            #length = (2*len(rollout))//3
            #rollout = rollout[:length]
            roll = np.zeros((len(rollout), 7, size[0], size[1]))
            for i, scene in enumerate(rollout):
                channels = [(scene==j).astype(float) for j in range(1,8)]
                roll[i] = np.stack([(cv2.resize(c, size, cv2.INTER_MAX)>0).astype(float) for c in channels])
            roll = np.flip(roll, axis=2)
            trajectory = (np.sum(roll[:,targetchannel], axis=0)>0).astype(float)
            if not(with_base and j == 0):
                action = (np.sum(roll[:,0], axis=0)>0).astype(float)
            #goal_prior = dist_map(roll[0, 2] + roll[0, 3])
            #roll[0, 0] = goal_prior
            # TESTING ONLY
            #roll[0, 1] = roll[0, 0]
            if with_base and j == 0:
                base = trajectory
            else:
                action_ball = roll[0, 0].copy()
                roll[0, 0] = np.zeros_like(roll[0,0])
                #print(goal_prior)
                # Contains the initial scene without action
                X.append(roll[0])
                # Contains goaltarget, actiontarget, basetrajectory
                Y.append(np.stack((trajectory, action, base if with_base else np.zeros_like(roll[0,0]), action_ball)))
                #plt.imshow(trajectory)
                #plt.show()
    print("Finished preparing!")
    return X, Y

def extract_channels_and_paths(rollout, path_idxs=[1,0], size=(32,32), gamma=1):
    """
    returns init scenes from 'channels' followed by paths specified by 'path_idxs' 
    """
    paths = np.zeros((len(path_idxs), len(rollout), size[0], size[1]))
    alpha = 1
    for i, chans in enumerate(rollout):
        # extract color codings from channels
        #chans = np.array([(scene==ch).astype(float) for ch in channels])

        # if first frame extract init scene
        if not i:
            init_scene = np.array([(cv2.resize(chans[ch], size, cv2.INTER_MAX)>0).astype(float) for ch in range(len(chans))])

        # add path_idxs channels to paths
        for path_i, idx in enumerate(path_idxs):
            paths[path_i, i] = alpha*(cv2.resize(chans[idx], size, cv2.INTER_MAX)>0).astype(float)
        alpha *= gamma
    
    # flip y axis and concat init scene with paths
    paths = np.flip(np.max(paths, axis=1).astype(float), axis=1)
    init_scene = np.flip(init_scene, axis=1)
    result = np.concatenate([init_scene, paths])
    return result

def format_raw_rollout_data(data, size=(32,32)):
    targetchannel = 1
    data_bundle = []
    lib_dict = dict()
    print("Formating data...")
    #x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, (base, trial, info) in enumerate(data):
        print(f"at sample {i}; {info}")
        #base_path = extract_channels_and_paths(base, channels=[1], path_idxs=[0], size=size)[1]
        #trial_channels = extract_channels_and_paths(trial, size=size)
        #sample = np.append(trial_channels, base_path[None], axis=0)
        try:
            task, subtask, number = info
            base_path = extract_channels_and_paths(base, path_idxs=[1], size=size)[-1]
            trial_channels = extract_channels_and_paths(trial, path_idxs=[1,2,0], size=size)
            sample = np.append(trial_channels, base_path[None], axis=0)
            #plt.imshow(np.concatenate(tuple(np.concatenate((sub, T.ones(32,1)*0.5), axis=1) for sub in sample), axis=1))
            #plt.show()
            data_bundle.append(sample)
            
            # Create indexing dict
            key = task+':'+subtask
            if not key in lib_dict:
                lib_dict[key] = [i]
            else:
                lib_dict[key].append(i)
        except Exception as identifier:
            print(identifier)
    print("Finished preparing!")
    return data_bundle, lib_dict

def load_phyre_rollout_data(path, base=True):
    s = "/"
    fp ="observations.pickle"
    for task in os.listdir(path):
        for variation in os.listdir(path+s+task):
            if base:
                with open(path+s+task+s+variation+s+'base'+s+fp, 'rb') as handle:
                    base_rollout =  pickle.load(handle)
            for trialfolder in os.listdir(path+s+task+s+variation):                
                final_path = path+s+task+s+variation+s+trialfolder+s+fp
                with open(final_path, 'rb') as handle:
                    trial_rollout = pickle.load(handle)
                if base:
                    yield(base_rollout, trial_rollout, (task, variation, trialfolder))
                else:
                    yield(trial_rollout)
    
def draw_ball(w, x, y, r, invert_y=False):
        """inverts y axis """
        x = int(w*x)
        y = int(w*(1-y)) if invert_y else int(w*y)
        r = w*r
        X = T.arange(w).repeat((w, 1)).float()
        Y = T.arange(w).repeat((w, 1)).transpose(0, 1).float()
        X -= x # X Distance
        Y -= y # Y Distance
        dist = (X.pow(2)+Y.pow(2)).pow(0.5)
        return (dist<r).float()

def action_delta_generator(pure_noise=False):
    temp = 1
    radfac = 0.025
    coordfac = 0.1
    
    #for x,y,r in zip([0.05,-0.05,0.1,-0.1],[0,0,0,0],[-0.1,-0.2,-0.3,0]):
        #yield x,y,r

    if not pure_noise:
        for fac in [0.5,1,2]:
            for rad in [0,1,-1]:
                for xd,yd in [(1,0), (-1,0), (2,0), (-2,0), (-1,2), (1,2), (-1,-2), (-1,-2)]:
                    #print((fac*np.array((coordfac*xd, coordfac*yd, rad*radfac))))
                    yield (fac*np.array((coordfac*xd, coordfac*yd, rad*radfac)))
    count = 0
    while True:
        count += 1
        action = ((np.random.randn(3))*np.array([0.2,0.1,0.2])*temp)*0.1
        #print(count,"th", "ACTION:", action)
        if np.linalg.norm(action)<0.05:
            continue
        yield action
        temp = 1.04*temp if temp<5 else temp

def pic_to_action_vector(pic, r_fac=1):
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y,x]:
                X += pic[y,x]*x
                Y += pic[y,x]*y
    summed = pic.sum()
    X /= pic.shape[0]*summed
    Y /= pic.shape[0]*summed
    r = np.sqrt(pic.sum()/(3.141592*pic.shape[0]**2))
    return [X.item(), 1-Y.item(), r_fac*r.item()]

def grow_action_vector(pic, r_fac=1, show=False, num_seeds=1, mask=None, check_border=False, updates=5):
    id = int((T.rand(1)*100))
    #os.makedirs("result/flownet/solver/grower", exist_ok=True)
    #plt.imsave(f"result/flownet/solver/grower/{id}.png", pic)
    pic = pic*(pic>pic.mean())
    #plt.imsave(f"result/flownet/solver/grower/{id}_thresh.png", pic)


    wid = pic.shape[0]

    def get_value(x,y,r):
        ball = draw_ball(wid, x, y, r)
        potential = T.sum(ball)
        actual = T.sum(pic[ball.bool()])
        value =  (actual**0.5)*actual/potential
        if mask is not None:
            overlap = mask[ball>0].sum()
            if overlap>0:
                return -overlap
        if check_border and ((x-r)<-0.00 or (y-r)<-0.00 or (x+r)>1.00 or (y+r)>1.00):
            return min([x-r, 1-(x+r), y-r, 1-(y+r)])
        return value

    def move_and_grow(x,y,r,v):
        delta = 0.7
        positions = [(x+dx,y+dy) for (dx,dy) in [(-(0.3+delta)/30,0), ((0.3+delta)/30,0), (0,-(0.3+delta)/30), (0,(0.3+delta)/30)] if (0<=x+dx<1) and (0<=y+dy<1)]
        bestpos = (x,y)
        bestrad = r
        bestv = v
        for pos in positions:
            value = get_value(*pos, r)
            rad, val = grow(*pos,r,value)
            if val>bestv:
                bestpos = pos
                bestrad = rad
                bestv = val
        return bestpos[0], bestpos[1], bestrad, bestv

    def grow(x,y,r,v):
        bestv = v
        bestrad = r
        for rad in [r+0.005, r+0.01, r+0.03, r-0.01]:
            if 0<rad<0.3:
                value = get_value(x,y,rad)
                if value>bestv:
                    bestv = value
                    bestrad = rad
        return bestrad, bestv

    seeds = []
    while len(seeds)<num_seeds:
        r = 0.04 +np.random.rand()*0.05
        try:
            y, x = random.choice(T.nonzero((pic>0.01)))+T.rand(2)*0.05
            seeds.append((x.item()/wid,y.item()/wid,r))
        except Exception as e:
            print("EXCEPTION", e)
            y, x = wid//2, wid//2
            seeds.append((x/wid,y/wid,r))

    final_seeds = []
    for (x,y,r) in seeds:
        v = get_value(x,y,r)
        #plt.imshow(pic+draw_ball(wid,x,y,r))
        #plt.show()
        for i in range(updates):
            x, y, r, v = move_and_grow(x,y,r,v)
            #r, v = grow(x,y,r,v)
            if show:
                print(x,y,r,v)
                plt.imshow(pic+draw_ball(wid,x,y,r))
                plt.show()
        final_seeds.append(((x,y,r),v))
        
    action = np.array(max(final_seeds, key= lambda x: x[1])[0])
    action[1] = 1-action[1]
    #plt.imsave(f"result/flownet/solver/grower/{id}_drawn.png", draw_ball(wid, *action, invert_y = True))
    action[2]*=r_fac
    compare_action = action.copy()
    action[2] = action[2] if action[2]<0.125 else 0.125
    action[0] = action[0] if action[0]> 0 else  0
    action[0] = action[0] if action[0]<1- 0 else 1- 0
    action[1] = action[1] if action[1]> 0 else  0
    action[1] = action[1] if action[1]<1- 0 else 1- 0
    if np.any(action!=compare_action):
        #print("something was out of bounce:", action, compare_action)
        pass
    return action

def sample_action_vector(pic, actions, uniform=False, radmode="random", show=False, mask=None, check_border=False):
    #os.makedirs("result/flownet/solver/grower", exist_ok=True)
    #plt.imsave(f"result/flownet/solver/grower/{id}.png", pic)
    pic = pic*(pic>pic.mean())
    wid = pic.shape[0]
    #plt.imsave(f"result/flownet/solver/grower/{id}_thresh.png", pic)
    if check_border:
        mask[:,0] = 1
        mask[:,-1] = 1
        mask[0,:] = 1
        mask[-1,:] = 1


    shape = pic.shape
    flat = pic.numpy().reshape(-1)
    flat = flat/np.sum(flat)
    indexes = np.where(flat)[0]
    probs = flat[flat>0]
    #print(probs, indexes)
    
    valid = False
    while not valid:
        # POS SAMPLING
        if uniform:
            flat_choice = np.random.choice(indexes)
        else:
            flat_choice = np.random.choice(indexes, p=probs)
        flat_mask = flat * 0
        flat_mask[flat_choice] = 1
        reshaped = flat_mask.reshape(shape)
        choice = np.where(reshaped)
        #print(choice)
        y, x = choice[0].item()/wid, choice[1].item()/wid

        # RADIUS Selection
        radii = actions[:,2]
        radii = radii[radii>0.005]

        if radmode=="random":
            rad = np.random.choice(radii)
            ball = draw_ball(wid, x, y, rad/8)>0
            if mask[ball].sum():
                continue
            else:
                valid = True
        else:
            rads = []
            tries = 0
            while len(rads)<10:
                tries += 1
                #print(tries)
                valid = True
                rad = np.random.choice(radii)
                ball = draw_ball(wid, x, y, rad/8)>0
                #print(ball)
                if tries>50:
                    print("tried 50 radii!")
                    valid = False
                    break

                if mask[ball].sum():
                    continue
                if radmode=="mean":
                    try:
                        value = pic[ball].mean()
                    except Exception as e:
                        print("CONTINUING", e)
                        continue
                elif radmode=="median":
                    try:
                        value = pic[ball].median()
                    except Exception as e:
                        print("CONTINUING", e)
                        continue
                rads.append((rad, value))
            
            if valid:
                rads.sort(key=lambda x: x[1])
                rad = rads[-1][0]

    #print(type(rad), rad)
    return np.array([x,1-y,rad/8])

def collect_actions():
    train, dev, test = phyre.get_fold('ball_within_template', 0)
    tasks = list(train + dev + test)
    #print(len(tasks))
    #bad = tasks.index("00024:440")
    tasks.remove("00024:440")
    sim = phyre.initialize_simulator(tasks, 'ball')
    actions = []
    cache = phyre.get_default_100k_cache('ball')
    cached_actions = cache.action_array

                
    for ti, task in enumerate(tasks):
        count = 0
        solutions = cached_actions[cache.load_simulation_states(task)==1]
        idxs = np.arange(len(solutions))
        selection = np.random.choice(idxs, 10)
        actions.extend(list(solutions[selection]))
        #print(max(idxs), selection)
        #print(list(solutions[selection]), task, len(solutions))

        """
        while True:
            try:
                action = random.choice(solutions)
            except:
                print("empty")
                action = sim.sample(ti)
            #res = sim.simulate_action(ti, action)
            #print("collected", count, "actions for", task, end='\r')
            #if res.status.is_solved():
            actions.append(action)
            count +=1
            if count >=10:
                break
        """

    actions = np.array(actions)
    #print(actions.shape)
    plt.hist(actions[:,2], bins=100)
    plt.savefig("action-hist.png")
    np.save("data/sample-actions.npy", actions)

def pic_hist_to_action(pic, r_fac=3):
    # thresholding
    pic = pic*(pic>0.2)
    # columns part of ball
    cols = [idx for (idx,val) in enumerate(np.sum(pic, axis=0)) if val>2]
    start, end = min(cols), max(cols)
    x = (start+end)/2
    x /= pic.shape[1]
    # rows part of ball
    rows = [idx for (idx,val) in enumerate(np.sum(pic, axis=1)) if val>2]
    start, end = min(rows), max(rows)
    y = (start+end)/2
    y /= pic.shape[0]
    # radius
    r = np.sqrt(pic.sum()/(3.141592*pic.shape[0]**2))
    r = 0.1
    return x, y, r

def scenes_to_channels(X, size=(32,32)):
    x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, scene in enumerate(X):
        channels = [(scene==j).astype(float) for j in range(1,8)]
        x[i] = np.flip(np.stack([(cv2.resize(c, size, cv2.INTER_MAX)>0).astype(float) for c in channels]), axis=1)
    return x

def rollouts_to_specific_paths(batch, channel, size=(32,32), gamma=1):
    trajectory = np.zeros((len(batch), size[0], size[1]))
    for j, r in enumerate(batch):
        path = np.zeros((len(r), size[0], size[1]))
        alpha = 1
        for i, scene in enumerate(r):
            chan = (scene==channel).astype(float)
            path[i] = alpha*(cv2.resize(chan, size, cv2.INTER_MAX)>0).astype(float)
            alpha *= gamma
        path = np.flip(path, axis=1)
        base = np.max(path, axis=0).astype(float)
        trajectory[j] = base
    return trajectory

def extract_individual_auccess(path):

    with open(path+"/auccess-dict.json") as fp:
        dic = json.load(fp)

    w_res = dict((i,[]) for i in range(25))
    c_res = dict((i,[]) for i in range(25))
    keys = dic.keys()
    w_keys = [k for k in keys if k.__contains__("within")]
    c_keys = [k for k in keys if k.__contains__("cross")]
    print(c_keys)

    for k in w_keys:
        templ = int(k.split('_')[4][-2:])
        w_res[templ].append(dic[k])
    print(w_res)

    for k in c_keys:
        templ = int(k.split('_')[4][-2:])
        c_res[templ].append(dic[k])
    print(c_res)
    
    with open(path+"/average-auccess-horizontal.txt", "w") as fp:
        fp.write("within cross\n")
        within = [sum(templ)/len(templ) for templ in [w_res[i] for i in range(25)]]
        cross = [sum(templ)/len(templ) for templ in [(c_res[i] or [0]) for i in range(25)]]
        #fp.writelines([f"{('0000'+str(i))[-5:]} {w} {c}\n" for i,(w,c) in enumerate(zip(within, cross))])
        fp.writelines([str(round(item, 2))[-3:]+' & ' for item in within]+['\n'])
        fp.writelines([str(round(item, 2))[-3:]+' & ' for item in cross]+['\n'])
        fp.write(f"average {sum(within)/len(within)} {sum(cross)/(len(cross)-1)}")

def collect_traj_lookup(tasks, save_path, number_per_task, show=False, stride=10):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    print("Amount per task", number_per_task)

    keys = []
    values = []

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task)==1]
            print(f"collecting {n_collected+1} interactions from {task} with {tries} tries", end = end_char)
            if len(action)==0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                need_featurized_objects=True, stride=1)
            while res.status.is_invalid():
                action = np.random.rand(3)
                res = sim.simulate_action(idx, action,
                    need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                #print(res.images.shape)
                #print(len(res.bitmap_seq))
                #print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                #print(idx1, idx2)
                #print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx,red_idx]])/2
                for i,m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        pos = res.featurized_objects.features[i,[green_idx,red_idx],:2]
                        dist = np.linalg.norm(pos[1]-pos[0])
                        #print(dist, target_dist)
                        if not dist<target_dist+0.005:
                            continue

                        red_radius = res.featurized_objects.diameters[red_idx]*4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i, pos[0], action_at_interaction, target_dist)

                return (False, 0, (0,0), 0, 0)

            contact, i_step, green_pos, red_pos, summed_radii = check_contact(res)
            if  contact:
                tries = 0

                step_n = 10
                # check whether contact happend too early
                if i_step-step_n < 0:
                    continue

                try:
                    green_idx = res.featurized_objects.colors.index('GREEN')
                    red_idx = res.featurized_objects.colors.index('RED')
                    green_minus, _ = res.featurized_objects.features[i_step-stride,[green_idx,red_idx],:2]
                    green_zero, _ = res.featurized_objects.features[i_step,[green_idx,red_idx],:2]
                    green_plus, _ = res.featurized_objects.features[i_step+stride,[green_idx,red_idx],:2]
                    green_key, _ = green_minus-green_zero, 0
                    green_value, _ = green_zero-green_plus, 0
                    keys.append((green_key[0], green_key[1]))
                    values.append((green_value[0], green_value[1]))
                except:
                    continue
                
                n_collected += 1

            if tries>max_tries:
                break

    keys = np.round(256*np.array(keys))
    k_x_max = keys[np.argmax(np.abs(keys[:,0])),0]
    k_y_max = keys[np.argmax(np.abs(keys[:,1])),1]
    """keys[:,0] /= k_x_max/5
    keys[:,1] /= k_y_max/5
    k_x_max = np.max(np.abs(keys[:,0]))
    k_y_max = np.max(np.abs(keys[:,1]))"""
    values = np.round(256*np.array(values))
    v_x_max = values[np.argmax(np.abs(values[:,0])), 0]
    v_y_max =  values[np.argmax(np.abs(values[:,1])), 1]
    """values[:,0] /= v_x_max/5
    values[:,1] /= v_y_max/5
    v_x_max = np.max(np.abs(values[:,0]))
    v_y_max = np.max(np.abs(values[:,1]))"""

    table = dict()
    for i in range(len(keys)):
        k = tuple(keys[i])
        v = tuple(values[i])
        if k in table:   
            table[k][v] = table[k][v] + 1 if v in table[k] else 1
        else:
            table[k] = {v:1}


    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/lookup.pickle', 'wb') as fp:
        pickle.dump(table, fp)
    print(f"FINISH collecting trajectory lookup!")
    return keys, values, k_x_max, k_y_max, v_x_max, v_y_max, table

def visualize_actions_from_cache(amount):
    cache = phyre.get_default_100k_cache("ball")
    actions = cache.action_array[:amount]
    plt.scatter(actions[:,0], actions[:,1], alpha=0.3, s=1000*actions[:,2], c=actions[:,2])
    plt.show()

def print_folds():
    eval_setup = 'ball_within_template'
    for fold_id in range(10):
        #print(phyre.get_fold(eval_setup, fold_id)[0][:10])
        print(phyre.get_fold(eval_setup, fold_id)[0][:10] 
            == phyre.get_fold(eval_setup, fold_id)[0][:10])

def get_auccess_for_n_tries(n):
    eva = phyre.Evaluator(['00000:000'])
    for _ in range(n-1):
        eva.maybe_log_attempt(0, phyre.SimulationStatus.NOT_SOLVED)
    for _ in range(101-n):
        eva.maybe_log_attempt(0, phyre.SimulationStatus.SOLVED)
    return eva.compute_all_metrics()

def get_auccess_for_n_tries_first_only(n):
    eva = phyre.Evaluator(['00000:000'])
    for i in range(1,101):
        if n==i:
            eva.maybe_log_attempt(0, phyre.SimulationStatus.SOLVED)
        else:
            eva.maybe_log_attempt(0, phyre.SimulationStatus.NOT_SOLVED)
    return eva.get_auccess()

def add_dijkstra_to_data(path):
    if os.path.exists(path+"/data.pickle") and os.path.exists(path+"/index.pickle"):
        with open(path+'/data.pickle', 'rb') as fp:
            data = pickle.load(fp)
            X = T.tensor(data).float()
    else:
        print("Path not found")
    
    for scene in X:
        red = T.stack((X[:,0],X[:,0]*0,X[:,0]*0), dim=-1)
        green = T.stack((X[:,1],X[:,0]*0,X[:,0]*0), dim=-1)
        blues = T.stack((X[:,2],X[:,0]*0,X[:,0]*0), dim=-1)
        blued = T.stack((X[:,3],X[:,0]*0,X[:,0]*0), dim=-1)
        grey = T.stack((X[:,4],X[:,4],X[:,4]), dim=-1)
        black = T.stack((X[:,5],X[:,0]*0,X[:,0]*0), dim=-1)

def create_eval_overview(paths, wid = 64):
    inspect = "result/flownet/inspect/"
    train = "result/flownet/training/"
    fold = 0
    tasks_img = None
    for setup in ['within', 'cross']:
        comb_viz = []
        names = []
        aucc_names = []
        res = []
        losses = []
        for path in paths:
            base = inspect+path+"/default/"
            mid = f"ball_{setup}_template_fold_{fold}/"
            end = "eval-viz-tensor.pt"
            
            tstart = train+path+"/default/"
            auccess = f"ball_{setup}_template_{fold}-auccess.txt"
            loss_pic = f"loss_plot.png"
            loss_txt = f"loss.txt"
            
            try:
                #train:
                tmp_res = np.loadtxt(tstart+auccess, usecols=2)
                if tmp_res.shape[0] == 4:
                    res.append(tmp_res)
                    aucc_names.append(path)
                else:
                    continue
                    print(res[-1].shape)

                with open(tstart+loss_txt, "r") as fp:
                    data = fp.readlines()
                data = [[float(item) for item in (line.replace("tensor([", "").replace("])\n", "")).split(",")] for line in data]
                log = np.array(data)
                #print(log.shape)

                loss_labels = ['combined', 'base', 'target', 'act-path', 'act-ball']
                for i in range(5):
                    plt.plot(np.mean(log[:,i].reshape(-1,10), axis=1), label=loss_labels[i])
                plt.legend()
                plt.title(aucc_names[-1])
                plt.ylim(0,0.25)
                plt.grid()
                plt.savefig(tstart+loss_pic)
                plt.close()
                loss_plot = cv2.imread(tstart+loss_pic)

                #print(loss_plot.shape)
                losses.append(loss_plot)

                #inspect:
                viz = T.load(base+mid+end)
                if tasks_img is None:
                    pic = "poch_0_0_diff.png"
                    img = cv2.imread(base+mid+pic)
                    tasks_img = cv2.imread(base+mid+pic)[:,:40]
                comb_viz.append(viz[:,7,None])
                #print(base+mid+end, comb_viz[-1].shape)
                names.append(path)


            except Exception as e:
                print(e)

        # save auccess file:
        np.savetxt(f"{setup}-aucces.csv", np.stack(res, axis=0).T, header=",".join(aucc_names), delimiter=",", fmt='%1.7f')

        # save loss plots:
        loss_plots = Image.fromarray(np.concatenate(losses, axis=1))
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 20)
        draw = ImageDraw.Draw(loss_plots)
        for i,x in enumerate(range(0,loss_plots.width, int(loss_plots.width/len(names)))):
            draw.text((10+x, 5), names[i], fill= (0,0,0), font=font)

        loss_plots.save(f"{setup}-losses.png")


        if False:
            comb_viz = T.cat(comb_viz, dim=1)
            viz = np.array(vis_batch(comb_viz, "./", f"{setup}-inspects", text=names, save=False))
            height = viz.shape[0]
            tasks_img = tasks_img[-height:]
            final_viz = np.concatenate((tasks_img, viz), axis=1)
            plt.imsave(f"{setup}-inspects.png", final_viz, dpi=1000)

def check_results(path, run):
    fullpath = "./result/flownet/result/"+path+"/"+run+"/results.json"
    with open(fullpath, "r") as fp:
        dic = json.load(fp)    
    res = np.zeros((2,25,10,2,30))
    for si, setup in enumerate(["within", "cross"]):
        sfiltered = [k for k in dic.keys() if k.__contains__(setup)]
        for ei, epoch in enumerate([f"e{i}-" for i in range(30)]):
            efiltered = [k for k in sfiltered if k.__contains__(epoch)]
            for ti, template in enumerate([f"0000{i}"[-5:] for i in range(25)]):
                tfiltered = [k for k in efiltered if k.__contains__(template)]
                for fi, fold in enumerate([f"f{i}-" for i in range(10)]):
                    ffiltered = [k for k in tfiltered if k.__contains__(fold)]
                    if ffiltered:
                        assert len(ffiltered)==1, f"Should only be one key left {ffiltered}"
                        auc, perc = dic[ffiltered[0]]
                        res[si,ti,fi,0,ei] = auc
                        res[si,ti,fi,1,ei] = perc
    
    #print("within epoch0 comb per fold", res[0,:,:,:,0].sum(dim=-1)/res[0,:,:,:,0].count_nonzero(dim=-1))
    print("mean for templates over all folds with best epoch value", np.mean(np.max(res[0],axis=3), axis=1))
    print("mean for folds over all templates with best epoch value", np.mean(np.max(res[0],axis=3), axis=0))
    print("mean for folds and templates with best epoch value", np.mean(np.mean(np.max(res[0],axis=3), axis=0),axis=0))
    print("mean for templates over all folds with best epoch value", np.sum(np.max(res[1],axis=3), axis=1)/np.sum(np.max(res[1],axis=3)>0, axis=1))
    print("mean for folds over all templates with best epoch value", np.sum(np.max(res[1],axis=3), axis=0)/np.sum(np.max(res[1],axis=3)>0, axis=0))
    print("mean for folds and templates with best epoch value", np.sum(np.sum(np.max(res[1],axis=3), axis=0)/ np.sum(np.max(res[1],axis=3)>0, axis=0),axis=0)/np.sum(0<np.sum(np.max(res[1],axis=3), axis=0)/np.sum(np.max(res[1],axis=3)>0, axis=0),axis=0))

    
    #print("within template-mean", res[0].mean(dim=1))
    #print("within auc", res[0,:,:,0])
    #print("within perc", res[0,:,:,1])


def get_arr_from_fig(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256))
    return img/255

if __name__ == "__main__":
    #visualize_actions_from_cache(1000)
    #print(get_auccess_for_n_tries(10))
    # Collecting trajectory lookup
    #collect_actions()
    #exit()
    if True:
        check_results("GEN-COMB", "solve")
        exit()

    if True:
        print(get_auccess_for_n_tries(50))
        exit()

    if True:
        eval_setup = 'ball_within_template'
        for fold_id in range(9):
            train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
            all_tasks = set(train_ids+dev_ids+test_ids)
            train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
            all_tasks2 = set(train_ids+dev_ids+test_ids)

            print(all_tasks == all_tasks2)
        exit()



    cache = phyre.get_default_100k_cache('ball')
    cactions = cache.action_array
    for setup in ["ball_within_template", "ball_cross_template"]:
        done = []
        train_ids, dev_ids, test_ids = phyre.get_fold(setup, 0)
        tasks = train_ids + dev_ids +test_ids
        sim = phyre.initialize_simulator(tasks, 'ball')
        for ti, task in enumerate(tasks):
            #if task[:5] in done:
            #    continue
            #done.append(task[:5])
            print(setup, task)
            actions = cactions[cache.load_simulation_states(task)==1]
            fig = plt.figure(figsize=(25,4))
            #figs = []
            fs = (10,10)

            # INIT SCENE
            ax = fig.add_subplot(161)
            obs = phyre.observations_to_uint8_rgb(sim.initial_scenes[ti])
            plt.imshow(obs)

            # Overlay
            ax = fig.add_subplot(162)
            plt.imshow(obs)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("y")
            #ax.axes.set_ylim(bottom=0, top=1) 
            #ax.axes.set_xlim(left=0, right=1)
            #print(actions[:2])
            ax.set_aspect('equal')
            ax.scatter(255*actions[:,0], 255-255*actions[:,1], s=1, c=actions[:,2], alpha=0.5)
            cbaxes = fig.add_axes([1/3 +0.0355, 0.15, 0.005, 0.69]) 
            cb = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=cm.jet), cax=cbaxes, shrink=0.5, ticks=[])
            #cb = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=cm.jet), cax=cbaxes, shrink=0.5,  ticks=[0,0.2,0.4,0.6,0.8,1])
            #plt.set_clim(0, 1)
        

            # From radius
            ax = fig.add_subplot(163)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("y")
            ax.set_aspect('equal')
            ax.axes.set_ylim(bottom=0, top=1) 
            ax.axes.set_xlim(left=0, right=1)
            ax.scatter(actions[:,0], actions[:,1], s=1)

            # From x
            ax = fig.add_subplot(164)
            ax.axes.set_xlabel("r")
            ax.axes.set_ylabel("y")
            ax.set_aspect('equal')
            ax.axes.set_ylim(bottom=0, top=1) 
            ax.axes.set_xlim(left=0, right=1)
            ax.scatter(actions[:,2], actions[:,1], s=1)

            # From y
            ax = fig.add_subplot(165)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("r")
            ax.set_aspect('equal')
            ax.axes.set_ylim(bottom=0, top=1) 
            ax.axes.set_xlim(left=0, right=1)
            ax.scatter(actions[:,0], actions[:,2], s=1)
            
            # 3D
            ax = fig.add_subplot(166, projection='3d')
            #ax = fig.gca()
            ax.axes.set_xlim3d(left=0, right=1) 
            ax.axes.set_ylim3d(bottom=0, top=1) 
            ax.axes.set_zlim3d(bottom=0, top=1)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("radius") 
            ax.axes.set_zlabel("y")
            #ax.set_aspect('equal')
            ax.scatter(actions[:,0],  actions[:,2], actions[:,1], s=1)

            #plt.tight_layout()
            plt.savefig(f"actionspace/{setup}-{task}-space.png")
            plt.close()
    exit()

    cache = phyre.get_default_100k_cache('ball')
    cactions = cache.action_array
    for setup in ["ball_within_template", "ball_cross_template"]:
        done = []
        train_ids, dev_ids, test_ids = phyre.get_fold(setup, 0)
        tasks = train_ids + dev_ids +dev_ids
        sim = phyre.initialize_simulator(tasks, 'ball')
        for ti, task in enumerate(tasks):
            #if task[:5] in done:
            #    continue
            #done.append(task[:5])
            print(setup, task)
            actions = cactions[cache.load_simulation_states(task)==1]
            #fig = plt.figure(figsize=(1,1))
            figs = []
            fs = (10,10)

            # INIT SCENE
            fig = plt.figure(figsize=fs)
            ax = fig.add_subplot(111)
            obs = phyre.observations_to_uint8_rgb(sim.initial_scenes[ti])
            plt.imshow(obs/255)
            arr = get_arr_from_fig(fig)
            figs.append(arr)

            # 3D
            fig = plt.figure(figsize=fs)
            ax = fig.add_subplot(111, projection='3d')
            #ax = fig.gca()
            ax.axes.set_xlim3d(left=0, right=1) 
            ax.axes.set_ylim3d(bottom=0, top=1) 
            ax.axes.set_zlim3d(bottom=0, top=1)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("radius") 
            ax.axes.set_zlabel("y")
            ax.scatter(actions[:,0],  actions[:,2], actions[:,1], s=0.6)
            arr = get_arr_from_fig(fig)
            figs.append(arr)

            # Overlay
            fig = plt.figure(figsize=(fs[0], fs[1]*1.5))
            ax = fig.add_subplot(111)
            plt.imshow(obs)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("y")
            #ax.axes.set_ylim(bottom=0, top=1) 
            #ax.axes.set_xlim(left=0, right=1)
            #print(actions[:2])
            ax.scatter(1*actions[:,0], 1-1*actions[:,1], s=0.6, c=actions[:,2], alpha=0.5)
            plt.colorbar(cm.ScalarMappable(norm=colors.NoNorm(), cmap=cm.jet), shrink=0.5,  ticks=[0,0.2,0.4,0.6,0.8,1])
            #plt.set_clim(0, 1)
            arr = get_arr_from_fig(fig)
            figs.append(arr)
            # From radius
            fig = plt.figure(figsize=fs)
            ax = fig.add_subplot(111)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("y")
            ax.axes.set_ylim(bottom=0, top=1) 
            ax.axes.set_xlim(left=0, right=1)
            ax.scatter(actions[:,0], actions[:,1], s=0.6)
            arr = get_arr_from_fig(fig)
            figs.append(arr)

            # From x
            fig = plt.figure(figsize=fs)
            ax = fig.add_subplot(111)
            ax.axes.set_xlabel("r")
            ax.axes.set_ylabel("y")
            ax.axes.set_ylim(bottom=0, top=1) 
            ax.axes.set_xlim(left=0, right=1)
            ax.scatter(actions[:,2], actions[:,1], s=0.6)
            arr = get_arr_from_fig(fig)
            figs.append(arr)

            # From y
            fig = plt.figure(figsize=fs)
            ax = fig.add_subplot(111)
            ax.axes.set_xlabel("x")
            ax.axes.set_ylabel("r")
            ax.axes.set_ylim(bottom=0, top=1) 
            ax.axes.set_xlim(left=0, right=1)
            ax.scatter(actions[:,0], actions[:,2], s=0.6)
            arr = get_arr_from_fig(fig)
            figs.append(arr)
            
            #plt.tight_layout()
            combined = np.concatenate(figs, axis=1)
            plt.imsave(f"actionspace/{setup}-{task}-space.png", combined)
            plt.close()
    exit()

    if False:
        pic = draw_ball(32,0.5,0.2,0.3) + draw_ball(32,0.5,0.5,0.1)
        #print(grow_action_vector(pic, check_border=True, mask=draw_ball(32,0.5,0.5,0.1), show=True))
        cache = phyre.get_default_100k_cache('ball')
        print(sample_action_vector(pic, cache, radmode="random", check_border=True, mask=draw_ball(32,0.5,0.5,0.1)))
        print(sample_action_vector(pic, cache, radmode="mean", check_border=True, mask=draw_ball(32,0.5,0.5,0.1)))
        print(sample_action_vector(pic, cache, radmode="median", check_border=True, mask=draw_ball(32,0.5,0.5,0.1)))
        print(sample_action_vector(pic, cache, uniform=True, radmode="random", check_border=True, mask=draw_ball(32,0.5,0.5,0.1)))
        print(sample_action_vector(pic, cache, uniform=True, radmode="mean", check_border=True, mask=draw_ball(32,0.5,0.5,0.1)))
        print(sample_action_vector(pic, cache, uniform=True, radmode="median", check_border=True, mask=draw_ball(32,0.5,0.5,0.1)))
        exit()

    if False:
        paths = ['VS-uni-rand','VS-uni-mean', 'VS-uni-median', 'VS-rad-rand','VS-rad-median','VS-rad-mean', 'VS-base','VS-w128', 'VS-bs16', 'VS-bs64', 'VS-dijkstra', 'VS-direct', 
            'VS-dropout', 'VS-lr01', 'VS-lr03', 'VS-lr3', 'VS-neck','VS-nobase','VS-nper32',
            'VS-sched', 'VS-task-cons', 'VS-templ-cons', 'VS-withbase', 'VS-x2', 'VS-x4', 'VS-x05']
        create_eval_overview(paths)
        exit()

    shrink_data("./data")
    exit()

    extract_individual_auccess("./result/solver/result/GEN-64-20e/individ-10")
    exit()

    for n in range(1,20):
        print(get_auccess_for_n_tries_first_only(n))
    exit()

    make_visuals()
    exit()
    #exit()
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids+dev_ids+test_ids
    template13_tasks = [t for t in all_tasks if t.startswith('00013:')]
    template2_tasks = [t for t in all_tasks if t.startswith('00002:')]
    print(template2_tasks)
    #collect_specific_channel_paths(f'./data/template13_action_paths_10x', template13_tasks, 0)
    keys, values, kxm, kym, vxm, vym, table = collect_traj_lookup(all_tasks, 'result/traj_lookup/all_tasks', 10, stride=10)
    print(keys)
    print(values)
    print(kxm, kym, vxm, vym)
    print(table)
    
