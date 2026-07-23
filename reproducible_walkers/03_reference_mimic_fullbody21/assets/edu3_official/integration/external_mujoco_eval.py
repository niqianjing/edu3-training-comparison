import argparse, json, os
import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

USD_ORDER = [
    'left_thigh_pitch_joint','right_thigh_pitch_joint','torso_joint',
    'left_thigh_roll_joint','right_thigh_roll_joint','left_arm_pitch_joint',
    'right_arm_pitch_joint','left_thigh_yaw_joint','right_thigh_yaw_joint',
    'left_arm_roll_joint','right_arm_roll_joint','left_knee_joint','right_knee_joint',
    'left_arm_yaw_joint','right_arm_yaw_joint','left_ankle_pitch_joint',
    'right_ankle_pitch_joint','left_elbow_pitch_joint','right_elbow_pitch_joint',
    'left_ankle_roll_joint','right_ankle_roll_joint']
DEFAULT = {
    'left_thigh_pitch_joint':-0.10,'right_thigh_pitch_joint':0.10,
    'left_knee_joint':0.30,'right_knee_joint':-0.30,
    'left_ankle_pitch_joint':-0.20,'right_ankle_pitch_joint':0.20,
    'left_arm_pitch_joint':-0.15,'right_arm_pitch_joint':0.15,
    'left_arm_roll_joint':0.05,'right_arm_roll_joint':-0.05,
    'left_elbow_pitch_joint':0.60,'right_elbow_pitch_joint':-0.60}

def group(name):
    if 'thigh_pitch' in name or 'knee' in name:
        return 100.0,3.0,50.0,0.51,0.0432
    if 'ankle' in name:
        return 40.0,1.5,40.0,0.146,0.0306
    if 'thigh_roll' in name or 'thigh_yaw' in name or name == 'torso_joint':
        return 80.0,2.5,20.0,0.146,0.0306
    return 30.0,1.2,10.0,0.146,0.0306

def rpy_wxyz(q):
    return R.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mjcf',required=True)
    ap.add_argument('--policy',required=True)
    ap.add_argument('--out_prefix',required=True)
    ap.add_argument('--vx',type=float,default=0.30)
    ap.add_argument('--seconds',type=float,default=8.0)
    args=ap.parse_args()

    model=mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep=0.001
    model.dof_damping[:]=0.0
    model.dof_frictionloss[:]=0.0
    model.actuator_ctrllimited[:]=0
    model.actuator_forcelimited[:]=0
    data=mujoco.MjData(model)

    joint_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in USD_ORDER])
    if np.any(joint_ids<0): raise RuntimeError('missing semantic joint')
    qadr=model.jnt_qposadr[joint_ids].astype(int)
    dadr=model.jnt_dofadr[joint_ids].astype(int)
    act_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n+'_motor') for n in USD_ORDER])
    if np.any(act_ids<0): raise RuntimeError('missing semantic actuator')
    base_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    root_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'root')
    root_qadr=int(model.jnt_qposadr[root_id])

    default=np.array([DEFAULT.get(n,0.0) for n in USD_ORDER])
    kp=np.array([group(n)[0] for n in USD_ORDER]); kd=np.array([group(n)[1] for n in USD_ORDER])
    peak=np.array([group(n)[2] for n in USD_ORDER]); tc=np.array([group(n)[3] for n in USD_ORDER])
    visc=np.array([group(n)[4] for n in USD_ORDER])
    low=model.jnt_range[joint_ids,0].copy(); high=model.jnt_range[joint_ids,1].copy()

    mujoco.mj_resetData(model,data)
    data.qpos[root_qadr:root_qadr+3]=[0,0,0.40]
    data.qpos[root_qadr+3:root_qadr+7]=[1,0,0,0]
    data.qpos[qadr]=default
    mujoco.mj_forward(model,data)

    policy=torch.jit.load(args.policy,map_location='cpu'); policy.eval()
    action=np.zeros(21); target=default.copy(); hist=None
    logs={k:[] for k in ['pos','quat','q','dq','action','target','motor','friction','applied','contact_count']}
    renderer=mujoco.Renderer(model,height=480,width=640)
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth=135; cam.elevation=-10; cam.distance=1.25
    writer=cv2.VideoWriter(args.out_prefix+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),50.0,(640,480))

    total=int(args.seconds/model.opt.timestep); decimation=20
    fell=False; first_fall_time=None
    for step in range(total):
        q=data.qpos[qadr].copy(); dq=data.qvel[dadr].copy()
        quat=data.xquat[base_id].copy()
        if step%decimation==0:
            vel6=np.zeros(6); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base_id,vel6,1)
            omega=vel6[:3]
            grav=R.from_quat([quat[1],quat[2],quat[3],quat[0]]).apply([0,0,-1],inverse=True)
            frame=np.concatenate([omega,grav,[args.vx,0,0],q-default,dq,action]).astype(np.float32)
            if frame.size!=72: raise RuntimeError(frame.size)
            if hist is None: hist=np.tile(frame,(10,1))
            else: hist=np.vstack([hist[1:],frame])
            with torch.inference_mode(): raw=policy(torch.from_numpy(hist.reshape(1,-1)))[0].cpu().numpy()
            action=np.clip(raw,-1.0,1.0)
            target=default+0.25*action
            motor=np.clip(kp*(target-q)-kd*dq,-peak,peak)
            friction=tc*np.tanh(dq/0.01)+visc*dq
            applied=motor-friction
            logs['pos'].append(data.xpos[base_id].copy()); logs['quat'].append(quat.copy())
            logs['q'].append(q); logs['dq'].append(dq); logs['action'].append(action.copy())
            logs['target'].append(target.copy()); logs['motor'].append(motor.copy())
            logs['friction'].append(friction.copy()); logs['applied'].append(applied.copy())
            logs['contact_count'].append(data.ncon)
            cam.lookat[:]=data.xpos[base_id]; cam.lookat[2]=max(0.28,cam.lookat[2])
            renderer.update_scene(data,camera=cam)
            writer.write(cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
        else:
            motor=np.clip(kp*(target-q)-kd*dq,-peak,peak)
            friction=tc*np.tanh(dq/0.01)+visc*dq
            applied=motor-friction
        data.ctrl[act_ids]=applied
        mujoco.mj_step(model,data)
        rp=rpy_wxyz(data.xquat[base_id])
        bad=(data.xpos[base_id,2]<0.30 or abs(rp[0])>1.15 or abs(rp[1])>1.15)
        if bad and first_fall_time is None: first_fall_time=float(data.time); fell=True

    writer.release(); renderer.close()
    arr={k:np.asarray(v) for k,v in logs.items()}
    rpy=np.asarray([rpy_wxyz(x) for x in arr['quat']])
    rms=np.sqrt(np.mean(arr['motor']**2,axis=0)); applied_rms=np.sqrt(np.mean(arr['applied']**2,axis=0))
    sat=np.mean(np.abs(arr['motor'])>=0.99*peak,axis=0)
    span=high-low; edge=np.mean((arr['target']<=low+0.05*span)|(arr['target']>=high-0.05*span),axis=0)
    result={
      'engine':'mujoco','seed_policy':os.path.basename(os.path.dirname(os.path.dirname(args.policy))),
      'steps':int(len(arr['pos'])),'duration_s':float(data.time),'fell':bool(fell),'first_fall_time_s':first_fall_time,
      'forward_m':float(arr['pos'][-1,0]-arr['pos'][0,0]),'lateral_m':float(arr['pos'][-1,1]-arr['pos'][0,1]),
      'min_height_m':float(arr['pos'][:,2].min()),
      'yaw_change_deg':float(np.degrees(np.unwrap(rpy[:,2])[-1]-np.unwrap(rpy[:,2])[0])),
      'max_abs_roll_deg':float(np.degrees(np.abs(rpy[:,0]).max())),
      'max_abs_pitch_deg':float(np.degrees(np.abs(rpy[:,1]).max())),
      'target_oob_total':int(((arr['target']<low-1e-6)|(arr['target']>high+1e-6)).sum()),
      'friction_contract':'explicit Tc*tanh(dq/0.01)+b*dq after motor clipping; native MJCF damping/frictionloss zeroed',
      'control_delay_steps':0,
      'joint_names':USD_ORDER,
      'joints':{n:{'motor_rms_Nm':float(rms[i]),'applied_rms_Nm':float(applied_rms[i]),
                    'rms_over_probe_peak':float(rms[i]/peak[i]),'peak_saturation_rate':float(sat[i]),
                    'target_edge5_rate':float(edge[i]),'q_min_deg':float(np.degrees(arr['q'][:,i].min())),
                    'q_max_deg':float(np.degrees(arr['q'][:,i].max()))} for i,n in enumerate(USD_ORDER)}}
    np.savez(args.out_prefix+'.npz',**arr)
    with open(args.out_prefix+'.json','w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print('EXTERNAL_EDU3_MUJOCO_RESULT',json.dumps(result,ensure_ascii=False))
    print('EXTERNAL_EDU3_MUJOCO_VIDEO',args.out_prefix+'.mp4')

if __name__=='__main__': main()



