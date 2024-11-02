import os
import numpy as np
import rosbag
import cv2

# joint_states[0] ~ elbow_joint ~ joint3
# joint_states[1] ~ gripper ~ gripper
# joint_states[2] ~ shoulder_lift_joint ~ joint2
# joint_states[3] ~ shoulder_pan_joint ~ joint1
# joint_states[4] ~ wrist_1_joint ~ joint4
# joint_states[5] ~ wrist_2_joint ~ joint5
# joint_states[6] ~ wrist_3_joint ~ joint6

def reorder_joint_states(joint_state):
    # 调整关节状态的顺序，例如：
    # 这里假设需要的顺序是：[gripper, joint1, joint2, joint3, joint4, joint5, joint6]
    # 请根据实际情况调整索引顺序
    reordered = [
        joint_state[1],  # gripper
        joint_state[3],  # joint1
        joint_state[2],  # joint2
        joint_state[0],  # joint3
        joint_state[4],  # joint4
        joint_state[5],  # joint5
        joint_state[6]   # joint6
    ]
    return reordered

def process_rosbag(bag_file, output_dir, start_index, task=5001, user=1):
    os.makedirs(output_dir, exist_ok=True)

    timestamps = []
    images = []
    joint_states = []
    joint_timestamps = []

    # 读取rosbag文件
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == '/camera/rgb/image_raw':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                images.append(img)
                timestamps.append(t.to_sec())
            elif topic == '/joint_states':
                joint_states.append(msg.position)
                joint_timestamps.append(t.to_sec())

    # 转换为numpy数组
    timestamps = np.array(timestamps)  # 图像时间戳
    joint_timestamps = np.array(joint_timestamps)  # 关节时间戳
    joint_states = np.array(joint_states)

    # 对齐图像和关节状态时间戳
    aligned_data = []
    joint_index = 0

    for img_time in timestamps:
        # 找到最近的关节状态
        while (joint_index < len(joint_states) - 1 and 
               abs(joint_timestamps[joint_index + 1] - img_time) < 
               abs(joint_timestamps[joint_index] - img_time)):
            joint_index += 1
        
        # 确保时间戳对齐
        if joint_index < len(joint_states):
            reordered_joint_state = reorder_joint_states(joint_states[joint_index])
            aligned_data.append((img_time, reordered_joint_state))

    aligned_data = np.array(aligned_data, dtype=object)

    # 准备npy数据
    data = np.zeros((len(aligned_data), 10))
    data[:, 0] = (aligned_data[:, 0] * 1000).astype(int)  # 时间戳转换为毫秒并保留整型
    data[:, 1] = task  # task
    data[:, 2] = user  # user

    # 判别第四列
    joint_values = np.array([entry[1] for entry in aligned_data])
    data[:, 3] = np.where(joint_values[:, 0] > 0.01, 1, 0)  # 第四列判断
    
    # 其余关节状态
    data[:, 4:] = joint_values[:, 1:]  # 其余关节状

    # 生成文件夹和文件名
    folder_name = str(start_index).zfill(5)
    image_folder = os.path.join(output_base_dir, folder_name)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    npy_filename = os.path.join(output_dir, f"{folder_name}-joints.npy")
    txt_filename = os.path.join(output_dir, f"{folder_name}-joints.txt")

    # 保存npy文件
    np.save(npy_filename, data)

    # 保存txt文件
    with open(txt_filename, 'w') as f:
        f.write("timestamp,task,user,gripper,joint1,joint2,joint3,joint4,joint5,joint6\n")
        np.savetxt(f, data, fmt='%d,%d,%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f', delimiter=', ')

    # 保存图像
    for i, img in enumerate(images):
        if i < len(aligned_data):  # 确保索引不超出
            img_filename = os.path.join(image_folder, f"{int(aligned_data[i][0] * 1000)}.png")
            image_array = np.frombuffer(img, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            cv2.imwrite(img_filename, img)

if __name__ == '__main__':
    # 设置rosbag文件路径和开始索引
    start_index = 201  # 自定义开始数字
    bag_folder = '/home/wulinger/dataset_process_ws/black_hammer/bagfile'  # rosbag文件夹路径
    output_base_dir = '/media/wulinger/SSD1/origin_dataset/pick_up/red_hammer/dataset'  # 数据集输出路径
    rosbag_files = [os.path.join(bag_folder, f"{str(i).zfill(4)}.bag") for i in range(start_index, start_index + 1)]
    
    for i, bag_file in enumerate(rosbag_files):
        if os.path.exists(bag_file):
            process_rosbag(bag_file, output_base_dir, start_index + i)
            print(i)
        else:
            print(f"文件 {bag_file} 不存在，跳过处理。")
