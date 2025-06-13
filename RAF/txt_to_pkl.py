# command: CUDA_VISIBLE_DEVICES=0,1,2,3 python txt_to_pkl.py
# run time ~ 30s, generate pkl file
import torch
import clip
from torch.nn.parallel import DataParallel
import math
from tqdm import tqdm
import pickle
import subprocess
import os

def GeneratePkl():
    if torch.cuda.device_count() > 1:
        use_multi_gpu = True
    else:
        use_multi_gpu = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    if use_multi_gpu:
        model = DataParallel(model)

    data = torch.load('v3det_gpt_noun_chunk_coco_strict.pkl')
    ori_vocab = data['all_noun_chunks']['names']

    # Load data (relative path in disk2)
    with open("../OADP/OADP/LLM_api/output_synonym.txt", "r", encoding="utf-8") as f:
        our_vocab = [line.strip() for line in f]

    vocabulary = ori_vocab + our_vocab

    # Process text in batches to avoid CUDA out of mem
    batch_size = 256
    num_items = len(vocabulary)
    num_batches = math.ceil(num_items / batch_size)

    text_features_list = []

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_items)
        batch_vocabulary = vocabulary[start_idx:end_idx]
        
        text_token = clip.tokenize(batch_vocabulary).to(device)
        
        with torch.no_grad():
            batch_text_features = model.module.encode_text(text_token) if use_multi_gpu else model.encode_text(text_token)
            text_features_list.append(batch_text_features.cpu())  # Move to CPU to save GPU memory

    # Combine all batches
    text_features = torch.cat(text_features_list, dim=0)
    text_features = text_features.to(device)  # Move back to GPU for final computations

    # save as pkl
    data = {
        'all_noun_chunks':{
            'text_features': text_features,
            'names': vocabulary
        }
    }
    torch.save(data, 'v3det_noun_chunk_amber.pkl')    # modify the output file name (v3det_gpt_noun_chunk_coco_strict.pkl)
    print("finish generating pkl!")


# ============================== run raf.py (~ 2 hour ) ============================== #
def Raf():
    # env init
    cuda_devices = "0,1,2,3"  # Set fixed CUDA devices
    num_gpus = len(cuda_devices.split(','))

    # command for raf.py
    raf_command = [
        'torchrun',
        '--nproc_per_node=' + str(num_gpus),
        '--master_port=29501',
        'raf.py',
        '--dataset', 'coco',
        '--work_dir', 'output/raf_coco',
        '--concept_pkl_path', 'v3det_noun_chunk_amber.pkl',   # replace with the pkl file generated above
        '--oake_file_path', 'clip_region/coco_oake_info_strict.pkl'
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_devices

    print("start to run raf.py...")
    try:
        # Run the command
        subprocess.run(' '.join(raf_command), shell=True, check=True)
        print("raf.py completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running raf.py: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# ============================== run test.py (~ 1.5 hour) ============================== #
def Test():
    # command for test.py
    test_command = [
        'torchrun',
        '--nproc_per_node=1',
        '--master_port=29501',
        '-m', 'oadp.dp.test',
        './configs/dp/ralf/raf/coco_raf.py',            # modify this file to replace with our raf model
        'work_dirs/coco_ral/iter_32000_OADP_repo.pth'
    ]

    # move to OADP/
    working_dir = '../OADP/OADP'  # Update this with the actual absolute path
    env = os.environ.copy()
    env['PYTHONPATH'] = working_dir + ':' + env.get('PYTHONPATH', '')

    print("start to run test.py...")

    try:
        # Run the command
        subprocess.run(test_command, env=env, cwd=working_dir, check=True)
        print("test.py completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running test.py: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while running test.py: {e}")


if __name__ == "__main__":
    print("start generate plk...")
    GeneratePkl()

    print("start raf.py...")
    Raf()
    
    print("start test.py...")
    Test()

    print("all task finish!")
