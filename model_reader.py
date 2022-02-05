import torch
import sys
import os

# python3 model_reader.py "/mainfs/scratch/bm4g15/data/nalu-stable-exp/saves/test-single_layer_task_range/realnpu_op-div_oob-c_rs-linear-none_eps-0_rl-1000000-2000000_r-0.0-0-1_i--2--1_e--6--2_z-2-0.5-0.0_b128_s0_h2_z2_lr-adam-0.00500-0.0_L1T_rb-1e-09-1e-07-10000-10_rWnpuF_rg-0.pth" False False
# python3 model_reader.py "/mainfs/scratch/bm4g15/data/nalu-stable-exp/saves/single_layer_task_range/" True single_layer_task_range_weights.txt

def load_model(path):
  model = torch.load(path)
  model.eval()
  return model

def print_model_params(model):
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data)
  print()
  return param.data
      
def main():
  path = sys.argv[1]  # folder/ path for save file/s
  is_folder = sys.argv[2]  # bool - to iterate over a folder and print out the results
  #filter_parameter = sys.argv[3]  # concentrate on a single parameter, otherwise use False
  save_filename = sys.argv[3] # str - filename to save params, otherwise use False
 
  if save_filename != 'False':
    save_folder = r"/mainfs/home/bm4g15/data_backups/model_parameters/"
    save_path = os.path.join(save_folder, save_filename)
    sys.stdout = open(save_path, "w")
  
  total_runs = 0
  success_runs = 0
  if is_folder != 'False': 
    folder = path
    # print params for each saved model in the folder 
    # Assume 1st arg given is folder path
    for file in os.listdir(folder):
      total_runs += 1
      filename = os.fsdecode(file)
      print(filename)
      path = os.path.join(folder, filename)
      m = load_model(path)
      last_params = print_model_params(m)
      if torch.equal(last_params, torch.tensor([[1.,1.]])):
        success_runs += 1
    print("success rate (%): ", (success_runs/total_runs)*100)

  else:
    print(path)
    m = load_model(path)
    print_model_params(m)
  
  if save_filename != 'False':
    sys.stdout.close()
  
if __name__ == "__main__":
  main() 
  print("Script completed")
