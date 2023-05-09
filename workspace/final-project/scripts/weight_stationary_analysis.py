import os
from tqdm import tqdm

base_dir = "/home/workspace/final-project/example_designs"

def get_cmd(arch, layer_name, model='VGG01'):
    #e.g.
    #timeloop-mapper arch/simple_weight_stationary.yaml arch/components/*.yaml mapper/mapper.yaml constraints/*.yaml ../../layer_shapes/AlexNet/AlexNet_layer1.yaml

    timeloop_cmd = f"timeloop-mapper " \
    f"{base_dir}/{arch}/arch/simple_weight_stationary.yaml " \
    f"{base_dir}/{arch}/arch/components/*.yaml " \
    f"{base_dir}/{arch}/mapper/mapper.yaml " \
    f"{base_dir}/{arch}/constraints/*.yaml " \
    f"{base_dir}/../layer_shapes/{model}/{layer_name}"
    
    #> /dev/null 2>&1"

    return timeloop_cmd

    



def run_arch(arch, model='VGG01', layer_num=None):
    save_dir = f"{base_dir}/../tmp_runs/{arch}"

    # save_dir = f"{base_dir}/../tmp_runs/{arch}"


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Running Timeloop ...')

    layer_names = os.listdir(f"{base_dir}/../layer_shapes/{model}")

    if layer_num != None:
        layer_names = [name for name in layer_names if f"layer{str(layer_num)}." in name]
    
    for layer in tqdm(layer_names):
        print(layer)
        cmd = get_cmd(arch, layer)

        run_save_dir = f"{save_dir}/{layer.split('.')[0]}"
        if not os.path.exists(run_save_dir):
            os.mkdir(run_save_dir)
            
        os.chdir(run_save_dir)
        os.system(cmd)
    
    print('Timeloop finished!')


if __name__ == '__main__':
    arch_list = [
        '16bit_float_simple_weight_stationary',
        '8bit_linear_weight_stationary',
        '8bit_kmeans_weight_stationary',
        '4bit_kmeans_weight_stationary',
        '2bit_kmeans_weight_stationary'     
    ]

    for a in arch_list:
        print(f"NEW ARCH: {a}")
        run_arch(a)


   