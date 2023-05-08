import os
from tqdm import tqdm

base_dir = "/home/workspace/final-project/example_designs"

def get_cmd(arch, layer_name, model='VGG01'):
    #e.g.
    #timeloop-mapper arch/simple_output_stationary.yaml arch/components/*.yaml mapper/mapper.yaml constraints/*.yaml ../../layer_shapes/AlexNet/AlexNet_layer1.yaml

    timeloop_cmd = f"timeloop-mapper " \
    f"{base_dir}/{arch}/arch/simple_output_stationary.yaml " \
    f"{base_dir}/{arch}/arch/components/*.yaml " \
    f"{base_dir}/{arch}/mapper/mapper.yaml " \
    f"{base_dir}/{arch}/constraints/*.yaml " \
    f"{base_dir}/../layer_shapes/{model}/{layer_name}"
    
    #> /dev/null 2>&1"

    return timeloop_cmd

    



def run_arch(arch, model='VGG01'):
    save_dir = f"{base_dir}/../runs/{arch}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Running Timeloop ...')

    layers = os.listdir(f"{base_dir}/../layer_shapes/{model}")
    
    for layer in tqdm(layers):
        print(layer)
        cmd = get_cmd(arch, layer)

        run_save_dir = f"{save_dir}/{layer.split('.')[0]}"
        if not os.path.exists(run_save_dir):
            os.mkdir(run_save_dir)
            
        os.chdir(run_save_dir)
        os.system(cmd)
    
    print('Timeloop finished!')


if __name__ == '__main__':
    
    # run_arch('simple_weight_stationary')

    run_arch('2bit_kmeans_output_stationary')

    # arch_list = [
    #     '8bit_linear_output_stationary',
    #     '2bit_kmeans_output_stationary',
    #     '4bit_kmeans_output_stationary',
    #     '8bit_kmeans_weight_stationary'
    # ]

    # for a in arch_list:
    #     run_arch(a)

    # cmd = get_cmd('2bit_kmeans_weight_stationary', 'VGG01_layer1.yaml')
    # print(cmd)

    # timeloop-mapper /home/workspace/final-project/example_designs/2bit_kmeans_weight_stationary/arch/simple_weight_stationary.yaml /home/workspace/final-project/example_designs/2bit_kmeans_weight_stationary/arch/components/*.yaml /home/workspace/final-project/example_designs/2bit_kmeans_weight_stationary/mapper/mapper.yaml /home/workspace/final-project/example_designs/2bit_kmeans_weight_stationary/constraints/*.yaml /home/workspace/final-project/example_designs/../layer_shapes/VGG01/VGG01_layer1.yaml > /dev/null 2>&1