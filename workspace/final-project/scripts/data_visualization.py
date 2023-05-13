import matplotlib.pyplot as plt
import numpy as np

run_dir = '../runs'
model = 'VGG01'


def read_data(arch, run='05-07'):
    dir = f"{run_dir}/{run}/{arch}"

    ret = list()

    for i in range(1,3):
        file_path = f"{dir}/{model}_layer{i}/timeloop-mapper.stats.txt"

        breakdown = {
            'MAC': 0,
            'DRAM': 0,
            # 'pe_spad': 0,
            'other': 0
        }

        with open(file_path, 'r') as file:
            flag = False

            for line in file:
                current = line.strip()

                if not flag:
                    if (current == 'pJ/Compute'):
                        flag = True
                    continue
                if current == '':
                    continue
             
                current = current.split()

                if current[0] == 'mac':   
                    breakdown['MAC'] = float(current[-1])
                elif current[0] == 'DRAM' and current[1] == '=':
                    breakdown['DRAM'] = float(current[-1])
                # elif current[0] == 'pe_spad' and current[1] == '=':
                #     breakdown['pe_spad'] = float(current[-1])
                elif current[0] != 'Total':
                    breakdown['other'] += float(current[-1])


        ret.append(breakdown)
    return ret

            




def generate_stacked_bar_graph(data, group_labels, subgroup_labels, colors, breakdown_labels, title):
    # Set the width of a bar and calculate the number of groups and subgroups

    bar_width = 0.8 / len(subgroup_labels)
    num_groups = len(group_labels)
    num_subgroups = len(subgroup_labels)

    # Calculate the positions of the bars on the x-axis
    positions = np.arange(num_groups) * 1
    actual_positions = list()

    # Create the bars and stack them for each subgroup
    # tmp_labels = ['16-bit baseline', '2-bit k-means', '4-bit k-means', '8-bit k-means']
    for i in range(num_subgroups):
        bottoms = [0] * num_groups
        actual_positions.append(positions + i * bar_width)
        for j in range(len(data[0][0])):
            plt.bar(
                positions + i * bar_width,
                [d[i][j] for d in data],
                width=bar_width,
                bottom=bottoms,
                color=colors[j],
                label=breakdown_labels[j] if i == 0 else None,
                edgecolor='black'
            )
            bottoms = [bottoms[k] + data[k][i][j] for k in range(num_groups)]
        # plt.bar(
        #     positions + i * bar_width,
        #     [d[i][0] for d in data],
        #     width=bar_width,
        #     bottom=bottoms,
        #     color=colors[i],
        #     label=tmp_labels[i]
        # )
        print(bottoms)
        plt.text(
            positions[1] + i*bar_width - bar_width/3,
            bottoms[1] + 3,
            subgroup_labels[i],
            rotation=70
        )
        bottoms = [bottoms[k] + data[k][i][0] for k in range(num_groups)]


    # Customize the appearance of the graph
    plt.xticks(positions + (num_subgroups - 1) * bar_width / 2, group_labels)

    # custom_positions = np.concatenate(actual_positions)
    # custom_labels = [s for s in subgroup_labels] * 8
    # print(custom_labels)
    # plt.xticks(custom_positions, custom_labels, rotation=90)
    # plt.ylim(ymax = 17) #87

    plt.xlabel('VGG01 Layer')
    plt.ylabel('Energy (pJ/Compute)')
    plt.title(title)
    plt.legend()

    # Display the graph
    plt.show()



def graph():
    arch_list = [
        '16bit_float_simple_weight_stationary',
        '2bit_kmeans_weight_stationary',
        '4bit_kmeans_weight_stationary',
        '8bit_kmeans_weight_stationary',
        '8bit_linear_weight_stationary'
    ]

    group_labels = list(range(1,3))
    subgroup_labels = ['16-bit float', '2-bit k-means', '4-bit k-means', '8-bit k-means', '8-bit linear']
    # subgroup_labels = ['16base', '2k', '4k', '8k']
    # colors = ['#FF5733', '#33FF57', '#3357FF']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',]
    breakdown_labels = ['MAC', 'DRAM', 'other']
    # breakdown_labels = ['other']
    # title = 'Energy is Dominated by mac and DRAM'
    title = 'Energy is Dominated by MAC and DRAM'

    full_list_data = list()
    for arch in arch_list:
        data = read_data(arch)
        list_data = [[d[label] for label in breakdown_labels] for d in data]
        full_list_data.append(list_data)

    # print(np.transpose(np.array(full_list_data), (1,0,2)).shape)
    # input()
    full_list_data = np.transpose(np.array(full_list_data), (1,0,2)).tolist()
    # print(full_list_data)

    generate_stacked_bar_graph(full_list_data, group_labels, subgroup_labels, colors, breakdown_labels, title)


if __name__ == '__main__':

 


    # read_data('2bit_kmeans_weight_stationary')

    # graph_arch('16bit_float_simple_weight_stationary')

    # read_data('8bit_linear_weight_stationary')

    graph()


    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']

    # print(colors)