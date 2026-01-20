import pandas as pd
from mip import Model, BINARY, xsum, LinExpr
import matplotlib.pyplot as plt
from distinctipy import distinctipy
import os
import subprocess
import psutil
import copy
import random
random.seed(1)

max_time_all = [1200, 1200, 1200]
class machineBase:
    def __init__(self):
        self.__machine_type = [201, 203, 204, 205, 206, 207, 223, 227] # task_code
        self.__machine_nums = [  3,   2,   3,   2,   2,   1,   1,   1]
        #                      012   34   567   89   1011   12   13   14
        # 存储每个task_code对应的machine_id， 从0开始
        self.task_map_machineId = {}
        self.machine_count = -1
        task_count = -1
        for i in self.__machine_nums:
            task_count = task_count + 1
            temp = []
            for j in range(i):
                self.machine_count = self.machine_count + 1
                temp.append(self.machine_count)
            self.task_map_machineId[self.__machine_type[task_count]] = temp
        self.machine_count = self.machine_count + 1

def getFjsFile(original_excel, machine):
    """"
    此函数的功能是将读取的excel文件转化为fjs文件。
    excel文件的列名字为：data_id  craft_route_code  procedu_numb  task_code  prepare_time  act_time paraent_job_id
    输出jsp文件的格式为： jobId, paraent_job_id, operation_nums, [operatio[i]可以在几个机器上运行"machine_can_exe_oper_nums", 机器id, 运行时间]
    """
    # 存储最终的fjs文件
    fjs_output = []

    # 找出有多少个job及其对应的jobid
    job_ids = original_excel["job_id"].unique()
    # 针对每个jobid生成对应的fjs文件行，其中每行为：jobId, paraent_job_id, operation_nums, [operatio[i]可以在几个机器上运行"machine_can_exe_oper_nums", 机器id, 运行时间]
    for id in job_ids:
        job_row = []
        job_row.append(id)
        # 找出此任务的所有信息
        operations = original_excel[original_excel["job_id"] == id]
        operations = operations.reset_index(drop=True)
        operation_nums = operations.shape[0]

        # 找出paraent_job_id, 如果没有为-1
        job_row.append(operations["paraent_job_id"][0])
        job_row.append(operation_nums)

        # opertaions按procedu_numb进行排序
        operations = operations.sort_values("procedu_numb", ascending=True, inplace=False)  # ascending = true表示从小到大

        for row in range(operation_nums):
            task_code = operations["task_code"][row]
            machine_can_exe_oper_nums = len(machine.task_map_machineId[task_code])
            job_row.append(machine_can_exe_oper_nums)
            for machine_id_index in range(machine_can_exe_oper_nums):
                job_row.append(machine.task_map_machineId[task_code][machine_id_index])
                job_row.append(operations["act_time"][row])
        fjs_output.append(job_row)

    fjs_output.insert(0, [len(job_ids), machine.machine_count])
    return fjs_output

def generateMachinesAndTimes(benchmark_data):
    operations_machines = {} # 当前operator可以被哪些机器操作
    operations_times = {}
    machine2oper = {}
    for i in range(1,len(benchmark_data)):
        job_id = benchmark_data[i][0] # 与excel中job_id对应
        operator_id = 0 # 返回值中operator的标号从0开始
        j = 3 # 从下标3开始是 [operatio[i]可以在几个机器上运行"machine_can_exe_oper_nums", 机器id, 运行时间]
        while j < len(benchmark_data[i]): # 每个while循环处理一个operator的数据
            machine_count = benchmark_data[i][j] # 当前操作可以被machine_count个机器执行
            temp_machines = []
            for k in range(1,machine_count * 2 + 1,2):
                machine_id = benchmark_data[i][j+k]
                temp_machines.append(machine_id)
                operations_times[(job_id,operator_id,machine_id)] = benchmark_data[i][j + k + 1] # 当前operator被各个机器操作时所需的时间
                if machine2oper.__contains__(machine_id):
                    machine2oper[machine_id].append((job_id,operator_id))
                else:
                    machine2oper[machine_id] = [(job_id,operator_id)]
            operations_machines[(job_id,operator_id)] = temp_machines

            j = j + machine_count*2 + 1
            operator_id = operator_id + 1
    return operations_machines, operations_times, machine2oper

def solveMps(highs_path,mps_path,objective_id):
    print(os.path.exists("model{}.sol".format(objective_id)))
    if (os.path.exists("model{}.sol".format(objective_id))):
        os.remove("model{}.sol".format(objective_id))
    former_objective_id = objective_id - 1
    # --presolve=off
    max_time = max_time_all[objective_id]
    if former_objective_id < 0:
        args = highs_path + " --solution_file model{}.sol  --options_file=highs_options.txt  --time_limit {} ".format(objective_id, max_time) + mps_path # + "| tee -a big_log.txt"
    elif former_objective_id == 0:
        args = highs_path + " --solution_file model{}.sol  --options_file=highs_options.txt  --read_solution_file model{}.sol --time_limit {} ".format(objective_id, former_objective_id, max_time) + mps_path  # + "| tee -a big_log.txt"
    else:
        args = highs_path + " --solution_file model{}.sol  --options_file=highs_options.txt  --read_solution_file model{}.sol --time_limit {} ".format(objective_id, former_objective_id, max_time) + mps_path  # + "| tee -a big_log.txt"
    print(args)
    proc = subprocess.Popen(args, shell=False, close_fds=True)

    # proc = subprocess.Popen(args, shell=True)
    try:
        proc.wait(timeout=max_time + 1)
    except subprocess.TimeoutExpired:
        kill(proc.pid)
    print("跳出了solveMps")
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()
def readSol(solFilepath, start_time_min):
    # if not os.path.exists("model.sol"):
    #     assert False
    result_x = {}
    result_s = {}
    result_epsilon = {}
    count = 1
    objective = 0.0
    with open(solFilepath, "r") as f:
        for line in f.readlines():
            if line[0:2] == "x(":
                temp = line
                line = line[:-1]
                [line, value] = line.split(' ')
                value = int(float(value) +  0.5)
                start = line.find('(')
                end = line.find(')')
                index = line[start+1:end].split(',')
                index = [int(num) for num in index]
                if index[0] == 0 and index[1] == 0 and index[2] == 2:
                    a = 1
                result_x[index[0],index[1], index[2]] = value
            elif line[0:2] == "s(":
                [line, value] = line.split(' ')
                value = float(value)
                start = line.find('(')
                end = line.find(')')
                index = line[start + 1:end].split(',')
                index = [int(num) for num in index]
                result_s[index[0],index[1]] = value + start_time_min
            elif line[0:8] == "epsilon(":
                [line, value] = line.split(' ')
                value = float(value)
                start = line.find('(')
                end = line.find(')')
                index = line[start + 1:end].split(',')
                index = [int(num) for num in index]
                result_epsilon[index[0], index[1]] = value
            elif line[0:9] == "Objective":
                [line, value] = line.split(' ')
                objective = float(value)

            else:
                continue
        return result_x, result_s, result_epsilon, objective

def adjust_sol(objective_id, objective):
    # 修改sol文件，使highs可以读取初始解
    file_name = "model{}.sol".format(objective_id-1)
    # model0.sol文件加入Cmax约束
    # 找到# Dual solution values的行号
    # python 使用
    count = 0 # target_value在lines中的标号，从0开始
    target_value = "# Dual solution values"
    fp = open(file_name, 'r+', encoding='gbk')
    lines = []
    line_Cmax = []
    flag = True #标记是否找到了target_value行
    flag1 = False #标记是否找到了 # Rows 行
    flag2 = False #标记是否找到了Cmax行
    objFunction1_rowId = -1
    objFunction2_rowId = -1
    row_id = -1
    for line in fp:
        row_id = row_id + 1
        if line[0:6] == "# Rows":
            flag1 = True
        if line[:-1] == target_value:
            flag = False
        if flag:
            count = count + 1
        line = line.strip()
        if flag1:
            line = line[0:-len(line.split()[-1])] + str(int(line.split()[-1])+1)
            flag1 = False
        if line[:12] == "objFunction{}".format(objective_id - 1):
            objFunction1_rowId = row_id
        if line[:12] == "objFunction{}".format(objective_id):
            objFunction2_rowId = row_id
        lines.append(line)

    fp.close()
    lines[objFunction2_rowId], lines[objFunction1_rowId] = lines[objFunction1_rowId], lines[objFunction2_rowId]
    # a = 1
    # import re
    # p1 = re.compile(r'[(](.*?)[)]', re.S)
    # new_constr_id = int(re.findall(p1, lines[count-2].split()[0])[0]) + 1
    new_constr = "obj{} {}".format(objective_id-1, objective)
    lines.insert(count-1,new_constr)
    s = "\n".join(lines)
    fp = open('model{}.sol'.format(objective_id-1), 'w')
    fp.write(s)
    fp.close()

def getGamma(filename):
    import json
    with open(filename, 'r') as json_file:
        loaded_data = json.load(json_file)
    gamma_FN_str = loaded_data["gamma_FN"]
    gamma_F_str = loaded_data["gamma_F"]
    gamma_IN_str = loaded_data["gamma_IN"]
    gamma_I_str = loaded_data["gamma_I"]


    gamma_FN = {}
    gamma_F = {}
    gamma_IN = {}
    gamma_I = {}



    for key_str, value in gamma_FN_str.items():
        # 去除括号并使用逗号分割字符串键，创建元组键
        key_tuple = tuple(map(int, key_str.strip('()').split(',')))
        # 将新的键值对添加到新字典中
        gamma_FN[key_tuple] = value

    for key_str, value in gamma_F_str.items():
        # 去除括号并使用逗号分割字符串键，创建元组键
        key_tuple = tuple(map(int, key_str.strip('()').split(',')))
        # 将新的键值对添加到新字典中
        gamma_F[key_tuple] = value

    for key_str, value in gamma_IN_str.items():
        # 去除括号并使用逗号分割字符串键，创建元组键
        key_tuple = tuple(map(int, key_str.strip('()').split(',')))
        # 将新的键值对添加到新字典中
        gamma_IN[key_tuple] = value

    for key_str, value in gamma_I_str.items():
        # 去除括号并使用逗号分割字符串键，创建元组键
        key_tuple = tuple(map(int, key_str.strip('()').split(',')))
        # 将新的键值对添加到新字典中
        gamma_I[key_tuple] = value

    gamma = {}
    gamma["gamma_FN"] = gamma_FN
    gamma["gamma_F"] = gamma_F
    gamma["gamma_I"] = gamma_I
    gamma["gamma_IN"] = gamma_IN
    return gamma

def plot_gantt_high(x,s,epsilons, operations_times,jobs,fileId):
    colors = distinctipy.get_colors(len(jobs))
    for key in x.keys():
        if(x[key]>0.8):
            if key[0] in jobs:
                job_id = key[0]
                operation_id = key[1]
                machine_id = key[2]
                start_time = s[(job_id,operation_id)]
                length_time = operations_times[(job_id,operation_id,machine_id)]
                exchange_time = epsilons[job_id, operation_id]
                if length_time != 0:
                    plt.barh(y=(machine_id), left=start_time, width=length_time, edgecolor="black",
                             color='white')
                    if exchange_time != 0:
                        plt.barh(y=(machine_id), left=start_time, width=-exchange_time, edgecolor="black",
                                 color="black")
                    plt.text(x=start_time, y=machine_id-0.1, s="({},{})".format(job_id, operation_id),fontsize=9,color='black',fontweight='bold')

    plt.savefig("./result.svg",dpi=600)
    # plt.show()
    plt.close()
def saveSol(solution_file,solution):
    for i in solution:
        count = 0
        for j in i:
            count = count + 1
            solution_file.write(str(j))
            if count < len(i):
                solution_file.write(' ')
        solution_file.write('\n')
    solution_file.close()

def solve(benchmark_data,gamma,fileID, start_time = []):
    job_nums = benchmark_data[0][0]
    machine_nums = benchmark_data[0][1]
    objective_values = []
    operation_count = {}  # 存储每个任务需要的操作个数



    if len(start_time) == 0:
    # 如果考虑机器的开始时间不为0
        start_time = [0] * machine_nums
    start_time_min = min(start_time)
    start_time = [item - start_time_min for item in start_time]

    if benchmark_data == -1:
        return

    for i in range(1,len(benchmark_data)):
        operation_count[benchmark_data[i][0]] =  benchmark_data[i][2]

    operations_machines,operations_times,machine2oper = generateMachinesAndTimes(benchmark_data)


    """"
    随机生成换模时间、换模关系矩阵
    """

    gamma_FN = gamma["gamma_FN"]
    gamma_F = gamma["gamma_F"]
    gamma_I = gamma["gamma_I"]
    gamma_IN = gamma["gamma_IN"]
    # gamma_F = {}  # 任意工序在某个机器开始的换模时间
    # gamma_FN = {}  # 任意工序在某个机器开始是否要换模
    # for i in range(job_nums):
    #     for j in range(operation_count[i]):
    #         for k in operations_machines[(i, j)]:
    #             random_number = random.random()
    #             if random_number > 0.5:
    #                 gamma_FN[(i, j, k)] = 1
    #                 gamma_F[(i, j, k)] = 10
    #                 print(f"gama_F[{i},{j},{k}]:",gamma_F[(i, j, k)])
    #             else:
    #                 gamma_FN[(i, j, k)] = 0
    #                 gamma_F[(i, j, k)] = 0
    #
    # gamma_I = {}  # 任意两两工序之间的换模时间
    # gamma_IN = {}  # 任意两两工序之间是否有换模时间
    # for key in machine2oper.keys():
    #     machine_Operation = machine2oper[key]
    #     for i in range(len(machine_Operation)):
    #         ii = machine_Operation[i][0]
    #         jj = machine_Operation[i][1]
    #         for j in range(len(machine_Operation)):
    #             gg = machine_Operation[j][0]
    #             hh = machine_Operation[j][1]
    #             random_number = random.random()
    #             if random_number > 0.5:
    #                 gamma_IN[(ii, jj, gg, hh, key)] = 1
    #                 gamma_I[(ii, jj, gg, hh, key)] = 10
    #                 print(f"gama_I[{ii},{jj},{gg},{hh},{key}]",gamma_I[(ii, jj, gg, hh, key)])
    #             else:
    #                 gamma_IN[(ii, jj, gg, hh, key)] = 0
    #                 gamma_I[(ii, jj, gg, hh, key)] = 0
    """
    开始多目标求解
    """
    objexctive_nums = 3
    for objective_id in range(objexctive_nums):
        if objective_id != 0:
            m.clear()
        m = Model('FJSSP_{}'.format(objective_id),solver_name="CBC")
        Cmax = m.add_var(name="Cmax")

        x = {}  # 任务i的第j个操作是否在机器k上
        s = {} # 任务i的第j个操作的开始时间
        p = {} # 任务i的第j个操作的执行时间（不含换模时间）
        c = {} # 任务i的第j个操作的终止时间
        epsilon_hat = {} # 辅助变量，帮助表示任务i的第j个操作的换模时间
        epsilon_bar = {} # 辅助变量，帮助表示任务i的第j个操作的换模时间
        epsilon = {} # 任务i的第j个操作的换模时间

        for i in range(job_nums):
            for j in range(operation_count[i]):
                s[(i, j)] = m.add_var(name='s({},{})'.format(i, j))
                c[(i, j)] = m.add_var(name='c({},{})'.format(i, j))
                p[(i, j)] = m.add_var(name='p({},{})'.format(i, j))
                m.add_constr(s[(i, j)] >= 0)
                epsilon[(i, j)] = m.add_var(name='epsilon({},{})'.format(i, j))
                for k in operations_machines[(i, j)]:
                    x[(i, j, k)] = m.add_var(var_type=BINARY, name='x({},{},{})'.format(i, j, k))
                    epsilon_hat[(i, j, k)] = m.add_var(name='epsilon_hat({},{},{})'.format(i, j, k))
                    epsilon_bar[(i, j, k)] = m.add_var(name='epsilon_bar({},{},{})'.format(i, j, k))


        y = {}  # 0,1

        help_eq4_u = {} # 辅助变量，实现eq4右侧的max(右侧,0)功能
        help_eq4_v = {} # 辅助变量，实现eq4右侧的max(右侧,0)功能
        help_eq4 = {} # 辅助变量，实现eq4右侧的max(右侧,0)功能

        for key in machine2oper.keys():
            machine_Operation = machine2oper[key]
            help_eq4_u[key] = m.add_var(var_type=BINARY, name='help_eq4_u({})'.format(key))
            help_eq4_v[key] = m.add_var(var_type=BINARY, name='help_eq4_v({})'.format(key))
            help_eq4[key] = m.add_var(name='help_eq4({})'.format(key))
            for i in range(len(machine_Operation)):
                ii = machine_Operation[i][0]
                jj = machine_Operation[i][1]
                for j in range(len(machine_Operation)):
                    gg = machine_Operation[j][0]
                    hh = machine_Operation[j][1]
                    if ii == gg and jj == hh:
                        continue
                    y[(ii, jj, gg, hh, key)] = m.add_var(var_type=BINARY,
                                                        name='y({},{},{},{},{})'.format(ii, jj, gg, hh, key))
                    m.add_constr(y[(ii, jj, gg, hh, key)] <= x[(ii, jj, key)], name='eq2{}{}{}{}{}'.format(ii,jj,gg,hh,key))
                    m.add_constr(y[(ii, jj, gg, hh, key)] <= x[(gg, hh, key)], name='eq3{}{}{}{}{}'.format(ii,jj,gg,hh,key))


        L = 10000
        for k in machine2oper.keys():
            machine_Operation = machine2oper[k]

            constr = LinExpr()
            for i in range(len(machine_Operation)):
                ii = machine_Operation[i][0]
                jj = machine_Operation[i][1]
                for j in range(len(machine_Operation)):
                    gg = machine_Operation[j][0]
                    hh = machine_Operation[j][1]
                    if ii == gg and jj == hh:
                        continue
                    constr.add_term(y[(ii, jj, gg, hh, k)], coeff = 1)

            b = LinExpr()
            if constr.equals(b):
                continue

            m.add_constr(help_eq4[k] >= xsum(
                x[(machine_Operation[i][0], machine_Operation[i][1], k)] for i in range(len(machine_Operation))) - 1)
            m.add_constr(help_eq4[k] >= 0)

            m.add_constr(xsum(
                x[(machine_Operation[i][0], machine_Operation[i][1], k)] for i in range(len(machine_Operation))) - 1
                        >= help_eq4[k] - L * (1 - help_eq4_u[k]))
            m.add_constr(0 >= help_eq4[k] - L * (1 - help_eq4_v[k]))

            m.add_constr(help_eq4_u[k] + help_eq4_v[k] == 1)

            m.add_constr(constr == help_eq4[k])

        for k in machine2oper.keys():
            machine_Operation = machine2oper[k]
            for i in range(len(machine_Operation)):
                ii = machine_Operation[i][0]
                jj = machine_Operation[i][1]
                # 此处可能会出现无值的情况
                m.add_constr(xsum(y[(ii, jj, machine_Operation[j][0], machine_Operation[j][1], k)] \
                                        for j in range(len(machine_Operation)) if
                                        not (ii == machine_Operation[j][0] and jj == machine_Operation[j][1])) <= 1 ) #,name='eq5_{}_{}'.format(i,k))

                m.add_constr(xsum(y[(machine_Operation[j][0], machine_Operation[j][1], ii, jj, k)] \
                                        for j in range(len(machine_Operation)) if
                                        not (ii == machine_Operation[j][0] and jj == machine_Operation[j][1])) <= 1) #,name='eq6_{}_{}'.format(i,k))

        for i in range(job_nums):
            for j in range(operation_count[i]):
                m.add_constr(p[(i, j)] == xsum(
                    operations_times[(i, j, k)] * x[(i, j, k)] for k in operations_machines[(i, j)]),
                            name='eq7_{}{}'.format(i, j))
                m.add_constr(c[(i, j)] == s[(i, j)] + p[(i, j)]) #, name='eq8_{}{}'.format(i,j))

        for g in range(job_nums):
            for h in range(operation_count[g]):
                for k in operations_machines[(g, h)]:
                    m.add_constr(xsum(
                        y[(i[0], i[1], g, h, k)] * gamma_I[(i[0], i[1], g, h, k)] for i in machine2oper[k] if
                        not (i[0] == g and i[1] == h)) \
                                + (1 - xsum(
                        y[(i[0], i[1], g, h, k)] for i in machine2oper[k] if not (i[0] == g and i[1] == h))) \
                                * gamma_F[(g, h, k)] == epsilon_hat[(g, h, k)], name='eq9_{}_{}_{}'.format(g,h,k))

                    m.add_constr(epsilon_bar[(g, h, k)] <= L * x[(g, h, k)], name='eq10_{}_{}_{}'.format(g,h,k))
                    m.add_constr(epsilon_hat[(g, h, k)] - L * (1 - x[g, h, k]) <= epsilon_bar[(g, h, k)], name='eq11_left_{}_{}_{}'.format(g,h,k))
                    m.add_constr(epsilon_bar[(g, h, k)] <= epsilon_hat[(g, h, k)], name='eq11_right_{}_{}_{}'.format(g,h,k))

        for i in range(job_nums):
            for j in range(operation_count[i]):
                m.add_constr(epsilon[(i, j)] == xsum(epsilon_bar[(i, j, k)] for k in operations_machines[(i, j)])) #,name='eq12_{}_{}'.format(i,j))

        for i in range(job_nums):
            for j in range(operation_count[i]):
                for g in range(job_nums):
                    for h in range(operation_count[g]):
                        if i == g and j == h:
                            continue
                        # 判断两个list中是否有重复元素
                        intersection = [element for element in operations_machines[(i, j)] if
                                        element in operations_machines[(g, h)]]
                        if len(intersection) > 0:
                            m.add_constr(
                                c[(i, j)] - (1 - xsum(y[(i, j, g, h, k)] for k in intersection)) * L <= s[(g, h)] -
                                epsilon[(g, h)]) #, name='eq13_{}_{}_{}_{}'.format(i,j,g,h))

        for i in range(job_nums):
            for j in range(operation_count[i]):
                # print(operations_machines[(i,j)])
                m.add_constr(Cmax >= c[(i, j)]) #, name='eq14_{}{}'.format(i,j))
                m.add_constr(xsum(x[(i, j, k)] for k in operations_machines[(i, j)]) == 1) #,name='eq1_{}{}'.format(i, j))
                if j != 0:
                    # 路径上的时间要变
                    m.add_constr(s[(i, j)] - c[(i, j - 1)] >= 0) #, name='eq15_{}{}'.format(i,j))

        # 每台机器都有一个开始时间, 若x_ijk在机器k上，约束成立
        for i in range(job_nums):
            for j in range(operation_count[i]):
                for k in operations_machines[(i, j)]:
                    m.add_constr(s[(i, j)] - epsilon[(i, j)] >= start_time[k] - L * (1 - x[(i, j, k)])) #,name='eq16_{}{}{}'.format(i,j,k))
                    if start_time[k] != 0:
                        a =1

        # 考虑物料齐套性，通过order_information进行处理
        for i in range(job_nums):
            father_job_id = benchmark_data[i+1][1] # i+1跳过第一行
            child_job_id = benchmark_data[i+1][0]
            child_oper_count = operation_count[child_job_id]
            if father_job_id != -1:
                m.add_constr(s[(father_job_id, 0)] - c[(child_job_id, child_oper_count - 1)] >= 0)



        z = {}
        for g in range(job_nums):
            for h in range(operation_count[g]):
                for k in operations_machines[(g, h)]:
                    z[(g, h, k)] = m.add_var(name='z({},{},{})'.format(g, h, k))

                    m.add_constr(z[(g, h, k)] == 1 - xsum(
                        y[(i[0], i[1], g, h, k)] for i in machine2oper[k] if not (i[0] == g and i[1] == h)))
        # 如果z(i,j,k)和x(i,j,k)都为1，则表明i,j在k的第一道工序上，用乘法表示。下列表达式为整合之后的
        fi = {}
        for i in range(job_nums):
            for j in range(operation_count[i]):
                for k in operations_machines[(i, j)]:
                    fi[(i, j, k)] = m.add_var(name='fi({},{},{})'.format(i, j, k))
                    m.add_constr(fi[(i, j, k)] <= x[(i, j, k)])
                    m.add_constr(fi[(i, j, k)] <= z[(i, j, k)])
                    m.add_constr(fi[(i, j, k)] >= x[(i, j, k)] + z[(i, j, k)] - 1)

        obj0 = m.add_var(name="objFunction0")
        obj1 = m.add_var(name="objFunction1")
        obj2 = m.add_var(name="objFunction2")
        m.add_constr(obj0 == Cmax)
        m.add_constr(obj1 == xsum((c[(i, j)]) for i in range(job_nums) for j in range(operation_count[i])))
        m.add_constr(obj2 == xsum((y[(machine2oper[key][i][0], machine2oper[key][i][1], machine2oper[key][j][0], machine2oper[key][j][1], key)]) * \
                               gamma_IN[(machine2oper[key][i][0], machine2oper[key][i][1], machine2oper[key][j][0], machine2oper[key][j][1], key)] \
                               for key in machine2oper.keys() for i in range(len(machine2oper[key])) for j in range(len(machine2oper[key]))
                               if not (machine2oper[key][i][0]== machine2oper[key][j][0] and machine2oper[key][i][1]==machine2oper[key][j][1]))\
                                +xsum(fi[(i,j,k)] * gamma_FN[(i,j,k)] for i in range(job_nums) for j in range(operation_count[i]) for k in operations_machines[(i,j)]) \
                     )

        if objective_id == 0:
            m.objective = obj0
        elif objective_id == 1:
            m.add_constr(Cmax <= objective_values[0], name="obj0")
            m.objective = obj1
            adjust_sol(objective_id, objective_values[0])
        else:
            m.add_constr(Cmax <= objective_values[0], name="obj0")
            m.add_constr(xsum((c[(i, j)]) for i in range(job_nums) for j in range(operation_count[i])) <= objective_values[1], name="obj1")
            adjust_sol(objective_id, objective_values[1])
            m.objective = obj2
        if os.path.exists("model{}.mps.mps.gz".format(objective_id)):
            os.remove("model{}.mps.mps.gz".format(objective_id))
        if os.path.exists("model{}.mps.mps".format(objective_id)):
            os.remove("model{}.mps.mps".format(objective_id))
        if os.path.exists("model{}.mps".format(objective_id)):
            os.remove("model{}.mps".format(objective_id))
        if os.path.exists("model{}.lp".format(objective_id)):
            os.remove("model{}.lp".format(objective_id))


        m.write(file_path="model{}.lp".format(objective_id))
        # m.write(file_path="model{}.mps".format(objective_id))

        # with gzip.open('model{}.mps.mps.gz'.format(objective_id), 'rb') as f_in:
        #     # 使用 shutil 将文件解压到指定目录
        #     with open('model{}.mps'.format(objective_id), 'wb') as f_out:
        #         shutil.copyfileobj(f_in, f_out)



        # highs求解
        print(os.getcwd())
        highs_path = "./highs.exe"
        mps_path = os.getcwd() + "/" + "model{}.lp".format(objective_id)

        sol_path = os.getcwd() + "/" + "model{}.sol".format(objective_id)

        # 在当前路径highs生成的sol文件
        solveMps(highs_path, mps_path, objective_id)
        # 读取sol文件时也必须在当前路径
        xx = []
        ss = []
        epsilons = []
        [xx, ss, epsilons, objective] = readSol(sol_path, start_time_min)
        filename = "test2.sol"
        basename = os.path.basename(filename)
        dirname = os.path.dirname(filename)
        end_time = copy.deepcopy(start_time)
        # print("\n", "Completion time Cmax: ----------------------------", c.x)

        # 文件保存路径处理

        solution_path = dirname + "/" + "solution"
        folder = os.path.exists(solution_path)
        if not folder:
            os.makedirs(solution_path)
        solution_file = open(solution_path + "/" + basename[:-4] + ".sol", 'w')
        solution = [["bill_numb", "matril_code", "craft_route_code", 'paraent_bill_numb','paraent_matril_code','small_pack_id','small_pack_count','operation_count',"operation_code", "machine_code", "start_time", "act_time", "exchange_time"]]
        for i in range(job_nums):
            for j in range(operation_count[i]):
                exe_flag = False
                for k in operations_machines[(i, j)]:
                    if (xx[i, j, k] > 0.8):
                        exe_flag=True
                        print("任务{}的第{}个操作在机器{}执行，开始执行时间为:{}，执行时间为：{}".format(i, j, k, ss[(i, j)],operations_times[(i, j, k)]), end = " ")
                        print("换模时间为：", epsilons[(i, j)])
                        # temp = []
                        # for pp in order_information[i]:
                        #     temp.append(pp)
                        # temp.append(j)  # 没有从j映射到task_id
                        # temp.append(k)
                        # temp.append(ss[(i, j)])
                        # temp.append(operations_times[(i, j, k)])
                        # temp.append(epsilons[(i, j)])
                        # solution.append(temp)

                        if ss[(i, j)] + operations_times[(i, j, k)] > end_time[k]:
                            end_time[k] = ss[(i, j)] + operations_times[(i, j, k)]
                if exe_flag == False:
                    print("错误")
                    input()

        print("机器个数：", len(end_time))
        print("makespan", max(end_time))
        objective_values.append(objective)
        # print("目标函数值为：",m.objective.x)

        if len(objective_values) == objexctive_nums:
            plot_gantt_high(xx, ss, epsilons, operations_times, range(job_nums), fileID)
            saveSol(solution_file, solution)
            end_time = [item + start_time_min for item in end_time]
            return end_time

def copyResult(fileID):
    import shutil
    import os
    dest_path = "./benchmark_gen/testSet{}".format(fileID)
    src_path = ['HiGHS.log', 'model0.lp', 'model0.sol', 'model1.lp', 'model1.sol', 'model2.lp', 'model2.sol', 'result.svg']
    for i in src_path:
        if os.path.exists(dest_path + "/{}".format(i)):
            os.remove(dest_path + "/{}".format(i))
        shutil.move(i, dest_path)


def main(fileID):
    original_excel = pd.read_excel("./benchmark_gen/testSet{}/info_job{}.xlsx".format(fileID,fileID))
    machine = machineBase()
    fjs_list = getFjsFile(original_excel, machine)
    gamma = getGamma("./benchmark_gen/testSet{}/moldchg_job{}.json".format(fileID,fileID))
    solve(fjs_list, gamma=gamma, fileID=fileID)
    copyResult(fileID)


if __name__ == "__main__":
    main(0)
