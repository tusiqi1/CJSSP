import os
import random
import pandas
import pandas as pd
random.seed(1)

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
    def getMachineType(self):
        return self.__machine_type
    def getMachineNums(self):
        return self.__machine_nums

def generateJobsRandom(job_nums, each_job_operator_min, each_job_operator_max, fileId):
    machine = machineBase()

    machine_type = machine.getMachineType()  # [201, 203, 204, 205, 206, 207, 223, 227]
    machine_nums = machine.getMachineNums()  # [  3,   2,   3,   2,   2,   1,   1,   1]

    dir_path = os.getcwd()+"\\testSet{}".format(fileId)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_id = 0
    child_paraent_pair = set()

    # 定义pandas
    df = pd.DataFrame({'data_id': [],
                       'job_id': [],
                       'procedu_numb': [],
                       'task_code': [],
                       'act_time': [],
                       'paraent_job_id': [],
                        })

    operation_counts = {}
    operations_machines = {}
    machine2oper = {}
    for job_id in range(job_nums):
        # 随机选择有几个任务
        operation_nums = random.randint(each_job_operator_min, each_job_operator_max)
        operation_counts[job_id] = operation_nums
        # 0.1的概率有父job, 默认为-1表示没有
        paraent_job_id = -1
        random_number = random.random()
        if random_number < 0.1:
            paraent_job_id = random.randint(0, job_nums - 1)
            # 如果不存在(paraent_chlid，job_id), 则选择
            if ((paraent_job_id, job_id) not in child_paraent_pair) and paraent_job_id != job_id:
                child_paraent_pair.add((job_id, paraent_job_id))
            # 如果存在了，则取消
            else:
                paraent_job_id = -1

        # 生成指定的oper
        for procedu_numb in range(operation_nums):
            task_code = random.choice(machine_type)

            # 生成operations_machines 和 machine2oper
            operations_machines[(job_id, procedu_numb)] = list(machine.task_map_machineId[task_code])
            for machine_id in operations_machines[(job_id, procedu_numb)]:
                if machine2oper.__contains__(machine_id):
                    machine2oper[machine_id].append((job_id, procedu_numb))
                else:
                    machine2oper[machine_id] = [(job_id, procedu_numb)]


            act_time = random.randint(40,60)

            df.loc[len(df.index)] = [data_id, job_id, procedu_numb, task_code, act_time, paraent_job_id]
            data_id = data_id + 1

    df.to_excel('./testSet{}/info_job{}.xlsx'.format(fileId,fileId), index=False)

    gamma_FN = {}
    gamma_F = {}
    for i in range(job_nums):
        for j in range(operation_counts[i]):
            for k in operations_machines[(i, j)]:
                random_number = random.random()
                if random_number > 0.5:
                    gamma_FN[str((i, j, k))] = 1
                    gamma_F[str((i, j, k))] = random.randint(5,10)
                    # print(f"gama_F[{i},{j},{k}]:",gamma_F[(i, j, k)])
                else:
                    gamma_FN[str((i, j, k))] = 0
                    gamma_F[str((i, j, k))] = 0

    gamma_I = {}  # 任意两两工序之间的换模时间
    gamma_IN = {}  # 任意两两工序之间是否有换模时间
    for key in machine2oper.keys():
        machine_Operation = machine2oper[key]
        for i in range(len(machine_Operation)):
            ii = machine_Operation[i][0]
            jj = machine_Operation[i][1]
            for j in range(len(machine_Operation)):
                gg = machine_Operation[j][0]
                hh = machine_Operation[j][1]
                random_number = random.random()
                if random_number > 0.5:
                    gamma_IN[str((ii, jj, gg, hh, key))] = 1
                    gamma_I[str((ii, jj, gg, hh, key))] = random.randint(5,10)
                    # print(f"gama_I[{ii},{jj},{gg},{hh},{key}]", gamma_I[(ii, jj, gg, hh, key)])
                else:
                    gamma_IN[str((ii, jj, gg, hh, key))] = 0
                    gamma_I[str((ii, jj, gg, hh, key))] = 0

    mold_change = {}
    mold_change["gamma_FN"] = gamma_FN
    mold_change["gamma_F"] = gamma_F
    mold_change["gamma_I"] = gamma_I
    mold_change["gamma_IN"] = gamma_IN
    import json
    with open('./testSet{}/moldchg_job{}.json'.format(fileId,fileId), 'w') as json_file:
        json.dump(mold_change, json_file)

# generateJobsRandom(4, 2, 8, 0)

if __name__ == "__main__":
    generateJobsRandom(10, 4, 8, 0)