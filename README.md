# 《复杂车间调度问题的混合整数规划求解方法》建模代码

> 本项目为论文中MIP建模代码，并使用串行的highs_1.6.0版本进行求解。


## 环境依赖 (Requirements)

项目代码基于windows系统，建议使用pycharm。环境依赖见"requirments.txt"。

highs.exe来自于官方维护团队预编译好的1.6.0 windows版本。地址：https://github.com/JuliaBinaryWrappers/HiGHSstatic_jll.jl/releases/download/HiGHSstatic-v1.6.0%2B0/HiGHSstatic.v1.6.0.x86_64-w64-mingw32.tar.gz

## 代码运行 (Installation)

* 先执行 benchmark_gen 文件夹中的 benchmarkGen.py 随机生成测试案例及数据
* 再执行main.py对CJSSP问题进行建模及求解

