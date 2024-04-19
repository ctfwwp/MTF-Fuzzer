# MTF-Fuzzer
模糊测试（fuzzing）- MTF-Fuzzer

一个模糊测试工具

TRM.py 训练模型和生成测试用例的文件

check_tcp.py 判断Modbus TCP协议是否符合协议规范

select.py 选择变异字节

self_adaption 字节变异概率调整

注意，在TRM.py中我们引用了modbus_tk库。我们更改了modbus_tk\modbus.py文件。请将文件中的modbus.py与modbus_tk\modbus.py替换


TRM.py - Training model and generating test cases file

check_tcp.py - Checking if Modbus TCP protocol complies with protocol specifications

select.py - Selecting mutated bytes

self_adaption - Adjusting byte mutation probability

Note that in TRM.py, we referenced the modbus_tk library. We modified the modbus_tk\modbus.py file. Please replace the modbus.py in the file with modbus_tk\modbus.py.

