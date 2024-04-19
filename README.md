# MTF-Fuzzer
模糊测试（fuzzing）- MTF-Fuzzer

一个模糊测试工具

TRM.py 训练模型和生成测试用例的文件

check_tcp.py 判断Modbus TCP协议是否符合协议规范

注意，在TRM.py中我们引用了modbus_tk库。我们更改了modbus_tk\modbus.py文件。请将文件中的modbus.py与modbus_tk\modbus.py替换

在check_tcp.py文件中，我们给出判断测试用例是否符合Modbus TCP协议规范的代码


