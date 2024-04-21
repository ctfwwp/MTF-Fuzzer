# MTF-Fuzzer
模糊测试（fuzzing）- MTF-Fuzzer

一个模糊测试工具

TRM.py 训练模型和生成测试用例的文件

check_tcp.py 判断Modbus TCP协议是否符合协议规范。以下表格展示了不同功能码的基本协议规范，随着协议规范的详细程度增加，检查出的错误也会增加

| 功能码 | 规范 |
|-------|-------|
| 0x01  | 协议总字节数为12Byte   |
| 0x02  | 协议总字节数为12Byte  |
| 0x03  | 协议总字节数为12Byte  |
| 0x04  | 协议总字节数为12Byte  |
| 0x05  | 协议总字节数为12Byte; 最后两个字节为“FF00”或者“0000”  |
| 0x06  | 协议总字节数为12Byte   |
| 0x10  | 协议总字节数大于13Byte（小于则表明没有寄存器的值）; 写入的寄存器个数与寄存器的值的数量相等；字节长度等于寄存器的值的字节数|
| 0x0F  | 协议总字节数大于14Byte（小与则表明没有输出值）;字节长度等于输出的字节数  |

select.py 选择变异字节

self_adaption 字节变异概率调整

注意，在TRM.py中我们引用了modbus_tk库。我们更改了modbus_tk\modbus.py文件。请将文件中的modbus.py与modbus_tk\modbus.py替换

TRM.py - Training model and generating test cases file

check_tcp.py - Checking if Modbus TCP protocol complies with protocol specifications

select.py - Selecting mutated bytes

self_adaption - Adjusting byte mutation probability

Note that in TRM.py, we referenced the modbus_tk library. We modified the modbus_tk\modbus.py file. Please replace the modbus.py in the file with modbus_tk\modbus.py.

other fuzzer:

PSM Fuzzer: 

  https://github.com/ctfwwp/PSM-Fuzzer
  
Fuzzy-RNN：

  https://github.com/ctfwwp/Fuzzy-RNN
  
Boofuzz

  Boofuzz 的项目地址：
  
    https://github.com/jtpereyda/boofuzz
    
  Boofuzz 对Modbus TCP协议进行模糊测试的项目地址：
  
    https://github.com/youngcraft/boofuzz-modbus
    
Peach Fuzzer

  Peach 的项目地址：
  
    https://github.com/MozillaSecurity/peach
    
    https://github.com/TideSec/Peach_Fuzzing
    
  Peach 对Modbus TCP协议进行模糊测试的项目地址：
  
    https://github.com/jseidl/peach-pit
    
    https://github.com/uknowsec/ModbusPeachPit

