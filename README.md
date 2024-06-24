# ConvAccelerate 信息

本项目包含一些基本的卷积加速算法，感兴趣的可以尝试下载运行。

本项目可能存在错误和不足，欢迎指教。

本项目借鉴和使用了项目[convolution-flavors](https://github.com/gplhegde/convolution-flavors)的结构，增加了新的算法。感兴趣的也可以去看看这个项目。



# 环境

> openBlas
> m
> g++
> gcc

安装Openblas库可以参考博客 [Openblas安装方法](https://forcheetah.github.io/2024/05/15/openBlas/)


# 运行

1. main.c中关掉想要测试的函数注释

```c
    //1.  测试 weight_col matmul input_tensor_col
    TestIm2FlavorConvLayer();

    //2. 测试 input_tensor_col matmul weight_col
    // TestIm2ColConvIMW();

    //3.  测试 weight_cube matmul input_cube  逐个取数
    // TestIm2CubeConvLayer();

    //4.  测试 image -> col_2D -> cube with multi-mem-copy  批量取数
    // TestIm2MencopyConvLayer();

    //5.  测试 image -> col_2D -> cube with multi-mem-copy  批量取数  NHWC 版本
    // TestIm2BatchcopyConvLayer();

    //6.  测试 winograd alogrithm  F(2*2, 3*3)  
    // TestWinogradF23ConvLayer();

    //7.  测试 winograd alogrithm  F(6*6, 3*3)  
    // TestWinogradF63ConvLayer();

    //8.  测试 NHWC 排布 CPU卷积算法的正确性
    // TestCPU_NHWC_conv();

    //9.  测试 昇腾 卷积算法加速     NCHW 输入， NHWC输出
    // TestAscendConvLayer();

    //10.  测试 昇腾 卷积算法加速      NCHW 输入， NCHW输出
    // TestAscendConvLayerNCHW();

    //11. 测试 昇腾卷积算法加速 NHWC      NHWC 输入， NHWC输出
    // TestAscendConvLayerNHWC();
```

2. 修改 CMakeLists.txt文件中 openBlas 库的路径

3. 编译运行

```bash
    mkdir build && cd build
    cmake ..
    make
    ./alogrithm
```



